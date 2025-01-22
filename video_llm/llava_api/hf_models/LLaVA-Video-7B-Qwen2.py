# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava_custom.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
# from llava.mm_utils import get_model_name_from_path, process_images, 
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import click
from transformers import BitsAndBytesConfig
import torch.cuda
import os
from typing import List, Tuple, Optional, Any, Union

warnings.filterwarnings("ignore")

def load_video(
    video_path: str, 
    max_frames_num: int,
    fps: int = 1,
    force_sample: bool = False
) -> Tuple[np.ndarray, List[float], float]:
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames_num = len(vr)
    frame_time = []
    if force_sample:
        frame_idx = np.linspace(0, total_frames_num - 1, max_frames_num, dtype=int)
        frame_time = frame_idx / vr.get_avg_fps()
        video_time = total_frames_num / vr.get_avg_fps()
    else:
        frame_idx = []
        frame_time = []
        for i in range(0, total_frames_num, int(vr.get_avg_fps() / fps)):
            frame_idx.append(i)
            frame_time.append(i / vr.get_avg_fps())
            if len(frame_idx) == max_frames_num:
                break
        video_time = total_frames_num / vr.get_avg_fps()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames,frame_time,video_time

def process_frames_in_chunks(
    frames: np.ndarray, 
    image_processor: Any, 
    device: str, 
    chunk_size: int = 8
) -> torch.Tensor:
    """Process video frames in chunks to save memory."""
    processed_chunks = []
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i + chunk_size]
        # Process chunk
        processed = image_processor.preprocess(chunk, return_tensors="pt")["pixel_values"]
        processed = processed.to(device=device, dtype=torch.float16)  # Match model dtype
        if len(processed.shape) == 3:
            processed = processed.unsqueeze(0)
        processed_chunks.append(processed)
        torch.cuda.empty_cache()
    
    result = torch.cat(processed_chunks, dim=0)
    print(f"Processed video shape: {result.shape}")
    return result

@click.command()
@click.option('--video-path', default="201243_65592_Johnston_080624_P_Web.mp4",
              help='Path to video file or video URL')
@click.option('--prompt', default=None,
              help='Custom prompt for video analysis. If not provided, a default description prompt will be used.')
def main(
    video_path: str, 
    prompt: Optional[str]
) -> Optional[str]:
    """Process video and generate AI description using LLaVA model."""
    try:
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        
        print("Loading model...")
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained, None, model_name, 
            torch_dtype=torch.float16,
            device_map=device_map,
            load_4bit=True
        )
        
        # Convert vision tower to float16
        if hasattr(model, 'get_vision_tower'):
            vision_tower = model.get_vision_tower()
            if hasattr(vision_tower, 'vision_tower'):
                vision_tower.vision_tower = vision_tower.vision_tower.to(dtype=torch.float16)
        
        model.eval()
        
        print(f"Processing video from: {video_path}")
        max_frames_num = 16
        video, frame_time, video_time = load_video(
            video_path, max_frames_num, 1, force_sample=True)
        
        print(f"Raw video shape: {video.shape}")
        
        processed_video = process_frames_in_chunks(video, image_processor, device)
        print(f"Final video tensor shape: {processed_video.shape}")
        
        image_sizes = [[video.shape[1], video.shape[2]] for _ in range(len(video))]
        print(f"Image size: {image_sizes[0]}. Num images: {len(image_sizes)}")
        
        torch.cuda.empty_cache()
        
        conv_template = "qwen_1_5"
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video)} frames are uniformly sampled from it. These frames are located at {frame_time}."
        
        if prompt is None:
            prompt = f"Please describe this video in detail."
        
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(
            conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        print("Generating response...")
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        input_ids = input_ids.to(device)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[processed_video],
                attention_mask=attention_mask,
                modalities=["video"],
                do_sample=False,
                temperature=0.2,
                max_new_tokens=4096,
            )
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print("\nResponse:\n", outputs)
        
        torch.cuda.empty_cache()
        return outputs
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    response = main()
    with open("response.txt", "w") as f:
        f.write(response)
