import os
import click
import torch
import cv2
import requests
from PIL import Image
from openai import OpenAI
from transformers import SamModel, SamProcessor
from models import ImageActionFrame
import yaml
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
import datetime
import json

load_dotenv()
############################################################
# Configuration and Setup
############################################################

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

SAM_MODEL_NAME = config['sam_model_name']
LLM_PROMPT_TEMPLATE = config['llm_prompt_template']
DEFAULT_NUM_FRAMES = config['default_num_frames']
DEFAULT_TOP_K_SAM_ENTITIES = config['default_top_k_sam_entities']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize OpenAI client
client = OpenAI()

############################################################
# Helper Functions
############################################################

def load_sam_model():
    """
    Load the SAM model and processor from the Hugging Face Hub.
    """
    model = SamModel.from_pretrained(SAM_MODEL_NAME).to(DEVICE)
    processor = SamProcessor.from_pretrained(SAM_MODEL_NAME)
    return model, processor

def get_keyframes(video_path, num_frames=5):
    """
    Extract key frames from a video at evenly spaced intervals.
    If you have a CLIP-based keyframe selection method, integrate it here.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        raise ValueError("Video has no frames.")

    interval = max(frame_count // num_frames, 1)
    frames = []
    for i in range(num_frames):
        frame_index = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to PIL
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame)
        frames.append(pil_frame)
    cap.release()

    if not frames:
        raise ValueError("No frames were extracted from the video. Check the input video.")
    return frames

def run_sam_on_frames(model, processor, frames: List[Any], question: str, input_points: List[List[List[int]]] = None) -> List[Dict[str, Any]]:
    """
    Run SAM on a list of PIL image frames using a guiding question for segmentation.
    If input_points is None, we'll pick a central point as a simple prompt.
    """
    results = []
    for frame in frames:
        if input_points is None:
            w, h = frame.size
            # Use the question to determine input points or as a prompt (example)
            # This is a placeholder for more complex logic
            pts = [[[w//2, h//2]]]  # Example: central point
        else:
            pts = input_points

        # Optionally use the question in the processor logic
        inputs = processor(frame, input_points=pts, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        # Post-process masks
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Extract scores - handle the nested structure
        scores = outputs.iou_scores.cpu()
        if len(scores.shape) > 1:
            scores = scores[0]  # Take first batch
        scores = scores.tolist()
        
        # Each frame can have multiple masks; store them
        frame_result = {
            "masks": masks,   # list of BoolTensors (one per detected segment)
            "scores": scores,  # list of IoU scores corresponding to each mask
            "ious": scores  # list of IoU scores corresponding to each mask
        }
        results.append(frame_result)
    return results

def summarize_sam_output(sam_results, top_k_sam_entities=3):
    """
    Summarize the SAM output into a structured string.
    Report how many segments were found in each frame, their average IoU,
    and also select the top K segments across all frames by IoU score.
    """
    summary_lines = []
    all_segments = []  # We'll store (frame_idx, segment_idx, score) for all segments

    for i, frame_data in enumerate(sam_results):
        num_masks: int = len(frame_data["masks"])
        if num_masks > 0:
            # Ensure scores are flat list of floats
            scores = frame_data["scores"]
            if isinstance(scores[0], (list, torch.Tensor)):
                scores = [float(s) for s in scores[0]]  # Take first batch if nested
            else:
                scores = [float(s) for s in scores]
            
            avg_score = sum(scores) / num_masks
        else:
            avg_score = 0.0
        summary_lines.append(
            f"Frame {i+1}: Detected {num_masks} segments with average IoU {avg_score:.2f}."
        )

        # Store segments for later ranking
        for seg_idx, score in enumerate(scores if num_masks > 0 else []):
            all_segments.append((i+1, seg_idx+1, score))  # 1-based indexing for readability

    # Sort all segments by score descending
    all_segments.sort(key=lambda x: x[2], reverse=True)

    # Take top K segments
    top_segments = all_segments[:top_k_sam_entities]
    if top_segments:
        summary_lines.append("\nTop recognized segments:")
        for frame_idx, seg_idx, score in top_segments:
            summary_lines.append(f"  - Frame {frame_idx}, Segment {seg_idx} with IoU {score:.2f}")
    else:
        summary_lines.append("No segments found to highlight.")

    summary = "\n".join(summary_lines)
    return summary

def create_prompt_from_summary(summary, image_path, format_instruction=""):
    """
    Given a summary of SAM outputs and an image path, format the prompt for the LLM.
    """
    return LLM_PROMPT_TEMPLATE.format(
        summary=summary, image=image_path, format_instruction=format_instruction)

def get_lab_action_description_from_LLM(prompt: str) -> str:
    """
    Query GPT-4 to analyze the laboratory action.
    """
    system_prompt = config['system_prompt']
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()


############################################################
# Main CLI and Workflow
############################################################

@click.command()
@click.option("--video-path", required=True, type=click.Path(exists=True), help="Path to the input laboratory video.")
@click.option("--num-frames", default=DEFAULT_NUM_FRAMES, show_default=True, help="Number of frames to sample from the video.")
@click.option("--top-k-sam-entities", default=DEFAULT_TOP_K_SAM_ENTITIES, show_default=True, help="Number of top SAM segments to highlight.")
@click.option("--question", required=True, type=str, help="Input question to guide the analysis.")
def main(video_path: str, num_frames: int, top_k_sam_entities: int, question: str) -> None:
    # Step 1: Extract key frames from the video
    frames: List[Any] = get_keyframes(video_path, num_frames=num_frames)

    # Step 2: Load SAM model and processor
    model, processor = load_sam_model()

    # Step 3: Run SAM on frames
    sam_results: List[Dict[str, Any]] = run_sam_on_frames(model, processor, frames, question)

    # Step 4: Select key frame based on highest average IoU
    key_frame_index: int = max(
        range(len(sam_results)), 
        key=lambda i: sum(sum(ious) for ious in sam_results[i]['ious']) / len(sam_results[i]['ious'])
    )
    key_frame_results: Dict[str, Any] = sam_results[key_frame_index]

    # Step 5: Summarize SAM output for the key frame
    summary: str = summarize_sam_output(
        [key_frame_results], top_k_sam_entities=top_k_sam_entities)

    # Step 6: Create a prompt for the LLM
    prompt: str = create_prompt_from_summary(summary, video_path, format_instruction=question)

    # Step 7: Query the LLM 
    lab_action: str = get_lab_action_description_from_LLM(prompt)
    print("\nLikely Laboratory Action:\n", lab_action)

    # Document outputs
    results_dir = config['results_directory']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_results_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_results_dir, exist_ok=True)

    # Save key frame image
    key_frame_path = os.path.join(run_results_dir, f"key_frame_{key_frame_index}.png")
    frames[key_frame_index].save(key_frame_path)

    # Save results to JSON
    result_data = {
        "key_frame": key_frame_path,
        "sam_summary": summary,
        "final_output": lab_action
    }
    result_json_path = os.path.join(run_results_dir, "result.json")
    with open(result_json_path, "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    print(f"Results saved to: {run_results_dir}")


if __name__ == "__main__":
    main()
    # cli
    # python main.py --video-path ../data/V1_end.mp4 --num-frames 5 --top-k-sam-entities 3 --question "Pouring water into red cabbage filled beaker"
    # python main.py --video_path data/V1_end.mp4 --question "Pouring water into red cabbage filled beaker" --question "Turning on heat plate" --question "Putting red cabbage solution into test tube (first time)" --question "Putting red cabbage solution into test tube (second time)"
