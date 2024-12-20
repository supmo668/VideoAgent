import os
import click
import torch
import cv2
import requests
from PIL import Image
from openai import OpenAI
from transformers import SamModel, SamProcessor
from config import SAM_MODEL_NAME, LLM_PROMPT_TEMPLATE, DEFAULT_NUM_FRAMES, DEFAULT_TOP_K_SAM_ENTITIES

############################################################
# Configuration and Setup
############################################################

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

def run_sam_on_frames(model, processor, frames, input_points=None):
    """
    Run SAM on a list of PIL image frames.
    If input_points is None, we'll pick a central point as a simple prompt.
    In practice, adapt this according to your segmentation prompt needs.
    """
    results = []
    for frame in frames:
        if input_points is None:
            w, h = frame.size
            # Use a central point as a prompt (example)
            pts = [[[w//2, h//2]]]
        else:
            pts = input_points

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
            "scores": scores  # list of IoU scores corresponding to each mask
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

def create_prompt_from_summary(summary):
    """
    Given a summary of SAM outputs, create a prompt to ask an LLM 
    about the likely laboratory action.
    """
    return LLM_PROMPT_TEMPLATE.format(summary=summary)

def get_lab_action_description_from_LLM(prompt: str) -> str:
    """
    Query GPT-4 to analyze the laboratory action.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in laboratory procedures, specializing in analyzing and describing laboratory actions from visual data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return "Failed to analyze laboratory action"

############################################################
# Main CLI and Workflow
############################################################

@click.command()
@click.option("--video-path", required=True, type=click.Path(exists=True), help="Path to the input laboratory video.")
@click.option("--num-frames", default=DEFAULT_NUM_FRAMES, show_default=True, help="Number of frames to sample from the video.")
@click.option("--top-k-sam-entities", default=DEFAULT_TOP_K_SAM_ENTITIES, show_default=True, help="Number of top SAM segments to highlight.")
def main(video_path, num_frames, top_k_sam_entities):
    # Step 1: Extract key frames from the video
    frames = get_keyframes(video_path, num_frames=num_frames)

    # Step 2: Load SAM model and processor
    model, processor = load_sam_model()

    # Step 3: Run SAM on frames
    sam_results = run_sam_on_frames(model, processor, frames)

    # Step 4: Summarize SAM output
    summary = summarize_sam_output(sam_results, top_k_sam_entities=top_k_sam_entities)
    print("SAM Summary:\n", summary)

    # Step 5: Create a prompt for the LLM
    prompt = create_prompt_from_summary(summary)

    # Step 6: Query the LLM (mocked in this example)
    lab_action = get_lab_action_description_from_LLM(prompt)
    print("\nLikely Laboratory Action:\n", lab_action)


if __name__ == "__main__":
    main()
    # cli
    # python main.py --video-path ../data/V1_end.mp4 --num-frames 5 --top-k-sam-entities 3
