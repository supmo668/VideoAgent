# main.py
import os
import click
import uuid
import shutil
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from utils import load_config
from db_utils import init_cache_db, get_cached_embedding, save_embedding_to_cache
from video_utils import extract_frames
from embed_utils import cosine_similarity, get_text_embedding_openai, get_frame_description
from embedding_utils import (
    OpenAIEmbeddingProcessor,
    ClipEmbeddingProcessor,
    process_frames_for_question,
    save_and_report_results,
)

def setup_processing_environment(
    video_path, keep_temp_dir=True, suffix=""
    ):
    """Set up the processing environment including temporary and results directories."""
    video_name = Path(video_path).stem
    temp_dir = Path(video_path).parent / f"temp_frames-{video_name}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("results") / f"{video_name}_{timestamp}{suffix}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    return temp_dir, result_dir

def process_video_with_processor(
    processor,
    frames,
    questions,
    result_dir,
    record_top_k_frames,
    model_type
):
    """Process video frames with the given processor for each question."""
    for q in tqdm(questions, desc=f"Processing questions with {model_type.upper()}"):
        similarities = process_frames_for_question(frames, q, processor)
        save_and_report_results(
            similarities=similarities,
            question=q,
            result_dir=result_dir,
            record_top_k_frames=record_top_k_frames,
            model_type=model_type
        )

def cleanup_environment(temp_dir, keep_temp_dir):
    """Clean up temporary directory if needed."""
    if not keep_temp_dir:
        shutil.rmtree(temp_dir)

@click.group()
def cli():
    pass

@cli.command("openai-embed")
@click.option("--video_path", type=str, required=True, help="Path to input video.")
@click.option("--question", type=str, required=True, multiple=True, help="A list of descriptions or questions.")
@click.option("--sample_freq", type=int, default=30, help="Sampling frequency (e.g., every nth frame)")
@click.option("--config_path", type=str, default="config.yaml", help="Path to config YAML.")
@click.option("--cache_db_path", type=str, default="embeddings_cache.db", help="Path to SQLite cache database.")
@click.option("--keep_temp_dir", is_flag=True, default=True, help="Keep the temporary directory with extracted frames.")
@click.option("--record_top_k_frames", type=int, default=20, help="Number of top frames to record in results.")
def process_video_openai(
    video_path, question, sample_freq, config_path, 
    cache_db_path, keep_temp_dir,
    record_top_k_frames
    ):
    cfg = load_config(config_path)
    system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
    vision_prompt_template = cfg.get("vision_prompt", "Analyze this image in detail: {image_path} and {description}")

    # Initialize cache DB
    init_cache_db(cache_db_path)

    try:
        # Set up environment
        temp_dir, result_dir = setup_processing_environment(video_path, keep_temp_dir)
        
        # Extract frames and initialize processor
        frames = extract_frames(video_path, sample_freq, temp_dir)
        processor = OpenAIEmbeddingProcessor(system_prompt, vision_prompt_template, cache_db_path)

        # Process video
        process_video_with_processor(
            processor=processor,
            frames=frames,
            questions=question,
            result_dir=result_dir,
            record_top_k_frames=record_top_k_frames,
            model_type="openai"
        )
    finally:
        cleanup_environment(temp_dir, keep_temp_dir)

@cli.command("clip-embed")
@click.option("--video_path", type=str, required=True, help="Path to input video.")
@click.option("--question", type=str, required=True, multiple=True, help="A list of descriptions or questions.")
@click.option("--sample_freq", type=int, default=30, help="Sampling frequency (e.g., every nth frame)")
@click.option("--keep_temp_dir", is_flag=True, default=True, help="Keep the temporary directory with extracted frames.")
@click.option("--record_top_k_frames", type=int, default=20, help="Number of top frames to record in results.")
def process_video_clip(
    video_path, question, sample_freq,
    record_top_k_frames, keep_temp_dir
    ):
    """
    This command uses a local CLIP model to directly compute embeddings
    for both frames and the given questions.
    """
    from clip_utils import get_clip_text_embedding, get_clip_image_embedding
    
    try:
        # Set up environment
        temp_dir, result_dir = setup_processing_environment(
            video_path, 
            keep_temp_dir,
            suffix="_clip"
        )
        
        # Extract frames and initialize processor
        frames = extract_frames(video_path, sample_freq, temp_dir)
        processor = ClipEmbeddingProcessor()

        # Process video
        process_video_with_processor(
            processor=processor,
            frames=frames,
            questions=question,
            result_dir=result_dir,
            record_top_k_frames=record_top_k_frames,
            model_type="clip"
        )
    finally:
        cleanup_environment(temp_dir, keep_temp_dir)

if __name__ == "__main__":
    cli()
    # run openai with
    # python main.py openai-embed --video_path data/V1_end.mp4  --question "Putting red cabbage solution into test tube for the second time"
    # run clip with
    # python main.py clip-embed --video_path data/V1_end.mp4 --question "Pouring water into red cabbage filled beaker" --question "Turning on heat plate" --question "Putting red cabbage solution into test tube (first time)" --question "Putting red cabbage solution into test tube (second time)"
