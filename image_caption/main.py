# main.py
import os
import click
import uuid
import shutil
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
from typing import List, Dict, Tuple
import openai
from urllib.request import urlretrieve

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
    video_path: str, cache_db_path: str, keep_temp_dir: bool = True, suffix: str = ""
) -> Tuple[Path, Path]:
    """Set up the processing environment including temporary and results directories."""
    
    # Initialize cache DB
    init_cache_db(cache_db_path)

    # Assume video_path is an HTTP URL
    video_name = Path(video_path).stem
    temp_dir = Path(__file__).parent.parent / f"temp_frames-{video_name}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download video from HTTP URL
    if video_path.startswith("http"):
        video_name = Path(video_path).stem
        local_video_path = temp_dir / video_name
        if not local_video_path.exists():
            urlretrieve(video_path, str(local_video_path))
            video_path = str(local_video_path)
    
    # Create result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).parent.parent / "results" / f"{video_name}_{timestamp}{suffix}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    return temp_dir, result_dir

def process_video_with_processor(
    processor,
    frames,
    questions,
    cfg,
    result_dir,
    record_top_k_frames,
    model_type,
    generate_report
):
    """Process video frames with the given processor for each question."""
    for q in tqdm(questions, desc=f"Processing questions with {model_type.upper()}"):
        similarities: List[Tuple[str, float]] = process_frames_for_question(frames, q, processor)
        key_frame_path, top_results = save_and_report_results(
            similarities=similarities,
            question=q,
            result_dir=result_dir,
            record_top_k_frames=record_top_k_frames,
            model_type=model_type
        )
        
        # Generate report if flag is set
        if generate_report:
            report_data: List[Dict[str, str]] = []
            # Process only the top frame for each question
            if similarities:
                top_frame_path, _ = similarities[0]  # Assuming similarities is sorted by relevance
                # Generate frame description for the top frame
                frame_description = get_frame_description(
                    cfg["system_prompt"], 
                    cfg["vision_prompt"], 
                    top_frame_path
                )
                print(f"Report for frame {top_frame_path}: {frame_description}")
                key_frame_path_rel = Path(key_frame_path).relative_to(result_dir)
                report_data.append(
                    {"frame_path": str(key_frame_path_rel), "description": frame_description})
            
    # Log to markdown report
    log_to_markdown_report(
        result_dir / "workflow_report.md", report_data, cfg['report_template'])

def cleanup_environment(temp_dir, keep_temp_dir):
    """Clean up temporary directory if needed."""
    if not keep_temp_dir:
        shutil.rmtree(temp_dir)

def log_to_markdown_report(report_path: str, descriptions: List[Dict[str, str]], template: str):
    """
    Log descriptions to a markdown report using the provided template.
    """
    keyframe_descriptions = ""
    for desc in descriptions:
        keyframe_descriptions += f"### Frame: {desc['frame_path']}\n"
        image_name = Path(desc['frame_path']).name.replace(' ', '%20')
        keyframe_descriptions += f"![Frame Image]({image_name})\n"
        keyframe_descriptions += f"Description: {desc['description']}\n\n"

    report_content = template.replace("{{keyframe_descriptions}}", keyframe_descriptions)
    with open(report_path, 'w') as report_file:
        report_file.write(report_content)

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
@click.option("--generate-report", is_flag=True, help="Generate a detailed report for each keyframe.")
def process_video_openai(
    video_path, question, sample_freq, config_path, 
    cache_db_path, keep_temp_dir,
    record_top_k_frames, generate_report
):
    cfg = load_config(config_path)
    try:
        # Set up environment
        temp_dir, result_dir = setup_processing_environment(
            video_path, cache_db_path, keep_temp_dir)
        
        # Extract frames and initialize processor
        frames = extract_frames(video_path, sample_freq, temp_dir)
        processor = OpenAIEmbeddingProcessor(
            cfg["system_prompt"], cfg["vision_prompt"], cache_db_path)

        # Process video
        process_video_with_processor(
            processor=processor,
            frames=frames,
            questions=question,
            cfg=cfg,
            result_dir=result_dir,
            record_top_k_frames=record_top_k_frames,
            model_type="openai",
            generate_report=generate_report
        )
    finally:
        cleanup_environment(temp_dir, keep_temp_dir)

@cli.command("clip-embed")
@click.option("--video_path", type=str, required=True, help="Path to input video.")
@click.option("--question", type=str, required=True, multiple=True, help="A list of descriptions or questions.")
@click.option("--sample_freq", type=int, default=30, help="Sampling frequency (e.g., every nth frame)")
@click.option("--config_path", type=str, default="config.yaml", help="Path to config YAML.")
@click.option("--cache_db_path", type=str, default="embeddings_cache.db", help="Path to SQLite cache database.")
@click.option("--keep_temp_dir", is_flag=True, default=True, help="Keep the temporary directory with extracted frames.")
@click.option("--record_top_k_frames", type=int, default=20, help="Number of top frames to record in results.")
@click.option("--generate-report", is_flag=True, help="Generate a detailed report for each keyframe.")
def process_video_clip(
    video_path, question, sample_freq, config_path,
    cache_db_path, record_top_k_frames, keep_temp_dir, generate_report
    ):
    """
    This command uses a local CLIP model to directly compute embeddings
    for both frames and the given questions.
    """ 
    cfg = load_config(config_path)
    try:
        # Set up environment
        temp_dir, result_dir = setup_processing_environment(
            video_path, cache_db_path, keep_temp_dir, suffix="_clip"
        )
        
        # Extract frames and initialize processor
        frames = extract_frames(video_path, sample_freq, temp_dir)
        processor = ClipEmbeddingProcessor(cache_db_path)

        # Process video
        process_video_with_processor(
            processor=processor,
            frames=frames,
            questions=question,
            cfg=cfg,
            result_dir=result_dir,
            record_top_k_frames=record_top_k_frames,
            model_type="clip",
            generate_report=generate_report
        )
    finally:
        cleanup_environment(temp_dir, keep_temp_dir)

if __name__ == "__main__":
    cli()
    # run openai with
    # python main.py openai-embed --video_path ../data/V1_end.mp4 --question "Pouring water into red cabbage filled beaker" --question "Turning on heat plate" --question "Putting red cabbage solution into test tube (first time)" --question "Putting red cabbage solution into test tube (second time)"
    # run clip with
    # python main.py clip-embed --video_path data/V1_end.mp4 --question "Pouring water into red cabbage filled beaker" --question "Turning on heat plate" --question "Putting red cabbage solution into test tube (first time)" --question "Putting red cabbage solution into test tube (second time)"
