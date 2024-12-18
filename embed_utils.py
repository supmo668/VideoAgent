# main.py
import os
import click
import uuid
import shutil
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from db_utils import init_cache_db, get_cached_embedding, save_embedding_to_cache
from video_utils import extract_frames
from utils import load_config
from embed_utils import cosine_similarity, get_text_embedding_openai, get_frame_description
from clip_utils import get_clip_text_embedding, get_clip_image_embedding

@click.group()
def cli():
    pass

@cli.command("openai-embed")
@click.option("--video_path", type=str, required=True, help="Path to input video.")
@click.option("--question", type=str, required=True, multiple=True, help="A list of descriptions or questions.")
@click.option("--sample_freq", type=int, default=30, help="Sampling frequency (e.g., every nth frame)")
@click.option("--config_path", type=str, default="config.yaml", help="Path to config YAML.")
@click.option("--save_top_frame", type=str, default="top_frame.png", help="Path to save the most relevant frame.")
@click.option("--cache_db_path", type=str, default="embeddings_cache.db", help="Path to SQLite cache database.")
@click.option("--keep_temp_dir", is_flag=True, default=True, help="Keep the temporary directory with extracted frames.")
@click.option("--record_top_k_frames", type=int, default=20, help="Number of top frames to record in results.")
def process_video_openai(
    video_path, question, sample_freq, config_path, 
    save_top_frame, cache_db_path, keep_temp_dir,
    record_top_k_frames
    ):

    cfg = load_config(config_path)
    system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
    vision_prompt_template = cfg.get("vision_prompt", "Analyze this image in detail: {image_path} and {description}")

    # Initialize cache DB
    init_cache_db(cache_db_path)

    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("results") / f"{video_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = f"temp_frames-{video_name}-{uuid.uuid4().hex}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        frames = extract_frames(video_path, sample_freq, temp_dir)

        for q in tqdm(question, desc="Processing questions"):
            # Get question embedding (OpenAI)
            question_embedding = get_text_embedding_openai(q)

            similarities = []
            for frame_number, fp in tqdm(frames, desc=f"Processing frames for question: {q}"):
                # Check cache for existing embedding
                cached_embedding = get_cached_embedding(cache_db_path, video_path, frame_number, q)
                if cached_embedding is None:
                    # If not in cache, generate description and then get embedding
                    frame_description = get_frame_description(system_prompt, vision_prompt_template, fp, q)
                    frame_embedding = get_text_embedding_openai(frame_description)
                    # Save to cache
                    save_embedding_to_cache(cache_db_path, video_path, frame_number, q, frame_embedding)
                else:
                    frame_embedding = cached_embedding

                sim = cosine_similarity(question_embedding, frame_embedding)
                similarities.append((fp, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)

            # Save top frame and top `record_top_k_frames` results for each question
            top_frame_path = result_dir / save_top_frame
            if not top_frame_path.exists():
                shutil.copy(similarities[0][0], top_frame_path)

            top_results = [{"frame_path": f, "similarity": s} for f, s in similarities[:record_top_k_frames]]
            results_path = result_dir / f"results_{q}.json"
            with open(results_path, 'w') as f:
                json.dump({"question": q, "top_results": top_results}, f, indent=2)

            # Print results
            print(f"Top ranked frames for question '{q}' (path, similarity):")
            for result in top_results:
                print(f"{result['frame_path']}: {result['similarity']:.4f}")

            print(f"Most relevant frame for question '{q}' saved at {top_frame_path}")
            print(f"Results for question '{q}' saved to {results_path}")

    finally:
        if not keep_temp_dir:
            shutil.rmtree(temp_dir)


@cli.command("clip-embed")
@click.option("--video_path", type=str, required=True, help="Path to input video.")
@click.option("--question", type=str, required=True, multiple=True, help="A list of descriptions or questions.")
@click.option("--sample_freq", type=int, default=30, help="Sampling frequency (e.g., every nth frame)")
@click.option("--save_top_frame", type=str, default="top_frame_clip.png", help="Path to save the most relevant frame.")
@click.option("--keep_temp_dir", is_flag=True, default=True, help="Keep the temporary directory with extracted frames.")
@click.option("--record_top_k_frames", type=int, default=20, help="Number of top frames to record in results.")
def process_video_clip(
    video_path, question, sample_freq,
    save_top_frame, keep_temp_dir,
    record_top_k_frames
    ):
    """
    This command uses a local CLIP model to directly compute embeddings
    for both frames and the given questions.
    """

    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("results") / f"{video_name}_{timestamp}_clip"
    result_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = f"temp_frames-{video_name}-{uuid.uuid4().hex}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        frames = extract_frames(video_path, sample_freq, temp_dir)

        for q in tqdm(question, desc="Processing questions with CLIP"):
            # Compute CLIP text embedding
            question_embedding = get_clip_text_embedding(q)

            similarities = []
            for frame_number, fp in tqdm(frames, desc=f"Processing frames for question: {q}"):
                frame_embedding = get_clip_image_embedding(fp)
                sim = cosine_similarity(question_embedding, frame_embedding)
                similarities.append((fp, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)

            # Save top frame and top `record_top_k_frames` results for each question
            top_frame_path = result_dir / save_top_frame
            if not top_frame_path.exists():
                shutil.copy(similarities[0][0], top_frame_path)

            top_results = [{"frame_path": f, "similarity": s} for f, s in similarities[:record_top_k_frames]]
            results_path = result_dir / f"results_{q}_clip.json"
            with open(results_path, 'w') as f:
                json.dump({"question": q, "top_results": top_results}, f, indent=2)

            # Print results
            print(f"Top ranked frames for question '{q}' (CLIP-based) (path, similarity):")
            for result in top_results:
                print(f"{result['frame_path']}: {result['similarity']:.4f}")

            print(f"Most relevant frame for question '{q}' (CLIP-based) saved at {top_frame_path}")
            print(f"Results for question '{q}' saved to {results_path}")

    finally:
        if not keep_temp_dir:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    cli()
