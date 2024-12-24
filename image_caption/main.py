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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet

from utils import load_config, setup_processing_environment, cleanup_environment
from db_utils import init_cache_db, get_cached_embedding, save_embedding_to_cache
from video_utils import extract_frames
from embed_utils import cosine_similarity, get_text_embedding_openai, get_frame_description
from embedding_utils import (
    OpenAIEmbeddingProcessor,
    ClipEmbeddingProcessor,
    process_frames_for_question,
    save_and_report_results,
)
from s3 import S3Module
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS credentials from environment variables
AWS_REGION = os.getenv('AWS_REGION', 'us-east-2')
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Validate AWS credentials
if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY]):
    print("Warning: AWS credentials are not fully configured. S3 operations may fail.")

class VideoProcessor:
    def __init__(self):
        self.s3 = S3Module(
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )

    def _upload_to_s3(self, image_path: str, object_name: str) -> bool:
        """Upload image to S3 bucket"""
        return self.s3.upload_file(image_path, BUCKET_NAME, object_name, acl='public-read')

    def _get_s3_url(self, object_name: str) -> str:
        """Get public URL for uploaded S3 object"""
        return f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{object_name}"

def process_video_with_processor(
    processor,
    frames,
    questions,
    cfg,
    result_dir,
    record_top_k_frames,
    model_type,
    generate_report
) -> Dict[str, str]:
    """Process video frames with the given processor for each question."""
    vp = VideoProcessor()
    report_data: List[Dict[str, str]] = []
    key_frames = {}

    for q in tqdm(questions, desc=f"Processing questions with {model_type.upper()}"):
        similarities: List[Tuple[str, float]] = process_frames_for_question(frames, q, processor)
        if not similarities:
            print(f"No relevant frames found for question '{q}'.")
            continue
        key_frame_path, top_results = save_and_report_results(
            similarities=similarities,
            question=q,
            result_dir=result_dir,
            record_top_k_frames=record_top_k_frames,
            model_type=model_type
        )
        if generate_report:
            if similarities:
                top_frame_path, sim, frame_idex = similarities[0]
                frame_description = get_frame_description(
                    cfg["system_prompt"], 
                    cfg["vision_prompt"], 
                    top_frame_path
                )
                object_name = f"reports/{Path(top_frame_path).name}"
                vp._upload_to_s3(top_frame_path, object_name)
                s3_url = vp._get_s3_url(object_name)

                report_data.append(
                    {
                        "frame_path": str(Path(key_frame_path).relative_to(result_dir)),
                        "description": frame_description,
                        "image_path": str(top_frame_path),
                        "s3_url": s3_url
                    }
                )
                key_frames[q] = s3_url

    if generate_report:
        local_report_path = result_dir / "workflow_report_local.md"
        html_report_path = log_to_markdown_report(
            local_report_path, report_data, cfg['report_template']
        )

        s3_report_path = result_dir / "workflow_report.md"
        log_to_s3_markdown_report(s3_report_path, report_data, cfg['report_template'])

        pdf_report_path = local_report_path.with_suffix('.pdf')
        generate_pdf_report(pdf_report_path, report_data)

        return {
            "local_html": str(html_report_path),
            "local_pdf": str(pdf_report_path),
            "s3_markdown": str(s3_report_path),
            "key_frames": key_frames
        }

def log_to_markdown_report(report_path: str, report_data: List[Dict[str, str]], template: str):
    keyframe_descriptions = ""
    for desc in report_data:
        keyframe_descriptions += f"### Frame: {desc['frame_path']}\n"
        image_name = Path(desc['frame_path']).name.replace(' ', '%20')
        keyframe_descriptions += f"![Frame Image]({image_name})\n"
        keyframe_descriptions += f"Description: {desc['description']}\n\n"

    report_content = template.replace("{{keyframe_descriptions}}", keyframe_descriptions)
    with open(report_path, 'w') as report_file:
        report_file.write(report_content)

    html_report_path = report_path.with_suffix('.html')
    with open(report_path, 'r') as md_file, open(html_report_path, 'w') as html_file:
        import markdown
        html_content = markdown.markdown(md_file.read())
        html_file.write(html_content)

    return str(html_report_path)

def log_to_s3_markdown_report(report_path: str, report_data: List[Dict[str, str]], template: str):
    keyframe_descriptions = ""
    for desc in report_data:
        keyframe_descriptions += f"### Frame: {desc['frame_path']}\n"
        keyframe_descriptions += f"![Frame Image]({desc['s3_url']})\n"
        keyframe_descriptions += f"Description: {desc['description']}\n\n"

    report_content = template.replace("{{keyframe_descriptions}}", keyframe_descriptions)
    with open(report_path, 'w') as report_file:
        report_file.write(report_content)

def generate_pdf_report(output_path, report_data):
    doc = SimpleDocTemplate(str(output_path))
    styles = getSampleStyleSheet()
    elements = []

    for item in report_data:
        elements.append(Paragraph(item['description'], styles['BodyText']))
        if 'image_path' in item:
            elements.append(Image(item['image_path'], width=400, height=300))
    doc.build(elements)

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
        temp_dir, result_dir = setup_processing_environment(
            video_path, cache_db_path, keep_temp_dir)
        frames = extract_frames(video_path, sample_freq, temp_dir)
        processor = OpenAIEmbeddingProcessor(
            cfg["system_prompt"], cfg["vision_prompt"], cache_db_path)

        report_paths = process_video_with_processor(
            processor=processor,
            frames=frames,
            questions=question,
            cfg=cfg,
            result_dir=result_dir,
            record_top_k_frames=record_top_k_frames,
            model_type="openai",
            generate_report=generate_report
        )
        if report_paths:
            print(f"Reports saved to {report_paths['local_html']} and {report_paths['local_pdf']}.")
        return report_paths
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
    cfg = load_config(config_path)
    try:
        temp_dir, result_dir = setup_processing_environment(
            video_path, cache_db_path, keep_temp_dir, suffix="_clip"
        )
        frames = extract_frames(video_path, sample_freq, temp_dir)
        processor = ClipEmbeddingProcessor(cache_db_path)

        report_paths = process_video_with_processor(
            processor=processor,
            frames=frames,
            questions=question,
            cfg=cfg,
            result_dir=result_dir,
            record_top_k_frames=record_top_k_frames,
            model_type="clip",
            generate_report=generate_report
        )
        if report_paths:
            print(f"Reports saved to {report_paths['local_html']} and {report_paths['local_pdf']}.")
        return report_paths
    finally:
        cleanup_environment(temp_dir, keep_temp_dir)

if __name__ == "__main__":
    cli()
    # run openai with
    # python main.py openai-embed --video_path ../data/V1_end.mp4 --question "Pouring water into red cabbage filled beaker" --question "Turning on heat plate" --question "Putting red cabbage solution into test tube (first time)" --question "Putting red cabbage solution into test tube (second time)"
    # run clip with
    # python main.py clip-embed --video_path ../data/V1_end.mp4 --question "Pouring water into red cabbage filled beaker" --question "Turning on heat plate" --question "Putting red cabbage solution into test tube (first time)" --question "Putting red cabbage solution into test tube (second time)" --generate-report
