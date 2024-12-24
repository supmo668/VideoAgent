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
from typing import List, Dict, Tuple, Optional, Any, Union
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
from dotenv import load_dotenv

from s3 import S3Module
from models import VideoAnalysisRequest, VideoAnalysisResponse

# Load environment variables from .env file
load_dotenv()

# Get environment variables
AWS_REGION = os.getenv('AWS_REGION', 'us-east-2')
BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'myimagebucketlabar')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

if not all([AWS_REGION, BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY]):
    raise ValueError("Missing required AWS environment variables. Please check your .env file.")

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

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_video_with_processor(
    processor: Union[ClipEmbeddingProcessor, OpenAIEmbeddingProcessor],
    frames: List[Tuple[int, str]],
    questions: List[str],
    cfg: Dict[str, Any],
    result_dir: Path,
    record_top_k_frames: int,
    model_type: str,
    generate_report: bool
) -> Optional[Dict[str, str]]:
    """Process video frames with the given processor for each question."""
    vp = VideoProcessor()
    report_data: List[Dict[str, str]] = []
    key_frames: Dict[str, Tuple[int, str]] = {}

    # Get frames directory from the first frame path
    frames_directory = str(Path(frames[0][1]).parent) if frames else ""

    for q in tqdm(questions, desc=f"Processing questions with {model_type.upper()}"):
        similarities: List[Tuple[str, float]] = process_frames_for_question(frames, q, processor)
        if not similarities:
            print(f"No relevant frames found for question '{q}'.")
            continue

        save_and_report_results(
            similarities, q, result_dir,
            record_top_k_frames=record_top_k_frames,
            model_type=model_type
        )
        if generate_report and similarities:
            top_frame_path, sim, frame_idx = similarities[0]
            frame_description = get_frame_description(
                cfg["system_prompt"], 
                cfg["vision_prompt"], 
                top_frame_path
            )

            object_name = f"reports/{Path(top_frame_path).name}"
            vp._upload_to_s3(top_frame_path, object_name)
            s3_url = vp._get_s3_url(object_name)

            # Get relative path for markdown, keeping original path for file operations
            try:
                relative_path = str(Path(top_frame_path).relative_to(result_dir))
            except ValueError:
                # If paths are not relative, use the frame path relative to its parent
                relative_path = str(Path(top_frame_path).relative_to(Path(top_frame_path).parent.parent))

            report_data.append(
                {
                    "frame_path": relative_path,
                    "description": frame_description,
                    "image_path": str(top_frame_path),  # Keep absolute path for file operations
                    "s3_url": s3_url
                }
            )
            key_frames[q] = (frame_idx, s3_url)

    if generate_report:
        # Generate local report with local image paths
        local_report_path = result_dir / "workflow_report_local.md"
        local_reports = log_to_markdown_report(
            local_report_path, 
            report_data, 
            cfg['report_template'],
            use_s3_urls=False
        )

        # Generate S3 report with S3 URLs
        s3_report_path = result_dir / "workflow_report.md"
        s3_reports = log_to_markdown_report(
            s3_report_path, 
            report_data, 
            cfg['report_template'],
            use_s3_urls=True
        )

        response_key_frames = {k: v[1] for k, v in key_frames.items()}

        return VideoAnalysisResponse(
            local_markdown=local_reports['markdown'],
            local_html=local_reports['html'],
            local_pdf=local_reports['pdf'],
            s3_markdown=s3_reports['markdown'],
            s3_html=s3_reports['html'],
            s3_pdf=s3_reports['pdf'],
            key_frames=response_key_frames,
            result_dir=str(result_dir),
            frames_dir=frames_directory
        ).dict()

    return None

def log_to_markdown_report(
    report_path: str,
    report_data: List[Dict[str, str]],
    template: str,
    use_s3_urls: bool = False
) -> Dict[str, str]:
    """
    Log descriptions to a markdown report using the provided template.
    Can generate reports with either local image paths or S3 URLs.
    """
    # Create markdown report
    markdown_content = template + "\n\n"
    for item in report_data:
        image_path = item['s3_url'] if use_s3_urls else item['image_path']
        markdown_content += f"### Frame: {item['frame_path']}\n\n"
        markdown_content += f"![Frame]({image_path})\n\n"
        markdown_content += f"Description: {item['description']}\n\n"
        markdown_content += "---\n\n"

    # Save markdown report
    with open(report_path, 'w') as f:
        f.write(markdown_content)

    # Generate HTML version
    html_path = str(Path(report_path).with_suffix('.html'))
    html_content = markdown_content  # In a real app, convert markdown to HTML
    with open(html_path, 'w') as f:
        f.write(html_content)

    # Generate PDF version
    pdf_path = str(Path(report_path).with_suffix('.pdf'))
    generate_pdf_report(pdf_path, report_data)

    return {
        'markdown': str(report_path),
        'html': html_path,
        'pdf': pdf_path
    }

def generate_pdf_report(output_path: str, report_data: List[Dict[str, str]]) -> None:
    """Generate a PDF report from the report data."""
    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()
    elements = []

    for item in report_data:
        elements.append(Paragraph(f"Frame: {item['frame_path']}", styles['Heading3']))
        elements.append(Paragraph(f"Description: {item['description']}", styles['Normal']))
        if os.path.exists(item['image_path']):
            elements.append(Image(item['image_path'], width=400, height=300))
    doc.build(elements)

def process_video_clip_core(
    video_path: str,
    questions: List[str],
    sample_freq: int = 30,
    config_path: str = "config.yaml",
    cache_db_path: str = "embeddings_cache.db",
    keep_temp_dir: bool = True,
    record_top_k_frames: int = 20,
    generate_report: bool = True
) -> Optional[Dict[str, str]]:
    """Core function to process video using CLIP embedding model."""
    cfg = load_config(config_path)
    try:
        request = VideoAnalysisRequest(
            video_path=video_path,
            questions=list(questions),
            model_type="clip",
            sample_freq=sample_freq,
            record_top_k_frames=record_top_k_frames,
            generate_report=generate_report
        )

        temp_dir, result_dir = setup_processing_environment(
            request.video_path, cache_db_path, keep_temp_dir, suffix="_clip")
        frames: List[Tuple[int, str]] = extract_frames(request.video_path, request.sample_freq, temp_dir)
        processor = ClipEmbeddingProcessor(cache_db_path)

        results: Optional[Dict[str, str]] = process_video_with_processor(
            processor=processor,
            frames=frames,
            questions=request.questions,
            cfg=cfg,
            result_dir=result_dir,
            record_top_k_frames=request.record_top_k_frames,
            model_type=request.model_type,
            generate_report=request.generate_report
        )
        if results:
            print(f"Reports saved to {results['local_html']} and {results['local_pdf']}.")
        return results
    finally:
        cleanup_environment(temp_dir, keep_temp_dir)

def process_video_openai_core(
    video_path: str,
    questions: List[str],
    sample_freq: int = 30,
    config_path: str = "config.yaml",
    cache_db_path: str = "embeddings_cache.db",
    keep_temp_dir: bool = True,
    record_top_k_frames: int = 20,
    generate_report: bool = True
) -> Optional[Dict[str, str]]:
    """Core function to process video using OpenAI embedding model."""
    cfg = load_config(config_path)
    try:
        request = VideoAnalysisRequest(
            video_path=video_path,
            questions=list(questions),
            model_type="openai",
            sample_freq=sample_freq,
            record_top_k_frames=record_top_k_frames,
            generate_report=generate_report
        )

        temp_dir, result_dir = setup_processing_environment(
            request.video_path, cache_db_path, keep_temp_dir, suffix="_openai")
        frames: List[Tuple[int, str]] = extract_frames(request.video_path, request.sample_freq, temp_dir)
        processor = OpenAIEmbeddingProcessor(
            cfg["system_prompt"], cfg["vision_prompt"], cache_db_path)

        results: Optional[Dict[str, str]] = process_video_with_processor(
            processor=processor,
            frames=frames,
            questions=request.questions,
            cfg=cfg,
            result_dir=result_dir,
            record_top_k_frames=request.record_top_k_frames,
            model_type=request.model_type,
            generate_report=request.generate_report
        )
        if results:
            print(f"Reports saved to {results['local_html']} and {results['local_pdf']}.")
        return results
    finally:
        cleanup_environment(temp_dir, keep_temp_dir)

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
    """CLI command to process video using OpenAI embedding model."""
    return process_video_openai_core(
        video_path=video_path,
        questions=list(question),
        sample_freq=sample_freq,
        config_path=config_path,
        cache_db_path=cache_db_path,
        keep_temp_dir=keep_temp_dir,
        record_top_k_frames=record_top_k_frames,
        generate_report=generate_report
    )

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
    cache_db_path, keep_temp_dir,
    record_top_k_frames, generate_report
):
    """CLI command to process video using CLIP embedding model."""
    return process_video_clip_core(
        video_path=video_path,
        questions=list(question),
        sample_freq=sample_freq,
        config_path=config_path,
        cache_db_path=cache_db_path,
        keep_temp_dir=keep_temp_dir,
        record_top_k_frames=record_top_k_frames,
        generate_report=generate_report
    )

if __name__ == "__main__":
    cli()
    # run openai with
    # python main.py openai-embed --video_path ../data/V1_end.mp4 --question "Pouring water into red cabbage filled beaker" --question "Turning on heat plate" --question "Putting red cabbage solution into test tube (first time)" --question "Putting red cabbage solution into test tube (second time)"
    # run clip with
    # python main.py clip-embed --video_path ../data/V1_end.mp4 --question "Pouring water into red cabbage filled beaker" --question "Turning on heat plate" --question "Putting red cabbage solution into test tube (first time)" --question "Putting red cabbage solution into test tube (second time)" --generate-report
