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
    process_frames_for_comparison,
    save_and_report_results,
)
from summarization_utils import app, summarize_and_generate_title_async
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

class S3Processor:
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

async def process_and_summarize_video(
    video_path: str,
    fps: int = 60,
    keep_temp_dir: bool = False,
    config_path: str = "config.yaml",
    cache_db_path: str = "embeddings_cache.db"
) -> Optional[dict]:
    """
    Process a video by extracting frames, getting frame descriptions, and summarizing the content.

    Args:
        video_path (str): Path to the input video file.
        fps (float): Frames per second to extract from the video.
        config_path (str): Path to the configuration file.
        cache_db_path (str): Path to the embeddings cache database.

    Returns:
        Optional[dict]: A dictionary containing the summary and title, or None if processing fails.
    """
    temp_dir, result_dir = setup_processing_environment(
        video_path, cache_db_path, keep_temp_dir, suffix="_summary")
    try:
        # Load configuration
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)
        frames: List[Tuple[int, str]] = extract_frames(video_path, temp_dir, fps)

        # Initialize OpenAIEmbeddingProcessor
        processor = OpenAIEmbeddingProcessor(
            cfg["system_prompt"], cfg["vision_prompt"], cache_db_path
        )

        # Get frame descriptions
        frame_descriptions: List[str] = []
        for _, frame_path in frames:
            description = processor.get_frame_description(frame_path)
            frame_descriptions.append(description)

        # Summarize and generate title
        result = await summarize_and_generate_title_async(frame_descriptions)

        return {
            "summary": result["abstract"],
            "title": result["title"]
        }

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
def process_video_with_processor(
    processor: Union[ClipEmbeddingProcessor, OpenAIEmbeddingProcessor],
    frames: List[Tuple[int, str]],
    user_descs: List[str],
    cfg: Dict[str, Any],
    result_dir: Path,
    record_top_k_frames: int,
    model_type: str,
    generate_report: bool
) -> Optional[Dict[str, str]]:
    """Process video frames with the given processor for each descriptions."""
    vp = S3Processor()
    report_data: List[Dict[str, Any]] = []  # Updated type hint to Any for flexibility
    key_frames: Dict[str, Tuple[int, str]] = {}
    frame_descriptions: Dict[str, str] = {}

    s3_report_folder = f"reports/{result_dir.name}_{uuid.uuid4().hex[:8]}"

    # Get frames directory from the first frame path
    frames_directory = str(Path(frames[0][1]).parent) if frames else ""

    try:
        for q in tqdm(user_descs, desc=f"Processing descriptions with {model_type.upper()}"):
            similarities: List[Tuple[str, float]] = process_frames_for_comparison(frames, q, processor)
            if not similarities:
                print(f"No relevant frames found for description '{q}'.")
                continue

            save_and_report_results(
                similarities, q, result_dir,
                record_top_k_frames=record_top_k_frames,
                model_type=model_type
            )
            if generate_report and similarities:
                top_frame_path, sim, frame_idx = similarities[0]
                
                # Get frame description using the processor if it's OpenAI, otherwise use default
                if isinstance(processor, OpenAIEmbeddingProcessor):
                    frame_description = processor.get_frame_description(top_frame_path)
                else:
                    frame_description = get_frame_description(
                        cfg["system_prompt"], cfg["vision_prompt"], top_frame_path
                    )
                
                # Store frame description
                frame_descriptions[q] = frame_description
                
                # Upload frame to S3 under the unique run folder
                object_name = f"{s3_report_folder}/frames/{Path(top_frame_path).name}"
                vp._upload_to_s3(top_frame_path, object_name)
                s3_url = vp._get_s3_url(object_name)

                # Get relative path for markdown, keeping original path for file operations
                try:
                    relative_path = str(Path(top_frame_path).relative_to(result_dir))
                except ValueError:
                    # If paths are not relative, use the frame path relative to its parent
                    relative_path = str(Path(top_frame_path).relative_to(Path(top_frame_path).parent.parent))

                report_entry = {
                    "description": q,
                    "frame_path": relative_path,
                    "frame_description": frame_description,
                    "image_path": str(top_frame_path),
                    "s3_url": s3_url,
                    "similarity_score": float(sim),  # Convert numpy float to native float
                    "frame_number": int(frame_idx)
                }
                report_data.append(report_entry)
                key_frames[q] = (frame_idx, s3_url)

        if generate_report:
            # Save report data to JSON
            report_json_path = result_dir / f"report_data_{model_type}.json"
            try:
                with open(report_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "model_type": model_type,
                        "timestamp": datetime.now().isoformat(),
                        "report_entries": report_data
                    }, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving report JSON: {str(e)}")

            # Generate local report with local image paths
            local_report_path = result_dir / "workflow_report_local.md"
            local_reports = log_to_markdown_report(
                local_report_path, report_data,
                cfg['report_template'],
                use_s3_urls=False
            )

            # Generate S3 report with S3 URLs
            s3_report_path = result_dir / "workflow_report.md"
            s3_reports = log_to_markdown_report(
                s3_report_path, report_data, 
                cfg['report_template'],
                use_s3_urls=True
            )

            # Upload reports to S3
            s3_urls = {}
            for report_type, local_path in [
                ('markdown', s3_reports['markdown']),
                ('html', s3_reports['html']),
                ('pdf', s3_reports['pdf'])
            ]:
                # Upload to S3 under the unique run folder
                s3_object_name = f"{s3_report_folder}/{Path(local_path).name}"
                vp._upload_to_s3(local_path, s3_object_name)
                s3_urls[report_type] = vp._get_s3_url(s3_object_name)

            # Convert key_frames to dict of just URLs for VideoAnalysisResponse
            response_key_frames = {k: v[1] for k, v in key_frames.items()}

            return VideoAnalysisResponse(
                local_markdown=local_reports['markdown'],
                local_html=local_reports['html'],
                local_pdf=local_reports['pdf'],
                s3_markdown=s3_urls['markdown'],
                s3_html=s3_urls['html'],
                s3_pdf=s3_urls['pdf'],
                key_frames=response_key_frames,
                frame_descriptions=frame_descriptions,
                result_dir=str(result_dir),
                frames_dir=frames_directory
            ).model_dump()

        return None

    except Exception as e:
        error_msg = f"Error in process_video_with_processor: {str(e)}"
        print(error_msg)
        # Save error to JSON file
        error_json_path = result_dir / f"error_{model_type}.json"
        try:
            with open(error_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat(),
                    "model_type": model_type,
                    "partial_results": report_data
                }, f, indent=2, ensure_ascii=False)
        except Exception as e2:
            print(f"Error saving error JSON: {str(e2)}")
        raise

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
        markdown_content += f"Frame Description: {item['frame_description']}\n\n"
        markdown_content += f"Similarity Score: {item['similarity_score']}\n\n"
        markdown_content += f"Frame Number: {item['frame_number']}\n\n"
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
        elements.append(Paragraph(f"Frame Description: {item['frame_description']}", styles['Normal']))
        elements.append(Paragraph(f"Similarity Score: {item['similarity_score']}", styles['Normal']))
        elements.append(Paragraph(f"Frame Number: {item['frame_number']}", styles['Normal']))
        if os.path.exists(item['image_path']):
            elements.append(Image(item['image_path'], width=400, height=300))
    doc.build(elements)

def process_video_clip_core(
    video_path: str,
    user_descs: List[str],
    fps: float = 2.0,
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
            user_descs=user_descs,
            model_type="clip",
            fps=fps,
            record_top_k_frames=record_top_k_frames,
            generate_report=generate_report
        )
        
        temp_dir, result_dir = setup_processing_environment(
            request.video_path, cache_db_path, keep_temp_dir, suffix="_clip")
        frames: List[Tuple[int, str]] = extract_frames(request.video_path, temp_dir, request.fps)
        processor = ClipEmbeddingProcessor(cache_db_path)

        results: Optional[Dict[str, str]] = process_video_with_processor(
            processor=processor,
            frames=frames,
            user_descs=request.user_descs,
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
    user_descs: List[str],
    fps: float = 2.0,
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
            user_descs=user_descs,
            model_type="openai",
            fps=fps,
            record_top_k_frames=record_top_k_frames,
            generate_report=generate_report
        )
        
        temp_dir, result_dir = setup_processing_environment(
            request.video_path, cache_db_path, keep_temp_dir, suffix="_openai")
        frames: List[Tuple[int, str]] = extract_frames(request.video_path, temp_dir, request.fps)
        processor = OpenAIEmbeddingProcessor(
            cfg["system_prompt"], cfg["vision_prompt"], cache_db_path)

        results: Optional[Dict[str, str]] = process_video_with_processor(
            processor=processor,
            frames=frames,
            user_descs=request.user_descs,
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

@cli.command(name="clip-embed")
@click.option("--video_path", required=True, help="Path to input video file")
@click.option("--descriptions", required=True, multiple=True, help="List of descriptions to search for")
@click.option("--fps", type=int, default=30, help="Target frames per second for extraction")
@click.option("--config_path", default="config.yaml", help="Path to config file")
@click.option("--cache_db_path", default="embeddings_cache.db", help="Path to embeddings cache database")
@click.option("--keep-temp-dir", is_flag=True, help="Keep temporary directory after processing")
@click.option("--record-top-k-frames", default=20, help="Number of top frames to record")
@click.option("--generate-report", is_flag=True, help="Generate report with results")
def process_video_clip(
    video_path, descriptions, fps, config_path,
    cache_db_path, keep_temp_dir,
    record_top_k_frames, generate_report
):
    """Process video using CLIP embedding model."""
    process_video_clip_core(
        video_path=video_path,
        user_descs=descriptions,
        fps=fps,
        config_path=config_path,
        cache_db_path=cache_db_path,
        keep_temp_dir=keep_temp_dir,
        record_top_k_frames=record_top_k_frames,
        generate_report=generate_report
    )

@cli.command(name="openai-embed")
@click.option("--video_path", required=True, help="Path to input video file")
@click.option("--descriptions", required=True, multiple=True, help="List of descriptions to search for")
@click.option("--fps", type=int, default=30, help="Target frames per second for extraction")
@click.option("--config_path", default="config.yaml", help="Path to config file")
@click.option("--cache_db_path", default="embeddings_cache.db", help="Path to embeddings cache database")
@click.option("--keep-temp-dir", is_flag=True, help="Keep temporary directory after processing")
@click.option("--record-top-k-frames", default=20, help="Number of top frames to record")
@click.option("--generate-report", is_flag=True, help="Generate report with results")
def process_video_openai(
    video_path, descriptions, fps, config_path,
    cache_db_path, keep_temp_dir,
    record_top_k_frames, generate_report
):
    """Process video using OpenAI embedding model."""
    process_video_openai_core(
        video_path=video_path,
        user_descs=descriptions,
        fps=fps,
        config_path=config_path,
        cache_db_path=cache_db_path,
        keep_temp_dir=keep_temp_dir,
        record_top_k_frames=record_top_k_frames,
        generate_report=generate_report
    )

@cli.command(name="summarize")
@click.option("--video_path", required=True, help="Path to input video file")
@click.option("--fps", type=float, default=30, help="Target frames per second for extraction")
@click.option("--keep-temp-dir", is_flag=False, help="Keep temporary directory after processing")
@click.option("--config_path", default="config.yaml", help="Path to config file")
@click.option("--cache_db_path", default="embeddings_cache.db", help="Path to embeddings cache database")
def summarize_video(
    video_path: str,
    fps: float,
    keep_temp_dir: bool,
    config_path: str,
    cache_db_path: str
):
    """Process video and generate summary."""
    import asyncio
    async def run():
        result = await process_and_summarize_video(
            video_path=video_path,
            fps=fps,
            config_path=config_path,
            cache_db_path=cache_db_path
        )
        if result:
            click.echo(f"Title: {result['title']}")
            click.echo(f"Summary: {result['summary']}")
        else:
            click.echo("Video processing failed.")
    asyncio.run(run())
    
if __name__ == "__main__":
    cli()
    # run openai with
    # python main.py openai-embed --video_path https://myimagebucketlabar.s3.us-east-2.amazonaws.com/V1_end.mp4 --descriptions "Pouring water into red cabbage filled beaker" --descriptions "Turning on heat plate" --descriptions "Putting red cabbage solution into test tube (first time)" --descriptions "Putting red cabbage solution into test tube (second time)" --generate-report
    # python main.py openai-embed --video_path ../data/V1_end.mp4 --descriptions "Pouring water into red cabbage filled beaker" --descriptions "Turning on heat plate" --descriptions "Putting red cabbage solution into test tube (first time)" --descriptions "Putting red cabbage solution into test tube (second time)"
    # run clip with
    # python main.py clip-embed --video_path https://myimagebucketlabar.s3.us-east-2.amazonaws.com/V1_end.mp4 --descriptions "Pouring water into red cabbage filled beaker" --descriptions "Turning on heat plate" --descriptions "Putting red cabbage solution into test tube (first time)" --descriptions "Putting red cabbage solution into test tube (second time)" --generate-report
    # summarie
    # python main.py summarize --video_path https://myimagebucketlabar.s3.us-east-2.amazonaws.com/V1_end.mp4