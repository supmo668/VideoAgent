import os
from pathlib import Path
from typing import List, Optional, Dict, Any
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
import asyncio
import tempfile
from urllib.request import urlretrieve
import urllib.parse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VideoQAService")

import bentoml
from pydantic import BaseModel, Field

from main import (
    process_video_clip_core,
    process_video_openai_core,
    process_and_summarize_video,
    load_config
)
from models import VideoAnalysisRequest, VideoAnalysisResponse

@bentoml.service(
    name="videoQAFrameEmbed",
    workers=1,  # Increased workers for better concurrency
    resources={
        "cpu": "4000m",  # 4 CPU cores
        "memory": "8Gi",  # 8GB RAM
        "gpu": 1,  # 1 GPU device
    },
    traffic={
        "timeout": 3600,  # 1 hour timeout for long video processing
        "max_latency": 300000,  # 5 minutes max latency
        "max_concurrency": 8,  # Maximum concurrent requests
    }
)
class VideoAnalyzerService:
    def __init__(self):
        logger.info("Initializing VideoQAFrameEmbed service")
        try:
            self.cfg = load_config("config.yaml")
            logger.info("Config loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise
        self.cache_db_path = "embeddings_cache.db"
        logger.info("Service initialization complete")

    @bentoml.api
    async def videoQA_embed(
        self,
        video_path: str = Field(..., description="Path to the video file or HTTPS URL to an MP4 file"),
        descriptions: List[str] = Field(..., description="List of descriptions to match against"),
        model_type: str = Field(default="clip", description="Model type to use (clip or openai)"),
        fps: int = Field(default=30, description="Frames per second to process"),
        record_top_k_frames: int = Field(default=20, description="Number of top matching frames to record"),
        generate_report: bool = Field(default=True, description="Whether to generate a report")
    ) -> VideoAnalysisResponse:
        """
        Analyze a video by processing its frames against given descriptions 
        using either CLIP or OpenAI embedding models. The video_path can be a local file path
        or an HTTPS URL pointing to an MP4 file.
        """
        logger.info(f"Starting video analysis with parameters: model_type={model_type}, fps={fps}")
        logger.debug(f"Full parameters: video_path={video_path}, descriptions={descriptions}, "
                    f"record_top_k_frames={record_top_k_frames}, generate_report={generate_report}")
        
        try:
            # Handle URL if provided
            if video_path.startswith('http'):
                logger.info("Detected video URL, starting download")
                # Create a temporary file with .mp4 extension
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                    logger.debug(f"Created temporary file: {temp_path}")
                    # Download the video
                    try:
                        logger.info("Downloading video...")
                        urlretrieve(video_path, temp_path)
                        logger.info("Video download complete")
                        video_path = temp_path
                    except Exception as e:
                        logger.error(f"Failed to download video: {str(e)}")
                        raise ValueError(f"Failed to download video: {str(e)}")

            logger.info(f"Using model type: {model_type}")
            if model_type.lower() == "clip":
                logger.info("Starting CLIP model processing")
                try:
                    result = await process_video_clip_core(
                        video_path=video_path,
                        user_descs=descriptions,
                        fps=fps,
                        record_top_k_frames=record_top_k_frames,
                        generate_report=generate_report
                    )
                    logger.info("CLIP processing completed successfully")
                except Exception as e:
                    logger.error(f"Error in CLIP processing: {str(e)}")
                    raise
            elif model_type.lower() == "openai":
                logger.info("Starting OpenAI model processing")
                try:
                    result = await process_video_openai_core(
                        video_path=video_path,
                        user_descs=descriptions,
                        fps=fps,
                        record_top_k_frames=record_top_k_frames,
                        generate_report=generate_report
                    )
                    logger.info("OpenAI processing completed successfully")
                except Exception as e:
                    logger.error(f"Error in OpenAI processing: {str(e)}")
                    raise
            else:
                logger.error(f"Unsupported model type: {model_type}")
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Clean up temporary file if it was created
            if 'temp_path' in locals():
                try:
                    logger.debug(f"Cleaning up temporary file: {temp_path}")
                    os.unlink(temp_path)
                    logger.debug("Temporary file cleaned up successfully")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {str(e)}")
                
            logger.info("Analysis completed successfully")
            return VideoAnalysisResponse(**result)
        except Exception as e:
            # Clean up temporary file in case of error
            if 'temp_path' in locals():
                try:
                    logger.debug(f"Cleaning up temporary file after error: {temp_path}")
                    os.unlink(temp_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file after error: {str(cleanup_error)}")
            
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise ValueError(f"Error processing video: {str(e)}")

    @bentoml.api
    async def get_report_by_type(self, result_dir: str, report_type: str = "local_html") -> Dict[str, str]:
        """
        Get a specific type of report.
        """
        logger.info(f"Getting report of type {report_type} from directory {result_dir}")
        try:
            if report_type == "local_html":
                report_path = os.path.join(result_dir, "report.html")
            elif report_type == "local_md":
                report_path = os.path.join(result_dir, "report.md")
            elif report_type == "local_pdf":
                report_path = os.path.join(result_dir, "report.pdf")
            elif report_type == "s3_html":
                report_path = os.path.join(result_dir, "report_s3.html")
            else:
                logger.error(f"Unsupported report type: {report_type}")
                raise ValueError(f"Unsupported report type: {report_type}")
            
            logger.info(f"Report path: {report_path}")
            return {"report_path": report_path}
        except Exception as e:
            logger.error(f"Error getting report: {str(e)}", exc_info=True)
            raise

    @bentoml.api
    async def summarize_video(
        self,
        video_path: str = Field(..., description="Path to the video file"),
        fps: int = Field(default=30, description="Frames per second to process")
    ) -> Dict[str, Any]:
        """
        Process a video and generate a summary.
        """
        logger.info(f"Starting video summarization with parameters: fps={fps}")
        logger.debug(f"Full parameters: video_path={video_path}")
        
        try:
            result = await process_and_summarize_video(
                video_path=video_path,
                fps=fps
            )
            logger.info("Video summarization completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error summarizing video: {str(e)}", exc_info=True)
            raise ValueError(f"Error summarizing video: {str(e)}")

# To start
# bentoml serve app:VideoAnalyzerService --reload --port 8000
