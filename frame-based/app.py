import os
from pathlib import Path
from typing import List, Optional, Dict, Any
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
import asyncio

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
    name="video_frame_analyzer",
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
        self.cfg = load_config("config.yaml")
        self.cache_db_path = "embeddings_cache.db"

    @bentoml.api
    async def analyze_video(
        self,
        video_path: str = Field(..., description="Path to the video file"),
        descriptions: List[str] = Field(..., description="List of descriptions to match against"),
        model_type: str = Field(default="clip", description="Model type to use (clip or openai)"),
        fps: int = Field(default=30, description="Frames per second to process"),
        record_top_k_frames: int = Field(default=20, description="Number of top matching frames to record"),
        generate_report: bool = Field(default=True, description="Whether to generate a report")
    ) -> VideoAnalysisResponse:
        """
        Analyze a video by processing its frames against given descriptions 
        using either CLIP or OpenAI embedding models.
        """
        try:
            if model_type.lower() == "clip":
                result = await process_video_clip_core(
                    video_path=video_path,
                    user_descs=descriptions,
                    fps=fps,
                    record_top_k_frames=record_top_k_frames,
                    generate_report=generate_report
                )
            elif model_type.lower() == "openai":
                result = await process_video_openai_core(
                    video_path=video_path,
                    user_descs=descriptions,
                    fps=fps,
                    record_top_k_frames=record_top_k_frames,
                    generate_report=generate_report
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            return VideoAnalysisResponse(**result)
        except Exception as e:
            raise ValueError(f"Error processing video: {str(e)}")

    @bentoml.api
    async def get_report_by_type(self, result_dir: str, report_type: str = "local_html") -> Dict[str, str]:
        """
        Get a specific type of report.
        """
        if report_type == "local_html":
            report_path = os.path.join(result_dir, "report.html")
        elif report_type == "local_md":
            report_path = os.path.join(result_dir, "report.md")
        elif report_type == "local_pdf":
            report_path = os.path.join(result_dir, "report.pdf")
        elif report_type == "s3_html":
            report_path = os.path.join(result_dir, "report_s3.html")
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
            
        return {"report_path": report_path}

    @bentoml.api
    async def summarize_video(
        self,
        video_path: str = Field(..., description="Path to the video file"),
        fps: int = Field(default=30, description="Frames per second to process")
    ) -> Dict[str, Any]:
        """
        Process a video and generate a summary.
        """
        try:
            result = await process_and_summarize_video(
                video_path=video_path,
                fps=fps
            )
            return result
        except Exception as e:
            raise ValueError(f"Error summarizing video: {str(e)}")

# To start
# bentoml serve app:VideoAnalyzerService --reload --port 8000
