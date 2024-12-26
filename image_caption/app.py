import os
from pathlib import Path
from typing import List, Optional

from bentoml import service
import bentoml
from pydantic import Field

from main import (
    process_video_clip_core,
    process_video_openai_core,
    load_config
)
from models import VideoAnalysisRequest, VideoAnalysisResponse

@service(
    workers=4,  # Increased workers for better concurrency
    resources={
        "cpu": "4000m",  # 4 CPU cores
        "memory": "8Gi",  # 8GB RAM
        "gpu": 1,  # 1 GPU device
    },
    traffic={
        "timeout": 3600,  # 1 hour timeout for long video processing
        "max_latency": 300000,  # 5 minutes max latency
        "max_concurrency": 8,  # Maximum concurrent requests
        "external_queue": True
    }
)
class VideoAnalyzerService:
    def __init__(self):
        self.cfg = load_config("config.yaml")
        self.cache_db_path = "embeddings_cache.db"

    @bentoml.task
    async def analyze_video(
        self,
        video_path: str = Field(...),
        descriptions: List[str] = Field(...),
        model_type: str = Field(default="clip"),
        fps: int = Field(default=30),
        record_top_k_frames: int = Field(default=20),
        generate_report: bool = Field(default=True)
    ) -> VideoAnalysisResponse:
        """
        Analyze a video by processing its frames against given descriptions 
        using either CLIP or OpenAI embedding models.
        """
        try:
            # Process video with selected model
            if model_type.lower() == "openai":
                results = process_video_openai_core(
                    video_path=video_path,
                    user_descs=descriptions,
                    fps=fps,
                    record_top_k_frames=record_top_k_frames,
                    generate_report=generate_report
                )
            else:
                results = process_video_clip_core(
                    video_path=video_path,
                    user_descs=descriptions,
                    fps=fps,
                    record_top_k_frames=record_top_k_frames,
                    generate_report=generate_report
                )

            if not results:
                raise bentoml.exceptions.BentoMLException("No results were generated")

            return VideoAnalysisResponse(**results)
        
        except Exception as e:
            raise bentoml.exceptions.BentoMLException(str(e))

    @bentoml.api
    async def get_report_by_type(self, result_dir: str, report_type: str = "local_html") -> str:
        """
        Get a specific type of report.
        
        Args:
            result_dir: Directory containing the reports
            report_type: Type of report to retrieve (local_html, local_pdf, s3_html, s3_pdf)
        """
        report_paths = {
            "local_html": "workflow_report_local.html",
            "local_pdf": "workflow_report_local.pdf",
            "s3_html": "workflow_report.html",
            "s3_pdf": "workflow_report.pdf"
        }
        
        if report_type not in report_paths:
            raise bentoml.exceptions.InvalidArgument(f"Invalid report type. Must be one of: {', '.join(report_paths.keys())}")
        
        report_path = Path(result_dir) / report_paths[report_type]
        if not report_path.exists():
            raise bentoml.exceptions.NotFound(f"{report_type} report not found")
        
        if report_type.endswith('pdf'):
            # Return binary PDF data
            return report_path.read_bytes()
        else:
            # Return HTML content
            return str(report_path.read_text())

# To start
# bentoml serve app:VideoAnalyzerService --reload --port 8000
