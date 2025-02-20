"""BentoML service for video analysis."""

import os
from pathlib import Path
import asyncio
import tempfile
from urllib.request import urlretrieve
import json
import sys

import bentoml
from bentoml.io import JSON

from main import process_video_api
from models import VideoQARequest, ProcessingResults

@bentoml.service(
    name="video_frame_embed_qa",
    workers=4,
    resources={
        "cpu": "4000m",
        "memory": "8Gi",
        "gpu": 1,
    },
    traffic={
        "timeout": 3600,
        "max_latency": 300000,
        "max_concurrency": 8,
        "external_queue": True
    }
)
class VideoFrameEmbedQA:
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def _download_video(self, video_url: str) -> str:
        """Download video from URL to temporary file."""
        video_path = self.temp_dir / f"video_{hash(video_url)}.mp4"
        urlretrieve(video_url, video_path)
        return str(video_path)

    @bentoml.api(route="/videoQA")
    async def videoQA(self, request: VideoQARequest) -> ProcessingResults:
        """
        Analyze a video by processing its frames against given descriptions.
        Supports both local file paths and video URLs.
        """
        video_path = str(request.video_path)
        downloaded_path = None
        try:
            if video_path.startswith(('http://', 'https://')):
                downloaded_path = self._download_video(video_path)
                video_path = downloaded_path
            
            results = process_video_api(
                video_path=video_path,
                descriptions=request.descriptions,
                fps=request.fps,
                record_top_k_frames=request.record_top_k_frames,
                generate_report=request.generate_report,
                model_type=request.model_type
            )
            
            return ProcessingResults(**results)
            
        except Exception as e:
            raise bentoml.exceptions.BentoMLException(str(e))
        finally:
            # Cleanup temporary files if video was downloaded
            if downloaded_path and os.path.exists(downloaded_path):
                os.unlink(downloaded_path)

    @bentoml.api(route="/report/{result_dir}")
    async def get_analysis_details(self, result_dir: str) -> ProcessingResults:
        """Get the analysis details JSON for a processed video."""
        try:
            report_path = Path(result_dir) / "report" / "analysis_details.json"
            if not report_path.exists():
                raise bentoml.exceptions.NotFound("Analysis details not found")
            
            with open(report_path) as f:
                data = json.load(f)
                return ProcessingResults(**data)
                
        except Exception as e:
            raise bentoml.exceptions.BentoMLException(str(e))

# Example curl command:
"""
curl -X POST http://localhost:8080/videoQA \
--header 'Content-Type: application/json' \
--data '{
    "video_path": "https://myimagebucketlabar.s3.us-east-2.amazonaws.com/jove-video/extraction_of_aqueous_metabolites_from_cultured_adherent_cells_for_metabolomic_analysis_by_capillary_electrophoresis_mass_spectrometry.mp4",
    "descriptions": [
        "cell culture handling",
        "pipetting liquids",
        "metabolite extraction",
        "sample preparation"
    ],
    "fps": 2.0,
    "model_type": "combined",
    "generate_report": true,
    "record_top_k_frames": 5
}'
"""