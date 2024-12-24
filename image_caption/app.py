import os
import uuid
from pathlib import Path
from typing import List, Optional

from bentoml import service
import bentoml

# Upgrade bentoml to the latest version
# bentoml_version = "latest"
# bentoml = bentoml.load(bentoml_version)

from pydantic import BaseModel, Field

from main import (
    setup_processing_environment, 
    extract_frames, 
    process_video_with_processor, 
    load_config, 
    ClipEmbeddingProcessor, 
    OpenAIEmbeddingProcessor
)


class VideoAnalysisRequest(BaseModel):
    video_path: str
    questions: List[str]
    model_type: str = "clip"
    sample_freq: int = 30
    record_top_k_frames: int = 20
    generate_report: bool = True

@service(
    workers=2,
    resources={
        "cpu": "2000m"
    },
    traffic={
        "concurrency": 16,
        "external_queue": True
    }
)
class VideoAnalyzerService:
    runners = []

    @bentoml.task
    async def analyze_video(
        self,
        video_path: str = Field(...),
        questions: List[str] = Field(...),
        model_type: str = Field(default="clip"),
        sample_freq: int = Field(default=30),
        record_top_k_frames: int = Field(default=20),
        generate_report: bool = Field(default=True)
    ) -> str:
        """
        Analyze a video by processing its frames against given questions 
        using either CLIP or OpenAI embedding models.
        """
        try:
            # Load default configuration
            cfg = load_config("config.yaml")
            cache_db_path = "embeddings_cache.db"
            
            # Set up processing environment 
            temp_dir, result_dir = setup_processing_environment(
                video_path, 
                cache_db_path, 
                keep_temp_dir=True
            )
            
            # Extract frames
            frames = extract_frames(
                video_path, 
                sample_freq=sample_freq, 
                temp_dir=temp_dir
            )
            
            # Choose embedding processor based on model type
            if model_type.lower() == "openai":
                processor = OpenAIEmbeddingProcessor(
                    cfg["system_prompt"], 
                    cfg["vision_prompt"], 
                    cache_db_path
                )
            else:
                processor = ClipEmbeddingProcessor(cache_db_path)
            
            # Process video with selected processor
            process_video_with_processor(
                processor=processor,
                frames=frames,
                questions=questions,
                cfg=cfg,
                result_dir=result_dir,
                record_top_k_frames=record_top_k_frames,
                model_type=model_type.lower(),
                generate_report=generate_report
            )
            
            # Prepare response with result directory contents
            results = {
                "result_dir": str(result_dir),
                "report_path": str(result_dir / "workflow_report.md"),
                "frames_dir": str(temp_dir)
            }
            return str(results)
        
        except Exception as e:
            raise bentoml.exceptions.BentoMLException(str(e))

    @bentoml.api
    async def download_report(self, result_dir: str) -> str:
        """
        Download the generated HTML report.
        """
        html_report_path = Path(result_dir) / "workflow_report.html"
        if not html_report_path.exists():
            raise bentoml.exceptions.NotFound("HTML report not found")
        
        return str(html_report_path.read_text())

# To start
# bentoml serve app:VideoAnalyzerService --reload --port 8000