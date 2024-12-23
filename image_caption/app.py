import os
import uuid
from pathlib import Path
from typing import List, Optional

import bentoml
from bentoml.io import JSON, File
from pydantic import BaseModel

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

video_analyzer_runner = bentoml.Service("video_analyzer", runners=[])

@video_analyzer_runner.api(
    input=JSON(pydantic_model=VideoAnalysisRequest),
    output=JSON(),
)
def analyze_video(request: VideoAnalysisRequest):
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
            request.video_path, 
            cache_db_path, 
            keep_temp_dir=True
        )
        
        # Extract frames
        frames = extract_frames(
            request.video_path, 
            sample_freq=request.sample_freq, 
            temp_dir=temp_dir
        )
        
        # Choose embedding processor based on model type
        if request.model_type.lower() == "openai":
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
            questions=request.questions,
            cfg=cfg,
            result_dir=result_dir,
            record_top_k_frames=request.record_top_k_frames,
            model_type=request.model_type.lower(),
            generate_report=request.generate_report
        )
        
        # Prepare response with result directory contents
        results = {
            "result_dir": str(result_dir),
            "report_path": str(result_dir / "workflow_report.md"),
            "frames_dir": str(temp_dir)
        }
        
        return results
    
    except Exception as e:
        raise bentoml.exceptions.BentoMLException(str(e))

@video_analyzer_runner.api(
    input=JSON(),
    output=File(),
)
def download_report(result_dir: str):
    """
    Download the generated markdown report.
    """
    report_path = Path(result_dir) / "workflow_report.md"
    if not report_path.exists():
        raise bentoml.exceptions.NotFound("Report not found")
    
    return report_path
