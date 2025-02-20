"""Data models for video analysis."""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum

class ModelType(str, Enum):
    """Type of model to use for video analysis."""
    CLIP = "clip"
    BLIP = "blip"
    COMBINED = "combined"

class VideoQARequest(BaseModel):
    """Request model for video analysis API."""
    video_path: Union[str, HttpUrl] = Field(..., description="Local path or URL to video file")
    descriptions: List[str] = Field(..., description="List of action descriptions to identify")
    fps: float = Field(default=2.0, description="Frames per second to process")
    model_type: ModelType = Field(
        default=ModelType.COMBINED, description="Model type to use")
    generate_report: bool = Field(default=True, description="Whether to generate a report")
    record_top_k_frames: int = Field(default=5, description="Number of top matching frames to record")

    class Config:
        use_enum_values = True

class FrameMatch(BaseModel):
    """A single frame match with its similarity score."""
    frame_number: int = Field(..., description="Frame number in the video sequence")
    similarity: float = Field(..., description="Similarity score between frame and description")
    timestamp: float = Field(..., description="Timestamp of the frame in seconds")
    frame_path: Optional[str] = Field(None, description="Path to the frame image file")

class ActionSegment(BaseModel):
    """Time segment where an action occurs in the video."""
    start_frame: int = Field(..., description="Starting frame number of the action")
    end_frame: int = Field(..., description="Ending frame number of the action")
    start_time: float = Field(..., description="Starting time in seconds")
    end_time: float = Field(..., description="Ending time in seconds")
    confidence: float = Field(..., description="Confidence score for the segment")

class ActionResult(BaseModel):
    """Complete result for a single action description."""
    description: str = Field(..., description="The action description being analyzed")
    segment: ActionSegment = Field(..., description="The identified continuous segment")
    frame_matches: List[FrameMatch] = Field(..., description="List of matching frames")

class ProcessingResults(BaseModel):
    """Complete results of video processing."""
    timestamp: datetime = Field(default_factory=datetime.now, description="Time of analysis")
    actions: List[ActionResult] = Field(..., description="List of action analysis results")
    report_dir: Optional[str] = Field(None, description="Directory containing the generated report")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-02-20T19:50:24Z",
                "actions": [
                    {
                        "description": "Pouring water into beaker",
                        "segment": {
                            "start_frame": 17,
                            "end_frame": 22,
                            "start_time": 0.283,
                            "end_time": 0.367,
                            "confidence": 0.119
                        },
                        "frame_matches": [
                            {
                                "frame_number": 20,
                                "similarity": 0.209,
                                "timestamp": 0.667,
                                "frame_path": "frames/frame_0020.jpg"
                            }
                        ]
                    }
                ],
                "report_dir": "results/analysis_20250220_195024"
            }
        }
