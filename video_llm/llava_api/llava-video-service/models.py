"""Data models for the LLaVA Video Service."""
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEFAULTS = config["defaults"]
SERVICE_CONFIG = config["service"]

class ImageMetadata(BaseModel):
    """Metadata for image analysis."""
    prompt: str = Field(DEFAULTS["image"]["prompt"], description="Prompt name or text for image analysis")
    temperature: float = Field(DEFAULTS["image"]["temperature"], description="Temperature for generation")
    max_new_tokens: int = Field(DEFAULTS["image"]["max_new_tokens"], description="Maximum number of new tokens to generate")
    load_from_db: bool = Field(False, description="Whether to load prompt from database")

class VideoMetadata(BaseModel):
    """Metadata for video analysis."""
    video_url: Optional[str] = Field(None, description="URL of the video to analyze")
    prompt: str = Field(DEFAULTS["video"]["prompt"], description="Prompt name or text for video analysis")
    temperature: float = Field(DEFAULTS["video"]["temperature"], description="Temperature for generation")
    max_new_tokens: int = Field(DEFAULTS["video"]["max_new_tokens"], description="Maximum number of new tokens to generate")
    fps: int = Field(DEFAULTS["video"]["fps"], description="Frames per second to extract")
    load_from_db: bool = Field(False, description="Whether to load prompt from database")

class ImageResponse(BaseModel):
    """Response model for image analysis."""
    response: Optional[str] = Field(None, description="Generated response")
    error: Optional[str] = Field(None, description="Error message if any")

class VideoResponse(BaseModel):
    """Response model for video analysis."""
    response: Optional[str] = Field(None, description="Generated response")
    error: Optional[str] = Field(None, description="Error message if any")

class LlavaVideoRequest(BaseModel):
    """Request model for video analysis."""
    metadata: VideoMetadata = Field(..., description="Metadata for the video llava workflow")
    video: Optional[Path] = Field(None, description="Video file")

class LlavaImageRequest(BaseModel):
    """Request model for image analysis."""
    metadata: ImageMetadata = Field(..., description="Metadata for the image llava workflow")
    image: Optional[Path] = Field(None, description="Image file")

class TwelveLabsRequest(BaseModel):
    """Request model for TwelveLabs API."""
    video_url: str = Field(..., description="URL of the video to analyze")
    prompt: str = Field(DEFAULTS["twelvelabs"]["prompt"], description="Prompt for video analysis")
    language: str = Field(DEFAULTS["twelvelabs"]["language"], description="Language for transcription")
    provide_transcription: bool = Field(DEFAULTS["twelvelabs"]["provide_transcription"], description="Whether to provide transcription")
    enable_video_stream: bool = Field(DEFAULTS["twelvelabs"]["enable_video_stream"], description="Whether to enable video streaming")
    temperature: float = Field(DEFAULTS["twelvelabs"]["temperature"], description="Temperature for generation")

class TwelveLabsResponse(BaseModel):
    """Response model for TwelveLabs API."""
    response: Optional[Dict[str, Any]] = Field(None, description="Generated response")
    error: Optional[str] = Field(None, description="Error message if any")
