from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import tempfile
import shutil
from typing import Optional
import torch
import os
from service import (
    LabARVideoReportingService,
    VideoMetadata,
    ImageMetadata,
    TwelveLabsRequest,
    VideoResponse,
    ImageResponse,
    TwelveLabsResponse
)
from logging_config import get_logger

logger = get_logger()

# Set CUDA configuration
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

app = FastAPI(
    title="LabAR Video Reporting Service",
    description="API for video and image analysis using LLaVA-Video-7B-Qwen2",
    version="1.0.0"
)

# Initialize service with specific model configuration
service = LabARVideoReportingService()

@app.post("/analyze/video", response_model=VideoResponse)
async def analyze_video(
    prompt: str = Form("Please describe this video in detail."),
    temperature: float = Form(0.2),
    max_new_tokens: int = Form(512),
    context_len: int = Form(512),
    fps: int = Form(1),
    video_url: Optional[str] = Form(None),
    video_file: Optional[UploadFile] = File(None)
) -> VideoResponse:
    """
    Analyze a video using either a file upload or URL.
    
    Parameters:
    - video_url: Optional URL of the video to analyze
    - video_file: Optional video file upload (MP4)
    - prompt: Analysis prompt (default: "Please describe this video in detail.")
    - temperature: Generation temperature (default: 0.2)
    - max_new_tokens: Maximum tokens to generate (default: 512)
    - context_len: Context length (default: 512)
    - fps: Frames per second to extract (default: 1)
    
    Either video_url or video_file must be provided.
    The video will be processed with the following settings:
    - Max frames: 16 (force sampled uniformly)
    - Model: LLaVA-Video-7B-Qwen2
    - Device: CUDA with float16 precision
    """
    try:
        if not video_url and not video_file:
            raise HTTPException(
                status_code=400,
                detail="Either video_url or video_file must be provided"
            )
            
        if video_url and video_file:
            raise HTTPException(
                status_code=400,
                detail="Cannot provide both video_url and video_file"
            )

        # Clear CUDA cache before processing
        torch.cuda.empty_cache()

        # Create metadata object with exact configuration matching LLaVA-Video-7B-Qwen2
        metadata = VideoMetadata(
            video_url=video_url,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_len=context_len,
            fps=fps
        )

        if video_file:
            # Handle file upload case
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                shutil.copyfileobj(video_file.file, tmp)
                tmp_path = Path(tmp.name)
            
            try:
                logger.info(f"Processing uploaded video file")
                return await service.analyze_video_file(metadata, tmp_path)
            finally:
                # Cleanup temporary file
                tmp_path.unlink(missing_ok=True)
                torch.cuda.empty_cache()
        else:
            # Handle URL case
            logger.info(f"Processing video from URL: {video_url}")
            return await service.analyze_video(metadata)
            
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        torch.cuda.empty_cache()  # Ensure cleanup even on error
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/image", response_model=ImageResponse)
async def analyze_image(
    prompt: str = Form("Describe this image."),
    temperature: float = Form(0.2),
    max_new_tokens: int = Form(512),
    context_len: int = Form(512),
    image_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None)
) -> ImageResponse:
    """
    Analyze an image using either a file upload or URL.
    
    Parameters:
    - image_url: Optional URL of the image to analyze
    - image_file: Optional image file upload
    - prompt: Analysis prompt (default: "Describe this image.")
    - temperature: Generation temperature (default: 0.2)
    - max_new_tokens: Maximum tokens to generate (default: 512)
    - context_len: Context length (default: 512)
    
    Either image_url or image_file must be provided.
    """
    try:
        if not image_url and not image_file:
            raise HTTPException(
                status_code=400,
                detail="Either image_url or image_file must be provided"
            )
            
        if image_url and image_file:
            raise HTTPException(
                status_code=400,
                detail="Cannot provide both image_url and image_file"
            )

        # Clear CUDA cache before processing
        torch.cuda.empty_cache()

        # Create metadata object
        metadata = ImageMetadata(
            image_url=image_url,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_len=context_len
        )

        if image_file:
            # Handle file upload case
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                shutil.copyfileobj(image_file.file, tmp)
                tmp_path = Path(tmp.name)
            
            try:
                logger.info(f"Processing uploaded image file")
                return await service.analyze_image_file(metadata, tmp_path)
            finally:
                # Cleanup temporary file
                tmp_path.unlink(missing_ok=True)
                torch.cuda.empty_cache()
        else:
            # Handle URL case
            logger.info(f"Processing image from URL: {image_url}")
            return await service.analyze_image(metadata)
            
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        torch.cuda.empty_cache()  # Ensure cleanup even on error
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/video/summary", response_model=TwelveLabsResponse)
async def get_video_summary(request: TwelveLabsRequest) -> TwelveLabsResponse:
    """
    Get a summary of a video using TwelveLabs API.
    """
    try:
        return await service.summary(request)
    except Exception as e:
        logger.error(f"Error getting video summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/video/highlights", response_model=TwelveLabsResponse)
async def get_video_highlights(request: TwelveLabsRequest) -> TwelveLabsResponse:
    """
    Get highlights from a video using TwelveLabs API.
    """
    try:
        return await service.highlight(request)
    except Exception as e:
        logger.error(f"Error getting video highlights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/video/chapters", response_model=TwelveLabsResponse)
async def get_video_chapters(request: TwelveLabsRequest) -> TwelveLabsResponse:
    """
    Get chapter segments from a video using TwelveLabs API.
    """
    try:
        return await service.chapter(request)
    except Exception as e:
        logger.error(f"Error getting video chapters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
