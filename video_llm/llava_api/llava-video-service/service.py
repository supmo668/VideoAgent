# Standard library imports
import asyncio
import os
import copy
import io
import json
import tempfile
import urllib.parse
import urllib.request
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock

# Third-party imports
import aiohttp
import bentoml
import numpy as np
import torch
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
load_dotenv()
import cv2
from transformers import TextStreamer

# Local imports
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from prompt_library import PromptLibrary
from tracing import TracingConfig
from langsmith import traceable

# Initialize tracing
tracing_config = TracingConfig()
tracing_config.enable_tracing()

# Model request/response classes
class ImageMetadata(BaseModel):
    image_url: Optional[str] = Field(None, description="URL of the image to analyze")
    prompt: str = Field("Describe this image", description="Prompt for image analysis")
    temperature: float = Field(0.2, description="Temperature for generation")
    max_new_tokens: int = Field(512, description="Maximum number of new tokens to generate")
    context_len: int = Field(2048, description="Context length for generation")

class VideoMetadata(BaseModel):
    video_url: Optional[str] = Field(None, description="URL of the video to analyze")
    prompt: str = Field("Describe this video", description="Prompt for video analysis")
    temperature: float = Field(0.2, description="Temperature for generation")
    max_new_tokens: int = Field(512, description="Maximum number of new tokens to generate")
    context_len: int = Field(2048, description="Context length for generation")
    fps: int = Field(1, description="Frames per second to extract")

class ImageResponse(BaseModel):
    response: Optional[str] = Field(None, description="Generated response")
    status: str = Field("success", description="Processing status")
    error: Optional[str] = Field(None, description="Error message if any")

class VideoResponse(BaseModel):
    response: Optional[str] = Field(None, description="Generated response")
    status: str = Field("success", description="Processing status")
    error: Optional[str] = Field(None, description="Error message if any")

class LlavaVideoRequest(BaseModel):
    metadata: VideoMetadata = Field(..., description="Metadata for the video llava workflow")
    video: Optional[Path] = Field(None, description="Video file")

class LlavaImageRequest(BaseModel):
    metadata: ImageMetadata = Field(..., description="Metadata for the image llava workflow")
    image: Optional[Path] = Field(None, description="Image file")

class TwelveLabsRequest(BaseModel):
    video_url: str = Field(..., description="URL of the video to analyze")
    index_id: str = Field(..., description="Index ID for the video")
    language: str = Field("en", description="Language of the video")
    provide_transcription: bool = Field(False, description="Whether to provide transcription")
    enable_video_stream: bool = Field(True, description="Whether to enable video streaming")
    prompt: str = Field(default="Provide a detailed and technical summary of the video.", description="Prompt for the analysis")
    prompt_name: str = Field(None, description="Name of the prompt")
    temperature: float = Field(default=0.7, description="Temperature for generation")

class TwelveLabsResponse(BaseModel):
    task_id: Optional[str] = Field(None, description="Task ID from TwelveLabs")
    video_id: Optional[str] = Field(None, description="Video ID from TwelveLabs")
    status: str = Field("success", description="Processing status")
    error: Optional[str] = Field(None, description="Error message if any")
    id: Optional[str] = Field(None, description="ID of the analysis")
    summary: Optional[str] = Field(None, description="Summary of the video")
    highlights: Optional[List[str]] = Field(None, description="Highlights of the video")
    chapters: Optional[List[str]] = Field(None, description="Chapters of the video")

# LLaVA Service
@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 300}
)
class LLaVAVideoService:
    def __init__(self) -> None:
        disable_torch_init()
        self.pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        self.model_name = "llava_qwen"
        self.device = "cuda"

        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        self.tokenizer, self.model, self.image_processor, self.context_len = self._load_model()
        
        if hasattr(self.model, 'get_vision_tower'):
            vision_tower = self.model.get_vision_tower()
            if hasattr(vision_tower, 'vision_tower'):
                vision_tower.vision_tower = vision_tower.vision_tower.to(dtype=torch.float16)
        
        self.model.eval()

    def _load_model(self):
        from llava_custom.builder import load_pretrained_model
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.pretrained, 
            None, 
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_4bit=True
        )
        return tokenizer, model, image_processor, context_len

    @contextmanager
    def _temp_file_context(self, content: bytes, suffix: str = "") -> str:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"llava_{uuid.uuid4()}_"))
        temp_path = temp_dir / f"{uuid.uuid4()}{suffix}"
        try:
            with open(temp_path, "wb") as f:
                f.write(content)
            yield str(temp_path)
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
                if temp_dir.exists():
                    temp_dir.rmdir()
            except Exception as e:
                print(f"Warning: Failed to cleanup temporary file {temp_path}: {str(e)}")

    async def _download_file(self, url: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download file: {response.status}")
                return await response.read()

    async def _process_video(self, video_path: str, fps: int = 2) -> List[str]:
        try:

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Failed to open video file")

            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)

            while frame_count < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to bytes
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append(buffer.tobytes())
                frame_count += frame_interval

            cap.release()
            return frames
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")

    def _generate_response(self, prompt: str, image_paths: List[str], metadata: Union[ImageMetadata, VideoMetadata]) -> str:
        conv = conv_templates["llava_v1"].copy()
        roles = conv.roles

        image_tensors = []
        for image_path in image_paths:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_tensor = process_images([image], self.image_processor)
            image_tensors.append(image_tensor)

        if len(image_tensors) == 0:
            raise ValueError("No valid images found")

        image_tensor = torch.cat(image_tensors, dim=0)
        conv.append_message(roles[0], prompt)
        conv.append_message(roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                do_sample=True if metadata.temperature > 0 else False,
                temperature=metadata.temperature,
                max_new_tokens=metadata.max_new_tokens,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        return outputs

    @bentoml.api
    async def analyze_video(self, metadata: VideoMetadata) -> VideoResponse:
        try:
            if metadata.video_url:
                video_content; bytes = await self._download_file(metadata.video_url)
                with self._temp_file_context(video_content, ".mp4") as temp_video:
                    return analyze_video_file(metadata, temp_video)
        except Exception as e:
            return VideoResponse(
                status="failed",
                error=str(e),
            )
    
    @bentoml.api
    async def analyze_video_file(self, metadata: VideoMetadata, video: Path) -> VideoResponse:
        try:
            frames = await self._process_video(str(video), metadata.fps)
            with self._temp_file_context(frames[0], ".jpg") as temp_frame:
                    response = self._generate_response
                    (metadata.prompt, [temp_frame], metadata)
                return VideoResponse(response=response)
        except Exception as e:
            return VideoResponse(
                    status="failed",
                    error=str(e),
                )

    @bentoml.api
    async def analyze_image(self, metadata: ImageMetadata) -> ImageResponse:
        """Analyze a single image."""
        try:
            if metadata.image_url:
                image_content = await self._download_file(metadata.image_url)
                with self._temp_file_context(image_content, ".jpg") as temp_image:
                    response = self._generate_response(
                        metadata.prompt, [temp_image], metadata)
            else:
                response = self._generate_response(
                    metadata.prompt, [str(image)], metadata)

            return ImageResponse(response=response)
        except Exception as e:
            return ImageResponse(
                status="failed",
                error=str(e),
            )

# TwelveLabs Service
@bentoml.service(
    traffic={"timeout": 60},
    workers=1,
    concurrency=1
)
class TwelveLabsAPIService:
    def __init__(self) -> None:
        self.BASE_URL = "https://api.twelvelabs.io/v1.3"
        self.API_KEY = os.getenv("TWELVE_LABS_API_KEY")
        if not self.API_KEY:
            raise ValueError("TWELVE_LABS_API_KEY environment variable is not set")
        self.prompt_library = PromptLibrary()

    @traceable(run_type="chain")
    async def _video_qa_request(
        self, 
        request: TwelveLabsRequest,
        request_type: str,
    ) -> TwelveLabsResponse:
        try:
            # Get prompt from name if provided and prompt library is available
            if request.prompt_name and not request.prompt:
                if hasattr(self, 'prompt_library'):
                    prompt = await self.prompt_library.get_prompt(request.prompt_name)
                    request.prompt = prompt
                else:
                    print("Warning: Prompt library not available, using default prompt")

            # Upload video first
            upload_response: TwelveLabsResponse = await self.upload_video(request)
            if upload_response.status == "failed":
                return upload_response

            # Make analysis request
            headers = {
                "x-api-key": self.API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "video_id": upload_response.video_id,
                "type": request_type,
                "prompt": request.prompt,
                "temperature": request.temperature
            }
            
            endpoint = {
                "summary": "/summarize",
                "highlight": "/highlights",
                "chapter": "/chapters"
            }[request_type]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.BASE_URL}{endpoint}", headers=headers, json=payload) as response:
                    if response.status not in (200, 201):  # Accept both OK and Created
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    
                    result = await response.json()
                    return TwelveLabsResponse(
                        task_id=upload_response.task_id,
                        video_id=upload_response.video_id,
                        id=result.get("id"),
                        summary=result.get("summary"),
                        highlights=result.get("highlights"),
                        chapters=result.get("chapters")
                    )
        except Exception as e:
            return TwelveLabsResponse(
                status="failed",
                error=str(e)
            )

    async def upload_video(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        """Upload a video to TwelveLabs using video URL."""
        try:
            headers = {
                "x-api-key": self.API_KEY,
                "accept": "application/json",
            }

            # Check existing videos in the index
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/indexes/{request.index_id}/videos",
                    headers=headers,
                    params={
                        "page": 1,
                        # Maximum is 50
                        "page_limit": 50,
                        "sort_by": "created_at",
                        "sort_option": "desc"
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Failed to fetch videos: {error_text}")
                    # retrieve list of existing video data
                    existing_videos = await response.json()
                    for video in existing_videos.get("data", []):
                        system_metadata = video.get("system_metadata", {})
                        video_title = system_metadata.get("video_title", system_metadata.get("filename"))
                        if Path(request.video_url).name == video_title:
                            print(f"Video '{video_title}' already exists in index with ID: {video['_id']}")
                            return TwelveLabsResponse(
                                task_id=None,
                                video_id=video["_id"]
                            )
            print(f"Proceeding to upload.")
            boundary = "011000010111000001101001"
            fields = ["provide_transcription", "language", "enable_video_stream", "index_id", "video_url"]
            values = [str(request.provide_transcription).lower(), request.language, str(request.enable_video_stream).lower(), request.index_id, request.video_url]

            payload = "\r\n".join(
                [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}" for k, v in zip(fields, values)]
            ) + f"\r\n--{boundary}--"

            headers.update({
                "Content-Type": f"multipart/form-data; boundary={boundary}"
            })

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.BASE_URL}/tasks", headers=headers, data=payload) as response:
                    if response.status not in (200, 201):
                        raise Exception(f"Upload failed: {response.status} - {await response.text()}")
                    result = await response.json()
                    return TwelveLabsResponse(task_id=result.get("_id"), video_id=result.get("video_id"))
        except Exception as e:
            return TwelveLabsResponse(
                status="failed",
                error=str(e)
            )

    @bentoml.api
    async def highlight(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        """Get highlights from the video."""
        return await self._video_qa_request(request, "highlight")

    @bentoml.api
    async def chapter(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        """Get chapter segments from the video."""
        return await self._video_qa_request(request, "chapter")

    @bentoml.api
    async def summary(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        """Get a summary of the video."""
        return await self._video_qa_request(request, "summary")

# Combined Service
@bentoml.service(
    traffic={"timeout": 300},
    resources={"gpu": 1},
    workers=1,
    concurrency=1
)
class LabARVideoReportingService:
    def __init__(self) -> None:
        self.llava = LLaVAVideoService()
        self.twelvelabs = TwelveLabsAPIService()

    @bentoml.api(route="/llava/video")
    async def analyze_video(self, metadata: VideoMetadata, video: Path) -> VideoResponse:
        return await self.llava.analyze_video(metadata, video)

    @bentoml.api(route="/llava/image")
    async def analyze_image(self, metadata: ImageMetadata, image: Path = None) -> ImageResponse:
        return await self.llava.analyze_image(metadata, image)

    @bentoml.api(route="/twelvelabs/summary")
    async def summary(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        return await self.twelvelabs.summary(request)

    @bentoml.api(route="/twelvelabs/highlights")
    async def highlight(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        return await self.twelvelabs.highlight(request)

    @bentoml.api(route="/twelvelabs/chapters")
    async def chapter(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        return await self.twelvelabs.chapter(request)