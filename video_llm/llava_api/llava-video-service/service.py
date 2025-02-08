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
from typing import Any, Dict, List, Optional, Union, Tuple, Generator
import yaml

# Third-party imports
import aiohttp
import bentoml
import numpy as np
import torch
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
load_dotenv()
from PIL import Image
from transformers import TextStreamer, StoppingCriteria, AutoConfig
from decord import VideoReader, cpu

# Local imports
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from prompt_library import PromptLibrary
from tracing import TracingConfig
from langsmith import traceable
from logging_config import get_logger, log_function_call, sanitize_log_data

# Initialize logger
logger = get_logger()

# Load configuration
def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()
DEFAULTS = CONFIG["defaults"]
MODEL_CONFIG = CONFIG["model"]
SERVICE_CONFIG = CONFIG["service"]
VIDEO_CONFIG = CONFIG["video"]
ENV_CONFIG = CONFIG["environment"]

# Initialize tracing
tracing_config = TracingConfig()
tracing_config.enable_tracing()

# Model request/response classes
class ImageMetadata(BaseModel):
    image_url: Optional[str] = Field(None, description="URL of the image to analyze")
    prompt: str = Field(DEFAULTS["image"]["prompt"], description="Prompt for image analysis")
    temperature: float = Field(DEFAULTS["image"]["temperature"], description="Temperature for generation")
    max_new_tokens: int = Field(DEFAULTS["image"]["max_new_tokens"], description="Maximum number of new tokens to generate")
    context_len: int = Field(DEFAULTS["image"]["context_len"], description="Context length for generation")

class   VideoMetadata(BaseModel):
    video_url: Optional[str] = Field(None, description="URL of the video to analyze")
    prompt: str = Field(DEFAULTS["video"]["prompt"], description="Prompt for video analysis")
    temperature: float = Field(DEFAULTS["video"]["temperature"], description="Temperature for generation")
    max_new_tokens: int = Field(DEFAULTS["video"]["max_new_tokens"], description="Maximum number of new tokens to generate")
    context_len: int = Field(DEFAULTS["video"]["context_len"], description="Context length for generation")
    fps: int = Field(DEFAULTS["video"]["fps"], description="Frames per second to extract")

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
    resources=SERVICE_CONFIG["resources"],
    traffic=SERVICE_CONFIG["traffic"],
    worker=1,
    concurrency=1
)
class LLaVAVideoService:
    _model_instance: tuple = None
    _model_lock = asyncio.Lock()
    
    def __init__(self) -> None:
        logger.info("Initializing LLaVAVideoService")
        self.pretrained = MODEL_CONFIG["pretrained"]
        self.model_name = MODEL_CONFIG["model_name"]
        self.device = MODEL_CONFIG["device"]
        
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ENV_CONFIG["cuda_alloc_conf"]
        
        logger.debug(f"Loading model with config: {sanitize_log_data(MODEL_CONFIG)}")
        
        if self._model_instance is None:
            self.tokenizer, self.model, self.image_processor, self.context_len = self._load_model()
            self._model_instance = (self.tokenizer, self.model, self.image_processor, self.context_len)
        else:
            self.tokenizer, self.model, self.image_processor, self.context_len = self._model_instance
        
        if hasattr(self.model, 'get_vision_tower'):
            vision_tower = self.model.get_vision_tower()
            if hasattr(vision_tower, 'vision_tower'):
                vision_tower.vision_tower = vision_tower.vision_tower.to(dtype=torch.float16)
        
        self.model.eval()
        logger.info("LLaVAVideoService initialization completed")

    @log_function_call(skip_args=True)  # Skip args due to large model objects
    def _load_model(self):
        from llava_custom.builder import load_pretrained_model        
        # # Custom configuration to handle meta parameters
        # config = AutoConfig.from_pretrained(self.pretrained)
        # config.use_cache = True
        # config.torch_dtype = torch.float16
        
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.pretrained, 
            None, 
            self.model_name,
            torch_dtype=getattr(torch, MODEL_CONFIG["torch_dtype"]),
            device_map=MODEL_CONFIG["device_map"],
            load_4bit=MODEL_CONFIG["load_4bit"],
        )
        model.eval()
        return tokenizer, model, image_processor, context_len

    @log_function_call()
    @contextmanager
    def _temp_file_context(self, content: bytes, suffix: str = "") -> Generator[Path, None, None]:
        """Create a temporary file with the given content and yield its path."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            try:
                tmp.write(content)
                tmp.flush()
                yield Path(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception as e:
                    logger.error(f"Error deleting temporary file: {str(e)}")

    @log_function_call()
    async def _download_file(self, url: str) -> bytes:
        """Download a file from a URL and return its content as bytes."""
        logger.info(f"Downloading file from {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download file: {response.status}")
                return await response.read()

    @log_function_call()
    async def _process_video(self, video_path: str | Path, fps: int = DEFAULTS["video"]["fps"]) -> Tuple[np.ndarray, List[float], float]:
        """Process video using decord.
        
        Args:
            video_path: Path to the video file
            fps: Frames per second to extract
            
        Returns:
            Tuple containing:
            - Numpy array of video frames
            - List of frame times
            - Total video duration
        """
        logger.debug(f"Processing video: {video_path}")
        try:
            # Convert to string path if it's a Path object
            video_path = str(video_path) if isinstance(video_path, Path) else video_path
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            duration = total_frames / vr.get_avg_fps()
            
            # Calculate frame indices based on desired fps
            frame_indices = []
            frame_times = []
            target_interval = vr.get_avg_fps() / fps
            current_idx = 0
            
            while current_idx < total_frames:
                frame_indices.append(int(current_idx))
                frame_times.append(current_idx / vr.get_avg_fps())
                current_idx += target_interval
            
            # Read frames
            frames = vr.get_batch(frame_indices).asnumpy()
            logger.debug(f"Processed {len(frames)} frames from video")
            return frames, frame_times, duration
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    @log_function_call()
    async def _process_image(self, image_path: Union[str, bytes, Path], is_bytes: bool = False) -> Tuple[np.ndarray, List[float], float]:
        """Process an image from either a file path or bytes content.
        
        Args:
            image_path: Either a file path (str/Path) or image content (bytes)
            is_bytes: Whether the input is bytes content
            
        Returns:
            Tuple containing:
            - Numpy array of single image frame
            - Empty list of frame times
            - Zero video time (single frame)
        """
        try:
            if is_bytes:
                image = np.array(Image.open(io.BytesIO(image_path)))
            else:
                # Convert Path to string if necessary
                image_path = str(image_path) if isinstance(image_path, Path) else image_path
                image = np.array(Image.open(image_path))
            return np.expand_dims(image, 0), [], 0
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise Exception(f"Error processing image: {str(e)}")

    @log_function_call()
    def process_frames_in_chunks(self, frames: np.ndarray, chunk_size: int = 4) -> torch.Tensor:
        """Process frames in chunks to manage memory."""
        processed_chunks = []
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i:i + chunk_size]
            chunk_tensor = self.image_processor.preprocess([Image.fromarray(frame) for frame in chunk], return_tensors="pt")["pixel_values"]
            chunk_tensor = chunk_tensor.to(device=torch.device(self.device), dtype=torch.float16)
            processed_chunks.append(chunk_tensor)
            torch.cuda.empty_cache()
        logger.info(f"Processed frames: to {len(processed_chunks)} chunks")
        return torch.cat(processed_chunks, dim=0)

    @log_function_call(skip_args=True)  # Skip args due to potentially large frame data
    async def _generate_response(
        self, 
        prompt: str, 
        frames: np.ndarray, 
        metadata: Union[ImageMetadata, VideoMetadata],
        frame_time: List[float] = None,
        video_time: float = None
    ) -> str:
        """Generate response using the model.
        
        Args:
            prompt: The prompt text
            frames: Numpy array of frames (single frame for image, multiple for video)
            metadata: Metadata containing parameters
            frame_time: List of timestamps for each frame
            video_time: Total duration of the video
            
        Returns:
            Generated response text
        """
        logger.info("Generating response...")
        processed_frames = self.process_frames_in_chunks(frames)
        conv = copy.deepcopy(conv_templates["qwen_1_5"])

        # Add time instructions for video
        if len(frames) > 1:  # It's a video
            logger.info(f"Processing video with {len(frames)} frames")
            time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly sampled from it. These frames are located at {frame_time}."
            question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt}"
        else:
            question = DEFAULT_IMAGE_TOKEN + f"Answer the following based on the image:\n{prompt}"
        
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize input
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        input_ids = input_ids.to(device=torch.device(self.device))

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device=torch.device(self.device))

        with torch.inference_mode():
            logger.info("Model inference...")
            output_ids = self.model.generate(
                input_ids,
                images=[processed_frames],  # Wrap in list as per template
                attention_mask=attention_mask,
                modalities=["video"] if len(frames) > 1 else ["image"],
                do_sample=True,  # Match template settings
                temperature=metadata.temperature,
                max_new_tokens=metadata.max_new_tokens
            )

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        torch.cuda.empty_cache()
        return outputs

    @bentoml.api
    @log_function_call()
    async def analyze_video_file(self, metadata: VideoMetadata, video: Path) -> VideoResponse:
        """Analyze a video file."""
        try:
            frames, frame_times, duration = await self._process_video(video, metadata.fps)
            response = await self._generate_response(metadata.prompt, frames, metadata, frame_times, duration)
            return VideoResponse(response=response)
        except Exception as e:
            logger.error(f"Error analyzing video file: {str(e)}")
            return VideoResponse(
                status="error",
                error=str(e),
            )
    
    @bentoml.api
    @log_function_call()
    async def analyze_video(self, metadata: VideoMetadata, video: Path = None) -> VideoResponse:
        try:
            if metadata.video_url:
                video_content = await self._download_file(metadata.video_url)
                with self._temp_file_context(video_content, ".mp4") as temp_video:
                    result = await self.analyze_video_file(metadata, temp_video)
                    return result
            else:
                raise ValueError("video_url is required for analyze_video")
        except Exception as e:
            return VideoResponse(
                status="error",
                error=str(e),
            )

    @bentoml.api
    @log_function_call()
    async def analyze_image_file(self, metadata: ImageMetadata, image: Path) -> ImageResponse:
        """Analyze a single image from a file."""
        try:
            logger.info(f"Processing image file: {image}")
            frames, frame_time, video_time = await self._process_image(image, is_bytes=False)
            response = await self._generate_response(metadata.prompt, frames, metadata)
            return ImageResponse(response=response)
        except Exception as e:
            logger.error(f"Error analyzing image file: {str(e)}")
            return ImageResponse(
                status="error",
                error=str(e),
            )

    @bentoml.api
    @log_function_call()
    async def analyze_image(self, metadata: ImageMetadata, image: Path = None) -> ImageResponse:
        """Analyze a single image from URL."""
        try:
            if metadata.image_url:
                image_content = await self._download_file(metadata.image_url)
                with self._temp_file_context(image_content, ".jpg") as temp_image:
                    return await self.analyze_image_file(metadata, temp_image)
            else:
                raise ValueError("image_url is required for analyze_image")
        except Exception as e:
            return ImageResponse(
                status="error",
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
        logger.info("Initializing TwelveLabsAPIService")
        self.BASE_URL = DEFAULTS["twelvelabs"]["base_url"]
        self.API_KEY = os.getenv("TWELVE_LABS_API_KEY")
        if not self.API_KEY:
            logger.error("TWELVE_LABS_API_KEY environment variable is not set")
            raise ValueError("TWELVE_LABS_API_KEY environment variable is not set")
        self.prompt_library = PromptLibrary()
        logger.info("TwelveLabsAPIService initialization completed")

    @log_function_call()
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
                    logger.warning("Prompt library not available, using default prompt")

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

    @bentoml.api
    @log_function_call()
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
                            logger.info(f"Video '{video_title}' already exists in index with ID: {video['_id']}")
                            return TwelveLabsResponse(
                                task_id=None,
                                video_id=video["_id"]
                            )
                
            logger.info(f"Proceeding to upload.")
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
    @log_function_call()
    async def highlight(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        """Get highlights from the video."""
        return await self._video_qa_request(request, "highlight")

    @bentoml.api
    @log_function_call()
    async def chapter(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        """Get chapter segments from the video."""
        return await self._video_qa_request(request, "chapter")

    @bentoml.api
    @log_function_call()
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
        logger.info("Initializing LabARVideoReportingService")
        self.llava = LLaVAVideoService()
        self.twelvelabs = TwelveLabsAPIService()
        logger.info("LabARVideoReportingService initialization completed")

    
    @bentoml.api(route="/llava/video")
    async def analyze_video(self, metadata: VideoMetadata, video: Path = None) -> VideoResponse:
        return await self.llava.analyze_video(metadata, video)

    
    @bentoml.api(route="/llava/video-file")
    async def analyze_video_file(self, metadata: VideoMetadata, video: Path) -> VideoResponse:
        return await self.llava.analyze_video_file(metadata, video)

    
    @bentoml.api(route="/llava/image")
    async def analyze_image(self, metadata: ImageMetadata, image: Path = None) -> ImageResponse:
        return await self.llava.analyze_image(metadata, image)

    
    @bentoml.api(route="/llava/image-file")
    async def analyze_image_file(self, metadata: ImageMetadata, image: Path) -> ImageResponse:
        return await self.llava.analyze_image_file(metadata, image)

    
    @bentoml.api(route="/twelvelabs/summary")
    async def summary(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        return await self.twelvelabs.summary(request)

    
    @bentoml.api(route="/twelvelabs/highlights")
    async def highlight(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        return await self.twelvelabs.highlight(request)

    
    @bentoml.api(route="/twelvelabs/chapters")
    async def chapter(self, request: TwelveLabsRequest) -> TwelveLabsResponse:
        return await self.twelvelabs.chapter(request)