# LLaVA Video Analysis Service

This service provides video and image analysis capabilities using the LLaVA-Video-7B-Qwen2 model. It's built with BentoML and supports both file uploads and URL inputs.

## Features

- Video analysis with frame-by-frame processing
- Image analysis
- Support for both file uploads and URLs
- Custom prompts for analysis
- Memory-efficient processing with chunked video frame handling
- 4-bit quantization for efficient model loading

## Installation

1. Create the conda environment:
```bash
conda env create -f llava-video-service/llava_environ.yml
```

2. Activate the environment:
```bash
conda activate llava
```

## Usage

### Starting the Service

Navigate to the llava-video-service directory and run:
```bash
bentoml serve service:LLaVAVideoService
```

The service will start on http://localhost:3000 by default.

### API Endpoints

#### 1. Video Analysis

**Endpoint:** `/analyze_video`

Supports two methods of input:

- **File Upload:**
```bash
curl -X POST http://localhost:3000/analyze_video \
  -H "Content-Type: multipart/form-data" \
  -F "video=@./path/to/your/video.mp4" \
  -F "metadata={\"prompt\":\"Please describe what is happening in this video\"};type=application/json"
```

- **Video URL:**
```bash
curl -X POST http://localhost:3000/analyze_video \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "metadata": {"prompt": "Please describe what is happening in this video"}
  }'
```

Response format:
```json
{
  "response": "Description of the video content",
  "video_info": {
    "frame_count": 16,
    "video_duration": 10.5,
    "frame_times": [0.0, 0.7, 1.4, ...]
  },
  "status": "success",
  "error": null
}
```

#### 2. Image Analysis

**Endpoint:** `/analyze_image`

Supports two methods of input:

- **File Upload:**
```bash
curl -X POST http://localhost:3000/analyze_image \
  -H "Content-Type: multipart/form-data" \
  -F "image=@./path/to/your/image.jpg" \
  -F "metadata={\"prompt\":\"Please describe what is in this image\"};type=application/json"
```

- **Image URL:**
```bash
curl -X POST http://localhost:3000/analyze_image \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "metadata": {"prompt": "Please describe what is in this image"}
  }'
```

Response format:
```json
{
  "response": "Description of the image content",
  "status": "success",
  "error": null
}
```

### Request Parameters

Both endpoints accept the following parameters:

- `prompt` (optional): Custom instruction for analysis. If not provided, a default prompt will be used.
- For file uploads: Use multipart/form-data with either `video` or `image` field
- For URLs: Use JSON with either `video_url` or `image_url` field

## Model Details

The service uses the LLaVA-Video-7B-Qwen2 model with the following configurations:

- 4-bit quantization for efficient memory usage
- Flash Attention 2 for improved performance
- Automatic CUDA memory management
- Vision tower in float16 for optimized GPU memory usage

## Custom Modules

The service includes custom modules in the `llava_custom` directory that replace original LLaVA modules for compatibility:

- `builder.py`: Modified model builder for better quantization support
- Additional configuration files for model compatibility

## Error Handling

The service provides detailed error messages in the response when issues occur:

```json
{
  "response": "",
  "status": "failed",
  "error": "Detailed error message"
}
```

Common error cases:
- Invalid file format
- Failed URL download
- Invalid URL format
- Processing errors

## Resource Requirements

- CUDA-capable GPU
- Minimum 12GB GPU memory (recommended)
- Python 3.10+
- Dependencies as specified in llava_environ.yml
