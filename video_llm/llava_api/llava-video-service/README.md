# LLaVA Video Analysis Service

This repository contains a BentoML service for video and image analysis using the LLaVA-Video-7B-Qwen2 model.

## Setup

1. Create conda environment using the exported environment file:
```bash
conda env create -f environment_export.yml
conda activate llava-video-env
```

## Running the Service

1. Start the service:
```bash
bentoml serve service:LLaVAVideoService
```

2. The service will be available at `http://localhost:3000`

## API Usage

The service exposes two REST API endpoints for video and image analysis:

### 1. Video Analysis Endpoint

- Endpoint: `/analyze_video`
- Method: POST
- Input: Video file
- Optional Query Parameter: `prompt` (string) - Custom prompt for video analysis

Example curl request:
```bash
curl -X POST \
  "http://localhost:3000/analyze_video?prompt=Describe%20this%20video" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@path/to/your/video.mp4"
```

### 2. Image Analysis Endpoint

- Endpoint: `/analyze_image`
- Method: POST
- Input: Image file (JPEG, PNG)
- Optional Query Parameter: `prompt` (string) - Custom prompt for image analysis

Example curl request:
```bash
curl -X POST \
  "http://localhost:3000/analyze_image?prompt=Describe%20this%20image" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/your/image.jpg"
```

## Requirements

- CUDA-capable GPU
- CUDA Toolkit 11.8
- At least 16GB GPU memory (recommended)
- Python 3.10+

## Model Information

This service uses the LLaVA-Video-7B-Qwen2 model, which is optimized for both video and image understanding and analysis. The model is loaded in 4-bit quantization to reduce memory usage while maintaining good performance.

## Environment

The exact environment dependencies are specified in `environment_export.yml`. This includes:
- PyTorch 2.1.2
- CUDA Toolkit 11.8
- BentoML 1.1.9
- Other necessary dependencies for video and image processing

## License

Please refer to the LLaVA-NeXT repository for model licensing information.
