# VideoAgent Frame Analysis Tool

The **VideoAgent Frame Analysis Tool** is designed to analyze videos by identifying and matching specific actions or events using advanced machine learning models. It supports both local video files and remote video URLs, utilizing CLIP and LAVIS BLIP models for accurate frame analysis and action detection.

## Features
- **Multimodal Video Analysis**: Combines CLIP and LAVIS BLIP models for robust frame analysis
- **Chronological Action Detection**: Identifies actions in sequence, respecting their temporal order
- **Remote Video Support**: Process videos from both local files and HTTP/HTTPS URLs
- **REST API Interface**: Easy-to-use API endpoints for video processing
- **Detailed JSON Reports**: Generates comprehensive analysis reports with frame matches and timestamps
- **Adaptive Similarity Matching**: Uses dynamic thresholds based on average similarity scores

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- FFmpeg for video processing

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd app/VideoAgent/frame-based

# Install dependencies
pip install -r requirements.txt
```

## Usage

### REST API
The tool provides a REST API interface using BentoML. Start the server with:

```bash
bentoml serve app:VideoAnalyzerService --reload --port 3000
```

#### Process a Video
```bash
curl -X POST http://localhost:3000/videoQA \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "https://example.com/video.mp4",
    "descriptions": [
      "Person picks up beaker",
      "Person pours liquid",
      "Person places beaker down"
    ],
    "fps": 2.0,
    "model_type": "combined",
    "generate_report": true,
    "record_top_k_frames": 20
  }'
```

#### Get Analysis Details
```bash
curl http://localhost:3000/report/{result_dir}
```

### Command Line Interface
For direct command-line usage:

```bash
python main.py process-video \
  --video_path PATH_TO_VIDEO \
  --descriptions "Action 1" "Action 2" "Action 3" \
  --fps 2.0 \
  --model-type combined \
  --generate-report
```

### Parameters

#### API Parameters
- `video_path`: Local file path or HTTP/HTTPS URL to the video
- `descriptions`: List of action descriptions to identify (in chronological order)
- `fps`: Frames per second to process (default: 2.0)
- `model_type`: Model to use - "clip", "blip", or "combined" (default: "combined")
- `generate_report`: Whether to generate a detailed report (default: true)
- `record_top_k_frames`: Number of top matching frames to record (default: 20)

#### CLI Options
- `--video_path`: Path to input video (local file or URL)
- `--descriptions`: One or more action descriptions (in order)
- `--fps`: Frame extraction rate (default: 2.0)
- `--model-type`: Model selection (default: combined)
- `--generate-report`: Generate analysis report
- `--record-top-k-frames`: Number of top frames to record

## Output Format
The tool generates a JSON report containing:
```json
{
  "timestamp": "2025-02-20T09:54:17Z",
  "actions": [
    {
      "description": "Action description",
      "segment": {
        "start_frame": 10,
        "end_frame": 20,
        "start_time": 0.33,
        "end_time": 0.67,
        "confidence": 0.85
      },
      "frame_matches": [
        {
          "frame_number": 15,
          "similarity": 0.92,
          "timestamp": 0.5
        }
      ]
    }
  ]
}
```

## Model Details
- **CLIP**: Efficient for matching visual content with text descriptions
- **LAVIS BLIP**: Advanced vision-language model for detailed frame analysis
- **Combined**: Uses both models for more robust results

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
[Specify License]

---
For more details, see the [API Documentation](docs/api.md) or [CLI Guide](docs/cli.md).
