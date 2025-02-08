# VideoAgent Captioning Tool

The **VideoAgent Captioning Tool** is designed to facilitate video analysis by leveraging advanced machine learning models for identifying keyframes and extracting meaningful captions. It supports both CLIP-based and OpenAI's text embedding-based methods to analyze videos and identify keyframes that match user-provided descriptions.

## Features
- **CLIP-based Video Analysis**: Utilizes the CLIP model to match video frames with textual descriptions, ideal for identifying specific moments in videos.
- **Frame Caption-based Video Analysis**: Uses OpenAI's text embeddings to generate captions for video frames, allowing for a broader understanding of video content.
- **Keyframe Identification**: Extracts and identifies keyframes from videos based on textual queries, making it easier to locate relevant video segments.
- **Report Generation**: Generates detailed reports in markdown, HTML, and PDF formats, including frame descriptions, similarity scores, and frame numbers.

## Installation
Before you begin, ensure you have **Python 3.7+** installed on your system. Then, follow these steps to set up the VideoAgent Captioning Tool:

### Clone the repository:
```bash
git clone <repository-url>
cd app/VideoAgent/caption
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
The tool is accessed through a command-line interface (CLI). Below are the main commands and their usage:

### 1. Image Embedding
To process a video using the selected embedding model (CLIP, BLIP, or combined), use the following command:

```bash
python main.py image-embed --video_path PATH_TO_VIDEO --descriptions "DESCRIPTION1" "DESCRIPTION2" --model [clip|blip|combined] --fps FRAMES_PER_SECOND --generate-report
```

**Options**:
- `--video_path`: Path to the input video file.
- `--descriptions`: List of descriptions to search for within the video. You can provide multiple descriptions enclosed in quotes.
- `--model`: Choose the embedding model to use (`clip`, `blip`, or `combined`). Default is `blip`.
- `--fps` (optional): Target frames per second for extraction. Default is 30.
- `--generate-report` (optional): Flag to generate a report with results. Reports are saved in markdown, HTML, and PDF formats.

### 2. OpenAI Embedding
For processing a video using OpenAI's embedding model, use the command:

```bash
python main.py openai-embed --video_path PATH_TO_VIDEO --descriptions "DESCRIPTION1" "DESCRIPTION2" --fps FRAMES_PER_SECOND --generate-report
```

The options are similar to those for the Image Embedding command.

### 3. Video Summarization
To generate a summary of the video, use:

```bash
python main.py summarize --video_path PATH_TO_VIDEO --fps FRAMES_PER_SECOND
```

## Updated Structure
- `main.py` and `app.py` are now located at the top level of the `frame-based` directory for easier access to CLI and BentoML service.

## Configuration
The tool allows for some customization through a `config.yaml` file. You can adjust settings related to the models, embedding processors, and other system-level configurations. Ensure this file is correctly set up before running the tool.

## Contributing
Contributions to the VideoAgent Captioning Tool are welcome. Please ensure to follow the project's coding standards and submit pull requests for any new features or bug fixes.

## License
[Specify License Here]

---

Ensure to replace `<repository-url>` with the actual
