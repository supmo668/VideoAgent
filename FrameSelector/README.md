# Frame Selector

A simple GUI application for viewing and navigating through frames of a video stored in an S3 bucket.

## Features

- Video playback interface with frame-by-frame navigation
- Frame slider for quick navigation
- Frame counter display
- Arrow key navigation (Left/Right)
- S3 bucket video loading

## Setup

1. Create a `.env` file in the project root with your AWS credentials:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET=your_bucket_name
VIDEO_KEY=path/to/your/video.mp4
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python main.py
```

2. Click "Load S3 Video" to load the video from your S3 bucket
3. Navigate through the video:

   - Use the slider to jump to specific frames
   - Use Left/Right arrow keys to move frame by frame
   - View the current frame number at the bottom of the interface

## Requirements

- Python 3.7+
- AWS account with S3 access
- Required Python packages (specified in requirements.txt)
