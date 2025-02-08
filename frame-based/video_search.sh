#!/bin/bash

# CLIP Embedding Search
curl -X POST http://localhost:8000/process_video_clip \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "https://myimagebucketlabar.s3.us-east-2.amazonaws.com/session_20241219_213106/session_video.mp4",
    "questions": ["What is happening in this video?", "Describe the main action"],
    "sample_freq": 30,
    "record_top_k_frames": 5
  }'

# OpenAI Embedding Search
curl -X POST http://localhost:8000/process_video_openai \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "https://myimagebucketlabar.s3.us-east-2.amazonaws.com/session_20241219_213106/session_video.mp4",
    "questions": ["What is happening in this video?", "Describe the main action"],
    "sample_freq": 30,
    "record_top_k_frames": 5
  }'

# Make the script executable
chmod +x /workspace/VideoAgent/image_caption/video_search.sh
