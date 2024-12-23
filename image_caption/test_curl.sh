curl -X POST http://localhost:3000/analyze_video \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "path/to/your/video.mp4",
    "questions": ["What is happening in the video?"],
    "model_type": "clip",
    "sample_freq": 30,
    "record_top_k_frames": 20,
    "generate_report": true
  }'

  curl -X POST http://localhost:3000/download_report \
  -H "Content-Type: application/json" \
  -d '"result_dir_path_from_previous_response"' \
  --output report.md