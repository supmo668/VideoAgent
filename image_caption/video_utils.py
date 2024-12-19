# video_utils.py
import os
import cv2

def extract_frames(video_path: str, sample_freq: int, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_freq == 0:
            frame_id = frame_count // sample_freq
            frame_path = os.path.join(output_dir, f"frame_{frame_id:06d}.png")
            cv2.imwrite(frame_path, frame)
            saved_frames.append((frame_id, frame_path))
        frame_count += 1

    cap.release()
    return saved_frames
