# video_utils.py
import os
import cv2
from tqdm import tqdm
from typing import List, Tuple


def get_sample_freq(fps: float) -> int:
    """Determine sample frequency based on video FPS.
    If FPS <= 2, sample every frame. Otherwise, sample at FPS rate.
    """
    if fps <= 2:
        return 1
    elif fps > 2 and fps <= 10:
        return int(fps/2)
    return int(2)


def extract_frames(video_path: str, output_dir: str, fps: float = 2.0) -> List[Tuple[int, str]]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_freq = get_sample_freq(fps)
    
    print(f"Total frames in video: {total_frames}")
    print(f"Target FPS: {fps}")
    print(f"Sampling frequency: {sample_freq}")
    
    saved_frames = []
    pbar = tqdm(total=total_frames, desc="Extracting frames")

    frame_count = 0
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
        pbar.update(1)
    
    pbar.close()
    cap.release()
    print(f"Extracted {len(saved_frames)} frames")
    return saved_frames
