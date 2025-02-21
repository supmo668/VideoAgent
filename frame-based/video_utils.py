# video_utils.py
import os
import cv2
from tqdm import tqdm
from typing import List, Tuple


def get_sample_freq(fps: float, n_frames: int) -> int:
    """Determine sample frequency based on video FPS and total frames.
    Adjust sampling to balance performance and frame extraction.
    """
    if n_frames <= 20:
        return 1
    elif fps <= 2:
        return 1
    elif fps <= 10:
        return max(1, int(fps / 2))
    elif n_frames <= 100:
        return max(1, int(fps / 1.5))
    else:
        return max(1, int(fps)*max(1, n_frames//1000))

def extract_frames(video_path: str, output_dir: str, fps: float = 2.0) -> List[Tuple[int, str]]:
    """
    Extract frames from a video at a specified FPS rate using direct frame positioning.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        fps: Target frames per second to extract
        
    Returns:
        List of tuples containing (frame_id, frame_path)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    sample_freq = get_sample_freq(fps, total_frames)
    
    # Verify video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    # Calculate number of frames that will be extracted
    num_frames_to_extract = total_frames // sample_freq
    if total_frames % sample_freq != 0:
        num_frames_to_extract += 1
    
    print(f"Total frames in video: {total_frames}")
    print(f"Video FPS: {video_fps}")
    print(f"Target sampling FPS: {fps}")
    print(f"Sampling frequency: {sample_freq}")
    print(f"Will extract {num_frames_to_extract} frames")
    
    saved_frames = []
    pbar = tqdm(total=num_frames_to_extract, desc="Extracting frames")

    try:
        for frame_id in range(num_frames_to_extract):
            # Calculate exact frame position
            frame_pos = frame_id * sample_freq
            if frame_pos >= total_frames:
                break
                
            # Set frame position and read frame
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos):
                print(f"Warning: Could not set frame position to {frame_pos}")
                continue
                
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Warning: Could not read frame at position {frame_pos}")
                continue
                
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{frame_id:06d}.png")
            if cv2.imwrite(frame_path, frame):
                saved_frames.append((frame_id, frame_path))
                pbar.update(1)
            else:
                print(f"Warning: Could not save frame to {frame_path}")
    
    except Exception as e:
        print(f"Error during frame extraction: {str(e)}")
        raise
    
    finally:
        pbar.close()
        cap.release()
    
    print(f"Successfully extracted {len(saved_frames)} frames")
    return saved_frames
