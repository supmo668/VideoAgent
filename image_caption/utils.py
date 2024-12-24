import yaml
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Tuple
from dotenv import load_dotenv
from db_utils import init_cache_db
from urllib.request import urlretrieve

load_dotenv()

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_processing_environment(
    video_path: str, cache_db_path: str, keep_temp_dir: bool = True, suffix: str = ""
) -> Tuple[Path, Path]:
    """Set up the processing environment including temporary and results directories."""
    
    # Initialize cache DB
    init_cache_db(cache_db_path)

    # Assume video_path is an HTTP URL
    video_name = Path(video_path).stem
    temp_dir = Path(__file__).parent.parent / f"temp_frames-{video_name}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download video from HTTP URL
    if video_path.startswith("http"):
        video_name = Path(video_path).stem
        local_video_path = temp_dir / video_name
        if not local_video_path.exists():
            urlretrieve(video_path, str(local_video_path))
            video_path = str(local_video_path)
    
    # Create result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).parent.parent / "results" / f"{video_name}_{timestamp}{suffix}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    return temp_dir, result_dir


def cleanup_environment(temp_dir, keep_temp_dir):
    """Clean up temporary directory if needed."""
    if not keep_temp_dir:
        shutil.rmtree(temp_dir)
