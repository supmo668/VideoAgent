from abc import ABC, abstractmethod
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import re
import re

from db_utils import init_cache_db, get_cached_embedding, save_embedding_to_cache
from embed_utils import get_text_embedding_openai, get_frame_description
from clip_utils import get_frame_embedding_clip, get_text_embedding_clip

def cosine_similarity(embedding1, embedding2):
    # Assuming embeddings are numpy arrays
    return (embedding1 @ embedding2) / (
        (embedding1 @ embedding1) ** 0.5 * (embedding2 @ embedding2) ** 0.5
    )

class EmbeddingProcessor(ABC):
    @abstractmethod
    def get_text_embedding(self, text: str) -> Any:
        pass

    @abstractmethod
    def get_frame_embedding(self, frame_path: str) -> Any:
        pass

class OpenAIEmbeddingProcessor(EmbeddingProcessor):
    def __init__(
        self, system_prompt: str, vision_prompt_template: str, cache_db_path: str
        ):
        self.system_prompt = system_prompt
        self.vision_prompt_template = vision_prompt_template
        self.cache_db_path = cache_db_path

    def get_text_embedding(self, text: str) -> Any:
        return get_text_embedding_openai(text)

    def get_frame_description(self, frame_path: str) -> str:
        self.frame_desc = get_frame_description(
            self.system_prompt, 
            self.vision_prompt_template, 
            frame_path
        )
        
    def get_frame_embedding(self, frame_path: str) -> Any:
        video_path = str(Path(frame_path).parent.parent)
        frame_number = int(Path(frame_path).stem.split('_')[1])
        cached_embedding = get_cached_embedding(self.cache_db_path, video_path, frame_number, '', 'openai')
        if cached_embedding is not None:
            return cached_embedding
        if hasattr(self, 'frame_desc'):
            frame_description = self.frame_desc
        else:
            frame_description = self.get_frame_description(frame_path)
        frame_embedding = self.get_text_embedding(frame_description)
        save_embedding_to_cache(self.cache_db_path, video_path, frame_number, '', 'openai', frame_embedding)
        return frame_embedding

class ClipEmbeddingProcessor(EmbeddingProcessor):
    def __init__(self, cache_db_path: str):
        self.cache_db_path = cache_db_path

    def get_text_embedding(self, text: str) -> Any:
        return get_text_embedding_clip(text)

    def get_frame_embedding(self, frame_path: str) -> Any:
        video_path = str(Path(frame_path).parent.parent)
        frame_number = int(Path(frame_path).stem.split('_')[1])
        cached_embedding = get_cached_embedding(self.cache_db_path, video_path, frame_number, '', 'clip')
        if cached_embedding is not None:
            return cached_embedding

        frame_embedding = get_frame_embedding_clip(frame_path)
        save_embedding_to_cache(self.cache_db_path, video_path, frame_number, '', 'clip', frame_embedding)
        return frame_embedding

def process_frames_for_comparison(
    frames: List[Tuple[int, str]],
    user_desc: str,
    processor: EmbeddingProcessor,
) -> List[Tuple[str, float, int]]:
    """Process frames for a given user_desc using the specified embedding processor.
    
    Returns a list of tuples, where the first element of the tuple is the file path to the frame,
    the second element is the cosine similarity between the frame embedding and the user_desc embedding,
    and the third element is the frame number.
    """
    user_desc_embedding = processor.get_text_embedding(user_desc)
    similarities = []
    
    for frame_number, frame_path in tqdm(frames, desc=f"Processing frames for user_desc: {user_desc}"):
        frame_embedding = processor.get_frame_embedding(frame_path)
        sim = cosine_similarity(user_desc_embedding, frame_embedding)
        similarities.append((frame_path, sim, frame_number))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

def save_and_report_results(
    similarities: List[Tuple[str, float, int]],
    user_desc: str,
    result_dir: Path,
    record_top_k_frames: int,
    model_type: str
) -> Tuple[str, List[Dict[str, Any]]]:
    model_type: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """Save results and report them to console."""

    # Sanitize the user_desc to create a valid filename
    safe_user_desc = re.sub(r'[<>:"/\\|?*]', '', user_desc)
    # Save top frame
    top_frame_path, _, frame_number = similarities[0]  
    key_frame_path = result_dir / f"top_key_frame_Q-{safe_user_desc}_frame-{frame_number}.png"
    if not key_frame_path.exists():
        shutil.copy(
            top_frame_path, key_frame_path
            top_frame_path, key_frame_path
        )

    top_results = [
        {"frame_path": f, "similarity": s, "frame_number": n} for f, s, n in similarities[:record_top_k_frames]
        {"frame_path": f, "similarity": s, "frame_number": n} for f, s, n in similarities[:record_top_k_frames]
    ]
    results_path = result_dir / f"results_{safe_user_desc}_{model_type}.json"
    with open(results_path, 'w') as f:
        json.dump({"user_desc": user_desc, "top_results": top_results}, f, indent=2)

    return str(key_frame_path), top_results
    return str(key_frame_path), top_results
