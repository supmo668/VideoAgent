from typing import List, Tuple, Dict, Any, Optional, Union
import os
import re
import json
import shutil
import base64
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from abc import ABC, abstractmethod
import torch
from PIL import Image
import clip
from lavis.models import load_model_and_preprocess
from datetime import datetime

from models import (
    FrameMatch,
    ActionSegment,
    ActionResult,
    ProcessingResults
)
from db_utils import init_cache_db, get_cached_embedding, save_embedding_to_cache
sys.path.append("../models")
from llm.action_steps import ImageFrameStepDescriptor

class ModelType(Enum):
    CLIP = "clip"
    BLIP = "blip"
    COMBINED = "combined"

class MultimodalEmbedder:
    """Class for handling multimodal embeddings using CLIP and BLIP models."""
    
    def __init__(self, model_type: Union[ModelType, str] = ModelType.COMBINED):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = ModelType(model_type) if isinstance(model_type, str) else model_type
        
        # Initialize CLIP if needed
        if self.model_type in [ModelType.CLIP, ModelType.COMBINED]:
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", device=self.device)
        
        # Initialize BLIP if needed
        if self.model_type in [ModelType.BLIP, ModelType.COMBINED]:
            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip_feature_extractor", 
                model_type="base", 
                is_eval=True, 
                device=self.device
            )
    
    def _get_frame_embedding_clip(self, frame_path: str) -> np.ndarray:
        """Get CLIP embedding for a frame."""
        image = Image.open(frame_path).convert('RGB')
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]
    
    def _get_frame_embedding_blip(self, frame_path: str) -> np.ndarray:
        """Get BLIP embedding for a frame."""
        raw_image = Image.open(frame_path).convert('RGB')
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            sample = {"image": image}
            features = self.blip_model.extract_features(sample, mode="image")
            # Use the projected embeddings and take the first token ([CLS])
            image_features = features.image_embeds_proj[:, 0, :]
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]
    
    def _get_text_embedding_clip(self, text: str) -> np.ndarray:
        """Get CLIP embedding for text."""
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]
    
    def _get_text_embedding_blip(self, text: str) -> np.ndarray:
        """Get BLIP embedding for text."""
        text_input = self.txt_processors["eval"](text)
        
        with torch.no_grad():
            sample = {"text_input": [text_input]}
            features = self.blip_model.extract_features(sample, mode="text")
            # Use the projected embeddings and take the first token ([CLS])
            text_features = features.text_embeds_proj[:, 0, :]
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]
    
    def get_frame_embedding(self, frame_path: str) -> np.ndarray:
        """Get embedding for a frame using specified model(s)."""
        if self.model_type == ModelType.CLIP:
            return self._get_frame_embedding_clip(frame_path)
        elif self.model_type == ModelType.BLIP:
            return self._get_frame_embedding_blip(frame_path)
        else:  # COMBINED
            clip_embedding = self._get_frame_embedding_clip(frame_path)
            blip_embedding = self._get_frame_embedding_blip(frame_path)
            return np.concatenate([clip_embedding, blip_embedding])
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using specified model(s)."""
        if self.model_type == ModelType.CLIP:
            return self._get_text_embedding_clip(text)
        elif self.model_type == ModelType.BLIP:
            return self._get_text_embedding_blip(text)
        else:  # COMBINED
            clip_embedding = self._get_text_embedding_clip(text)
            blip_embedding = self._get_text_embedding_blip(text)
            return np.concatenate([clip_embedding, blip_embedding])

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )

class EmbeddingProcessor(ABC):
    """Base class for embedding processors."""
    def __init__(self, cache_db_path: str):
        self.cache_db_path = cache_db_path

    def get_cached_embedding(self, video_path: str, frame_number: int, model_type: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        return get_cached_embedding(self.cache_db_path, video_path, frame_number, '', model_type)

    def save_embedding_to_cache(self, video_path: str, frame_number: int, model_type: str, embedding: np.ndarray):
        """Save embedding to cache."""
        save_embedding_to_cache(self.cache_db_path, video_path, frame_number, '', model_type, embedding)

    @abstractmethod
    def get_text_embedding(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_frame_embedding(self, frame_path: str) -> np.ndarray:
        pass

class MultimodalProcessor(EmbeddingProcessor):
    """Processor that uses CLIP and/or BLIP models for embeddings."""
    def __init__(self, cache_db_path: str, model_type: Union[ModelType, str] = ModelType.COMBINED):
        super().__init__(cache_db_path)
        self.embedder = MultimodalEmbedder(model_type)
        self.model_type = self.embedder.model_type.value

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using selected model(s)."""
        return self.embedder.get_text_embedding(text)

    def get_frame_embedding(self, frame_path: str) -> np.ndarray:
        """Get frame embedding using selected model(s)."""
        video_path = str(Path(frame_path).parent.parent)
        frame_number = int(Path(frame_path).stem.split('_')[1])
        
        cached_embedding = self.get_cached_embedding(video_path, frame_number, self.model_type)
        if cached_embedding is not None:
            return cached_embedding

        frame_embedding = self.embedder.get_frame_embedding(frame_path)
        self.save_embedding_to_cache(video_path, frame_number, self.model_type, frame_embedding)
        return frame_embedding

class BLIPProcessor(EmbeddingProcessor):
    """Processor that uses BLIP model for embeddings and frame description."""
    def __init__(
        self, 
        cache_db_path: str,
    ):
        super().__init__(cache_db_path)
        self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip_feature_extractor", 
            model_type="base", 
            is_eval=True, 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using BLIP."""
        text_input = self.txt_processors["eval"](text)
        
        with torch.no_grad():
            sample = {"text_input": [text_input]}
            features = self.blip_model.extract_features(sample, mode="text")
            # Use the projected embeddings and take the first token ([CLS])
            text_features = features.text_embeds_proj[:, 0, :]
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

    def get_frame_description(self, frame_path: str) -> str:
        """Get a description of an image frame using BLIP."""
        try:
            # Load and preprocess the image
            raw_image = Image.open(frame_path).convert('RGB')
            image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
            
            # Generate caption using BLIP
            with torch.no_grad():
                caption = self.blip_model.generate({"image": image})
                if isinstance(caption, list):
                    caption = caption[0]  # Take first caption if multiple are generated
                return caption
                
        except Exception as e:
            print(f"Error generating frame description: {str(e)}")
            return ""

    def get_frame_embedding(self, frame_path: str) -> np.ndarray:
        """Get frame embedding by first getting a description and then embedding it."""
        video_path = str(Path(frame_path).parent.parent)
        frame_number = int(Path(frame_path).stem.split('_')[1])
        
        cached_embedding = self.get_cached_embedding(video_path, frame_number, 'blip')
        if cached_embedding is not None:
            return cached_embedding

        frame_description = self.get_frame_description(frame_path)
        frame_embedding = self.get_text_embedding(frame_description)
        self.save_embedding_to_cache(video_path, frame_number, 'blip', frame_embedding)
        return frame_embedding

def find_continuous_action_segment(
    frame_matches: List[FrameMatch],
    last_end_frame: int = -1,
    fps: float = 60.0
) -> Optional[ActionSegment]:
    """
    Find the first continuous segment of frame matches after the last_end_frame
    that has above-average similarity. Uses frame rate to calculate accurate timestamps.
    """
    if not frame_matches:
        return None
        
    # Filter matches to only include those after the last end frame
    valid_matches = [
        match for match in frame_matches if match.frame_number > last_end_frame]
    if not valid_matches:
        return None
        
    # Calculate average similarity across all matches
    avg_similarity = sum(match.similarity for match in frame_matches) / len(frame_matches)
    
    # Sort matches by frame number
    sorted_matches = sorted(valid_matches, key=lambda x: x.frame_number)
    
    # Find first segment with above-average similarity
    for i in range(len(sorted_matches)):
        current_length = 1
        current_similarity = sorted_matches[i].similarity
        start_idx = i
        
        # Look for continuous frames starting at position i
        for j in range(i + 1, len(sorted_matches)):
            if sorted_matches[j].frame_number == sorted_matches[j-1].frame_number + 1:
                current_length += 1
                current_similarity += sorted_matches[j].similarity
            else:
                break
        
        # If segment's average similarity is above the overall average, use this segment
        segment_avg_similarity = current_similarity / current_length
        if segment_avg_similarity >= avg_similarity:
            return ActionSegment(
                start_frame=sorted_matches[start_idx].frame_number,
                end_frame=sorted_matches[start_idx + current_length - 1].frame_number,
                start_time=sorted_matches[start_idx].frame_number / fps,
                end_time=sorted_matches[start_idx + current_length - 1].frame_number / fps
            )
    
    # If no segment found, use the single frame with highest similarity
    best_match = max(sorted_matches, key=lambda x: x.similarity)
    return ActionSegment(
        start_frame=best_match.frame_number,
        end_frame=best_match.frame_number,
        start_time=best_match.frame_number / fps,
        end_time=best_match.frame_number / fps
    )

def process_frames_for_comparison(
    frames: List[Tuple[int, str]],
    frame_embeddings: List[np.ndarray],
    user_descs: List[str],
    desc_embeddings: List[np.ndarray],
    fps: float = 60.0
) -> ProcessingResults:
    """
    Process frames and find action segments for chronologically ordered descriptions.
    Returns segments where similarity is above average, ensuring chronological order.
    
    Args:
        frames: List of (frame_number, frame_path) tuples
        frame_embeddings: List of frame embeddings
        user_descs: List of action descriptions
        desc_embeddings: List of description embeddings
        fps: Frame rate of the video (default: 60.0)
    """
    
    results: Dict[str, ActionResult] = {}
    last_end_frame = -1  # Track the last end frame to ensure chronological order
    
    # Calculate similarities for all frames and descriptions
    all_matches: List[List[FrameMatch]] = []
    for desc_idx, (desc, desc_embedding) in enumerate(zip(user_descs, desc_embeddings)):
        frame_matches = []
        # Calculate similarities for all frames
        similarities = []
        for (frame_num, frame_path), frame_embedding in zip(frames, frame_embeddings):
            similarity = cosine_similarity(desc_embedding, frame_embedding)
            similarities.append(similarity)
            frame_matches.append(FrameMatch(
                frame_number=frame_num,
                frame_path=frame_path,
                similarity=float(similarity)
            ))
        
        # Calculate average similarity for this description
        avg_similarity = np.mean(similarities)
        
        # Filter matches above average similarity
        valid_matches = [
            match for match in frame_matches 
            if match.similarity > avg_similarity
        ]
        
        all_matches.append(sorted(valid_matches, key=lambda x: x.similarity, reverse=True))
    
    # Process each description in order
    for desc_idx, (desc, matches) in enumerate(zip(user_descs, all_matches)):
        # Find continuous segment for this description, ensuring it starts after last_end_frame
        segment = find_continuous_action_segment(matches, last_end_frame, fps)
        
        if segment:
            last_end_frame = segment.end_frame
            results[desc] = ActionResult(
                action_segment=segment,
                top_frames=sorted(
                    [m for m in matches if segment.start_frame <= m.frame_number <= segment.end_frame],
                    key=lambda x: x.similarity,
                    reverse=True
                )
            )
        else:
            print(f"Warning: No valid segment found for description: {desc}")
    
    return ProcessingResults(results=results)

def save_and_report_results(
    results: ProcessingResults,
    result_dir: Path
) -> Path:
    """Save results to a single JSON file."""
    report_dir = result_dir / "report"
    report_dir.mkdir(exist_ok=True)
    
    # Create a detailed results dictionary
    details = {
        "timestamp": datetime.now().isoformat(),
        "actions": []
    }
    
    for desc, action_result in results.results.items():
        action_details = {
            "description": desc,
            "segment": {
                "start_frame": action_result.action_segment.start_frame,
                "end_frame": action_result.action_segment.end_frame,
                "start_time": action_result.action_segment.start_time,
                "end_time": action_result.action_segment.end_time,
                "confidence": sum(match.similarity for match in action_result.top_frames[:10]) / 10
            },
            "frame_matches": [
                {
                    "frame_number": match.frame_number,
                    "similarity": match.similarity,
                    "timestamp": match.frame_number / 30.0  # Assuming 30 fps
                }
                for match in action_result.top_frames
            ]
        }
        details["actions"].append(action_details)
    
    # Save all results to a single JSON file
    with open(report_dir / "analysis_details.json", "w") as f:
        json.dump(details, f, indent=2)
    
    results.report_dir = str(report_dir)
    return report_dir
