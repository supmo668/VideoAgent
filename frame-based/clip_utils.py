# clip_utils.py
import torch
import numpy as np
from PIL import Image
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load models globally (adjust model names/device as needed)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load BLIP model
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def get_text_embedding_clip(text: str) -> np.ndarray:
    """Get CLIP text embedding."""
    text_token = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_token).cpu().numpy()[0]
    return text_embedding.astype(np.float32)

def get_frame_embedding_clip(image_path: str) -> np.ndarray:
    """Get CLIP image embedding."""
    image = Image.open(image_path).convert("RGB")
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = clip_model.encode_image(image_input).cpu().numpy()[0]
    return image_embedding.astype(np.float32)

def get_text_embedding_blip(text: str) -> np.ndarray:
    """Get BLIP text embedding."""
    inputs = blip_processor(text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = blip_model.get_text_features(**inputs)
        text_embedding = text_features.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    return text_embedding.astype(np.float32)

def get_frame_embedding_blip(image_path: str) -> np.ndarray:
    """Get BLIP image embedding."""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = blip_model.get_image_features(**inputs)
        image_embedding = image_features.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    return image_embedding.astype(np.float32)

def get_frame_embedding_combined(image_path: str) -> np.ndarray:
    """Get combined CLIP and BLIP image embeddings."""
    clip_embedding = get_frame_embedding_clip(image_path)
    blip_embedding = get_frame_embedding_blip(image_path)
    # Normalize each embedding
    clip_embedding = clip_embedding / np.linalg.norm(clip_embedding)
    blip_embedding = blip_embedding / np.linalg.norm(blip_embedding)
    # Concatenate and normalize again
    combined = np.concatenate([clip_embedding, blip_embedding])
    combined = combined / np.linalg.norm(combined)
    return combined.astype(np.float32)

def get_text_embedding_combined(text: str) -> np.ndarray:
    """Get combined CLIP and BLIP text embeddings."""
    clip_embedding = get_text_embedding_clip(text)
    blip_embedding = get_text_embedding_blip(text)
    # Normalize each embedding
    clip_embedding = clip_embedding / np.linalg.norm(clip_embedding)
    blip_embedding = blip_embedding / np.linalg.norm(blip_embedding)
    # Concatenate and normalize again
    combined = np.concatenate([clip_embedding, blip_embedding])
    combined = combined / np.linalg.norm(combined)
    return combined.astype(np.float32)
