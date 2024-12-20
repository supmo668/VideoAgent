# clip_utils.py
import torch
import numpy as np
from PIL import Image
import clip

# Load CLIP model globally (adjust model name/device as needed)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def get_text_embedding_clip(text: str) -> np.ndarray:
    text_token = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_token).cpu().numpy()[0]
    return text_embedding.astype(np.float32)

def get_frame_embedding_clip(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = clip_model.encode_image(image_input).cpu().numpy()[0]
    return image_embedding.astype(np.float32)
