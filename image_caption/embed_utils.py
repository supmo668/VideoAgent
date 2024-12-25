# openai_utils.py
import os

import numpy as np
from openai import OpenAI

# Initialize OpenAI client once
assert os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from langchain.output_parsers import PydanticOutputParser
from models import ImageActionFrame, BioAllowableActionTypes

def get_text_embedding_openai(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Generate text embedding using OpenAI embedding model.
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)

def get_frame_description(system_prompt: str, vision_prompt: str, frame_path: str) -> str:
    """
    Get a description of an image frame using GPT-4 Vision API.
    
    Args:
        system_prompt: System prompt for GPT-4 Vision
        vision_prompt: Vision prompt for GPT-4 Vision
        frame_path: Path to the image frame
        
    Returns:
        str: Description of the image frame
    """
    try:
        import openai
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the preview model as it's the current version
            messages=[
                {
                    "role": "system",
                    "content": system_prompt or "You are an AI that generates descriptions for images."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": vision_prompt or "Describe this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(frame_path)}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error getting frame description: {str(e)}")
        return f"Error processing frame: {str(e)}"

def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    import base64
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = a / np.linalg.norm(a)
    norm_b = b / np.linalg.norm(b)
    return float(np.dot(norm_a, norm_b))
