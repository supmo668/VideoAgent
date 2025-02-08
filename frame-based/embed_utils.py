# openai_utils.py
import os
import numpy as np
from openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from models import ImageActionFrame, BioAllowableActionTypes

def get_openai_client():
    """Get OpenAI client instance, initializing it if necessary."""
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY environment variable is required for OpenAI operations"
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_text_embedding_openai(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Generate text embedding using OpenAI embedding model.
    """
    client = get_openai_client()
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
    client = get_openai_client()
    try:
        import openai
        parser = PydanticOutputParser(pydantic_object=ImageActionFrame)
        vision_description_prompt = vision_prompt.format(format_instruction=parser.get_format_instructions()) or "Describe this image."
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
                            "text": vision_description_prompt
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
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting frame description: {e}")
        return ""

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
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
