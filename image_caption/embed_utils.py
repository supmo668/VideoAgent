# openai_utils.py
import numpy as np
from openai import OpenAI

# Initialize OpenAI client once
client = OpenAI()

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

def get_frame_description(
    system_prompt: str, vision_prompt_template: str, image_path: str) -> str:
    """
    Get a structured description of a video frame using GPT-4 Vision and parse it into an ImageActionFrame.
    """
    parser = PydanticOutputParser(pydantic_object=ImageActionFrame)
    format_instructions = parser.get_format_instructions()
    
    user_prompt = vision_prompt_template.format(
        image_path=image_path, format_instruction=format_instructions)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    
    try:
        frame_info = parser.parse(response.choices[0].message.content.strip())
        return frame_info.action_description
    except Exception as e:
        print(f"Warning: Failed to parse response into ImageActionFrame: {e}")
        return response.choices[0].message.content.strip()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = a / np.linalg.norm(a)
    norm_b = b / np.linalg.norm(b)
    return float(np.dot(norm_a, norm_b))
