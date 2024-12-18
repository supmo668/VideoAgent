# openai_utils.py
import numpy as np
from openai import OpenAI

# Initialize OpenAI client once
client = OpenAI()

def get_text_embedding_openai(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Generate text embedding using OpenAI embedding model.
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)

def get_frame_description(system_prompt: str, vision_prompt_template: str, image_path: str, description: str) -> str:
    user_prompt = vision_prompt_template.format(description=description, image_path=image_path)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

# embed_utils.py
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = a / np.linalg.norm(a)
    norm_b = b / np.linalg.norm(b)
    return float(np.dot(norm_a, norm_b))
