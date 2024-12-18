import pickle
from typing import List

import numpy as np

# Load cached CLIP embeddings from a pickle file
cache_clip = pickle.load(open("models/cache_clip.pkl", "rb"))


def get_embeddings(inputs: List[str]) -> np.ndarray:
    """
    Retrieve embeddings for a list of input descriptions from the cached CLIP embeddings.

    Args:
        inputs (List[str]): A list of input descriptions.

    Returns:
        np.ndarray: An array of embeddings corresponding to the input descriptions.
    """
    embeddings = [cache_clip[input] for input in inputs]
    return np.array(embeddings)


def frame_retrieval_seg_ego(descriptions, video_id, sample_idx):
    """
    Perform key frame identification by aligning text descriptions with video segments.
    
    Uses CLIP to find frames that best match the semantic content
    of provided descriptions. This enables semantic-visual alignment
    between textual descriptions and video frames.
    Args:
        descriptions (list): A list of dictionaries containing text descriptions and segment IDs.
        video_id (str): Identifier for the video to process.
        sample_idx (list): List of indices marking the start of each video segment.

    Returns:
        list: Indices of key frames identified for each description.
    """
    # Load frame embeddings for the specified video
    frame_embeddings = np.load(f"ego_features_448/{video_id}.npy")
    
    # Retrieve text embeddings for the descriptions
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions]
    )
    
    frame_idx = []
    for idx, description in enumerate(descriptions):
        # Determine the segment index for the current description
        seg = int(description["segment_id"]) - 1
        
        # Extract embeddings for frames within the specified segment
        seg_frame_embeddings = frame_embeddings[sample_idx[seg] : sample_idx[seg + 1]]
        
        # Handle cases where the segment is too short
        if seg_frame_embeddings.shape[0] < 2:
            frame_idx.append(sample_idx[seg] + 1)
            continue
        
        # Compute similarity between text and frame embeddings
        seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
        
        # Identify the frame with the highest similarity score
        seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
        frame_idx.append(seg_frame_idx)

    return frame_idx


if __name__ == "__main__":
    pass
