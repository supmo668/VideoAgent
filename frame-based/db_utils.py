# db_utils.py
import sqlite3
import numpy as np

def init_cache_db(db_path: str):
    """Initialize the SQLite database for caching embeddings."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS embeddings_cache (
        video_path TEXT,
        frame_number INTEGER,
        question TEXT,
        model_type TEXT,
        embedding BLOB,
        PRIMARY KEY (video_path, frame_number, question, model_type)
    )
    """)
    conn.close()

def get_cached_embedding(db_path: str, video_path: str, frame_number: int, question: str, model_type: str):
    """Retrieve cached embedding if exists."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT embedding FROM embeddings_cache 
    WHERE video_path = ? AND frame_number = ? AND question = ? AND model_type = ?
    """, (video_path, frame_number, question, model_type))
    row = cursor.fetchone()
    conn.close()
    if row is not None:
        # Convert BLOB back to np.array
        return np.frombuffer(row[0], dtype=np.float32)
    return None

def save_embedding_to_cache(db_path: str, video_path: str, frame_number: int, question: str, model_type: str, embedding: np.ndarray):
    """Save the given embedding to the cache."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
    INSERT OR REPLACE INTO embeddings_cache(video_path, frame_number, question, model_type, embedding)
    VALUES (?, ?, ?, ?, ?)
    """, (video_path, frame_number, question, model_type, embedding.tobytes()))
    conn.commit()
    conn.close()
