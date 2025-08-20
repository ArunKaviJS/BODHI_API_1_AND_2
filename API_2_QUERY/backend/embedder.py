import os
import numpy as np
from typing import List, Tuple
from openai import AzureOpenAI
from backend.db import embedding_collection  # Use shared Mongo collection

# === Azure Embedding Client ===
embed_client = AzureOpenAI(
    api_key="ddddddddd",
    api_version="www",
    azure_endpoint="hhhhhhhhh"
)

MIN_SIMILARITY = 0.7  # cutoff for relevance

def embed_text(text: str) -> list:
    """Generate embedding for given text."""
    response = embed_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2) -> float:
    """Compute cosine similarity between two vectors."""
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def get_similar_chunks(vector_search_text: str, machine_id: str, top_k: int = 3) -> List[Tuple[float, str, list]]:
    query_embedding = embed_text(vector_search_text)

    docs = list(embedding_collection.find(
        {"machine_id": machine_id},
        {"embedding": 1, "text": 1, "_id": 0}
    ))

    if not docs:
        return []

    similarities = []
    for doc in docs:
        if "embedding" not in doc or "text" not in doc:
            # Debug log to catch bad docs
            print(f"[Warning] Skipping doc missing fields: {doc}")
            continue
        score = cosine_similarity(query_embedding, doc["embedding"])
        similarities.append((score, doc["text"], doc["embedding"]))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]


