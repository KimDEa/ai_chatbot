import os
import numpy as np
import ollama

MODEL = "bge-m3"
BATCH = 100

def embed(texts):
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        result = ollama.embed(model=MODEL, input=batch)
        vectors.extend(result["embeddings"])
        if len(texts) > BATCH:
            print(f"임베딩 {len(vectors)}/{len(texts)}")
    vectors = np.array(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)