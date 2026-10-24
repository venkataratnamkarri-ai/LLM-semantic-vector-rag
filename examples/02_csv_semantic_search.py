"""
Pattern 02: Semantic Vector Search over CSV Dataset
Demonstrates:
1. Loading agile user stories from CSV with Pandas.
2. Generating semantic vector embeddings across all records.
3. Performing real-time semantic search using scikit-learn cosine similarity.
4. Retrieving conceptually related user stories even with zero keyword overlap.
"""

import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import cohere
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found. Please set it in your .env file.")

co = cohere.ClientV2(api_key=api_key)
MODEL = "embed-english-v3.0"

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "db_userstories.csv"

if not CSV_PATH.is_file():
    raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

# 1. Load User Stories
df = pd.read_csv(CSV_PATH)
if "story" not in df.columns:
    raise ValueError("CSV must contain a 'story' column")

stories = df["story"].dropna().astype(str).tolist()
print(f"Loaded {len(stories)} user stories from {CSV_PATH.name}")

# 2. Embed All User Stories
print(f"\nGenerating vector embeddings with model '{MODEL}'...")
response = co.embed(
    model=MODEL,
    texts=stories,
    input_type="search_document",
    embedding_types=["float"]
)

story_embeddings = np.array(response.embeddings.float)
print(f"Indexed matrix shape: {story_embeddings.shape}")

# 3. Semantic Search Function
def semantic_search(query: str, top_k: int = 4) -> None:
    query_response = co.embed(
        model=MODEL,
        texts=[query],
        input_type="search_query",
        embedding_types=["float"]
    )
    query_embed = np.array(query_response.embeddings.float[0])

    # Compute cosine similarities between query and all stored story vectors
    sims = cosine_similarity([query_embed], story_embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    print("\n" + "=" * 70)
    print(f"Search Query: \"{query}\"")
    print("=" * 70)
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. [Score: {sims[idx]:.4f}] {stories[idx]}")


if __name__ == "__main__":
    # Test queries showing conceptual matching
    semantic_search("As a user I want to save items to buy later")
    semantic_search("Canceling order due to delayed parcel delivery")
    semantic_search("Changing my account password or login credentials")
