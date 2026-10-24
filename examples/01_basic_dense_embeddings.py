"""
Pattern 01: Dense Vector Embeddings & Vector Similarity
Demonstrates the fundamental mathematics and mechanics of dense vector embeddings:
1. Converting text documents into high-dimensional vector representations using Cohere embed-v4.0.
2. Understanding search_document vs search_query input types.
3. Computing vector similarity scores using NumPy dot products.
4. Ranking and retrieving the most semantically relevant document.
"""

import os
import sys
import numpy as np
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

# 1. Knowledge Base: Software Testing FAQs
documents = [
    "Selenium WebDriver automates browser interactions.",
    "Pytest is used to run and organize Python tests.",
    "Page Object Model separates page locators from test logic.",
    "Flaky tests pass sometimes and fail intermittently.",
    "CI pipelines automatically execute tests after code changes."
]

print("=" * 70)
print("Knowledge Base Documents:")
for i, doc in enumerate(documents, 1):
    print(f"  {i}. {doc}")
print("=" * 70)

# 2. Convert documents into vectors (input_type="search_document")
print("\n[Step 1] Generating dense embeddings for documents with 'embed-v4.0'...")
doc_response = co.embed(
    model="embed-v4.0",
    texts=documents,
    input_type="search_document",
    embedding_types=["float"]
)

doc_vectors = np.array(doc_response.embeddings.float)
print(f"Document vectors shape: {doc_vectors.shape} (Count: {doc_vectors.shape[0]}, Dimension: {doc_vectors.shape[1]})")

# 3. User's search query (Notice: words 'browser' and 'testing' appear, but 'automate' vs 'automates')
query = "How do I automate browser testing?"
print(f"\n[Step 2] User Search Query: '{query}'")

# 4. Convert query into a vector (input_type="search_query")
query_response = co.embed(
    model="embed-v4.0",
    texts=[query],
    input_type="search_query",
    embedding_types=["float"]
)

query_vector = np.array(query_response.embeddings.float[0])
print(f"Query vector shape: {query_vector.shape}")

# 5. Compute similarity using dot product
# (Cohere embeddings are normalized; dot product equals cosine similarity)
scores = doc_vectors @ query_vector

print("\n[Step 3] Similarity Scores:")
for i, (doc, score) in enumerate(zip(documents, scores), 1):
    print(f"  Doc {i}: {score:.4f} | \"{doc}\"")

# 6. Retrieve best match
best_index = int(np.argmax(scores))

print("\n" + "=" * 70)
print(f"Query: {query}")
print(f"Best match: {documents[best_index]}")
print(f"Similarity score: {scores[best_index]:.4f}")
print("=" * 70)
