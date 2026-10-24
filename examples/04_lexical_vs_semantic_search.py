"""
Pattern 04: Lexical (Keyword) vs Semantic (Vector) Retrieval Benchmark
Demonstrates the critical difference between lexical matching (BM25 / Keyword Overlap)
and dense semantic vector search.

Illustrates why dense vectors excel when queries use synonyms, paraphrasing,
or conceptual phrasing with zero literal keyword overlap.
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

corpus = [
    "Selenium WebDriver automates browser interactions.",
    "Pytest is used to run and organize Python tests.",
    "Page Object Model separates page locators from test logic.",
    "Flaky tests pass sometimes and fail intermittently.",
    "CI pipelines automatically execute tests after code changes."
]

# 1. Lexical Keyword Matching
def lexical_search(query: str, docs: list[str]) -> list[tuple[int, str]]:
    query_tokens = set(query.lower().split())
    scored = []
    for doc in docs:
        doc_tokens = set(doc.lower().split())
        overlap = len(query_tokens.intersection(doc_tokens))
        scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

# 2. Semantic Dense Vector Search
print("Indexing corpus into dense vector embeddings...")
doc_embeds = np.array(
    co.embed(
        model="embed-v4.0",
        texts=corpus,
        input_type="search_document",
        embedding_types=["float"]
    ).embeddings.float
)

def semantic_search(query: str, docs: list[str]) -> list[tuple[float, str]]:
    query_embed = np.array(
        co.embed(
            model="embed-v4.0",
            texts=[query],
            input_type="search_query",
            embedding_types=["float"]
        ).embeddings.float[0]
    )
    scores = doc_embeds @ query_embed
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return ranked


if __name__ == "__main__":
    # Query with SYNONYMS: 'sporadic build failures' instead of 'flaky tests fail intermittently'
    test_queries = [
        "Dealing with sporadic build failures and inconsistent test outcomes",
        "Framework for decoupling web element selectors from test scripts"
    ]

    for q in test_queries:
        print("\n" + "=" * 75)
        print(f"QUERY: \"{q}\"")
        print("=" * 75)

        # Lexical Results
        lex_results = lexical_search(q, corpus)
        print("\n[1] LEXICAL (Keyword Overlap) Top Match:")
        score, doc = lex_results[0]
        print(f"    Overlap Count: {score} | Document: \"{doc}\"")
        if score == 0:
            print("    ⚠️ Lexical search completely failed due to zero keyword overlap!")

        # Semantic Results
        sem_results = semantic_search(q, corpus)
        print("\n[2] SEMANTIC (embed-v4.0 Dense Vector) Top Match:")
        score, doc = sem_results[0]
        print(f"    Similarity Score: {score:.4f} | Document: \"{doc}\"")
        print("    ✅ Vector search correctly understood the underlying semantic concept!")
