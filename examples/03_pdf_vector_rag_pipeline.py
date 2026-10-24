"""
Pattern 03: End-to-End PDF Vector RAG Pipeline
Demonstrates:
1. Extracting text from multi-page PDF documents using PyPDF2.
2. Sliding-window chunking (chunk_size=800, overlap=100) to preserve cross-boundary semantics.
3. Batch vector embedding generation with Cohere embed-v4.0.
4. Top-K semantic retrieval via vector dot-product similarity.
5. Grounded response generation using Cohere Command-A.
"""

import os
from pathlib import Path
import sys
import numpy as np
import PyPDF2
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

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "Agile_Process_Guide_5_Pages.pdf"

if not PDF_PATH.is_file():
    raise FileNotFoundError(f"PDF document not found at: {PDF_PATH}")

# 1. Read PDF Text
def extract_pdf_text(path: Path) -> str:
    pdf_text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"
    return pdf_text

# 2. Text Chunker with Overlap
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

# 3. Vector RAG Pipeline Execution
def run_vector_rag(query: str, chunks: list[str], chunk_vectors: np.ndarray, top_k: int = 3) -> tuple[str, list[str]]:
    # Embed the query
    query_response = co.embed(
        model="embed-v4.0",
        texts=[query],
        input_type="search_query",
        embedding_types=["float"]
    )
    query_vector = np.array(query_response.embeddings.float[0])

    # Compute dot product similarity scores
    scores = chunk_vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved_chunks = [chunks[int(idx)] for idx in top_indices]
    context = "\n\n--- CHUNK ---\n\n".join(retrieved_chunks)

    prompt = f"""
You are an Agile Process assistant. Answer the user question strictly using ONLY the context provided below.
If the answer cannot be determined from the context, state that the information was not found in the document.

Context:
{context}

Question:
{query}

Answer:
"""
    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=250
    )

    answer = ""
    for item in response.message.content:
        if hasattr(item, "text"):
            answer += item.text

    return answer.strip(), retrieved_chunks


if __name__ == "__main__":
    print(f"Reading PDF from: {PDF_PATH}")
    pdf_text = extract_pdf_text(PDF_PATH)
    chunks = chunk_text(pdf_text, chunk_size=800, overlap=100)
    print(f"Created {len(chunks)} overlapping chunks.")

    print("\nGenerating document vector index with 'embed-v4.0'...")
    doc_response = co.embed(
        model="embed-v4.0",
        texts=chunks,
        input_type="search_document",
        embedding_types=["float"]
    )
    chunk_vectors = np.array(doc_response.embeddings.float)
    print(f"Vector matrix shape: {chunk_vectors.shape}")

    query = "What are the key responsibilities of a Scrum Master?"
    print("\n" + "=" * 70)
    print(f"User Query: {query}")
    print("=" * 70)

    answer, retrieved = run_vector_rag(query, chunks, chunk_vectors, top_k=3)

    print(f"\n[Retrieved {len(retrieved)} Context Chunks via Vector Search]:")
    for idx, chunk in enumerate(retrieved, 1):
        preview = chunk[:180].replace("\n", " ")
        print(f"  {idx}. {preview}...")

    print("\n[Grounded Model Answer]:")
    print(answer)
