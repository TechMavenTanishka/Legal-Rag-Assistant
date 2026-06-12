from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
import chromadb


# ==========================================
# Load Embedding Model
# ==========================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# Connect ChromaDB
# ==========================================

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

client = chromadb.PersistentClient(
    path=str(ROOT / "data" / "chroma_db")
)

collection = client.get_collection(
    "legal_cases"
)


# ==========================================
# Load LLM
# ==========================================

llm = OllamaLLM(
    model="llama3.2"
)


# ==========================================
# Retrieve Context
# ==========================================

def get_context(query):

    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    return results  


# ==========================================
# Main RAG Function
# ==========================================

def ask_question(question):

    results = get_context(question)

    context = "\n\n".join(
        results["documents"][0]
    )

    prompt = f"""
Answer ONLY from the legal context below.

If the answer is not present in the context,
say:
"I could not find enough information in the legal database."

Context:
{context}

Question:
{question}
"""

    answer = llm.invoke(prompt)

    return {
        "answer": answer,
        "sources": results["documents"][0][:3],
        "num_sources": len(results["documents"][0][:3])
    }