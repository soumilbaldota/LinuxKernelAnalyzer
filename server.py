from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from langchain_community.vectorstores import Chroma
from pprint import pprint
import textwrap
import requests
import json

# -----------------------------
# Chroma / RAG setup
# -----------------------------
DATA_DIR = "/home/ubuntu/environment/chroma_rag"
COLLECTION = "kernel_chunks"  # must match your persisted collection
vs = Chroma(
    persist_directory=DATA_DIR,
)

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Kernel RAG + Qwen API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or list your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    kernel_log: str  # add this to hold the log text
    max_results: Optional[int] = 5  # optional, can default


# -----------------------------
# Helper: build prompt for Qwen
# -----------------------------
SYSTEM_PROMPT = """You are a Linux kernel analysis assistant.
Use the retrieved RAG code/context to explain kernel log behavior.
Cite file:line when possible, and be concise."""

def build_prompt(system_prompt: str, rag_results: List[Dict], kernel_log: str, user_query: str, max_chunks=1):
    """
    Build prompt with size limits to avoid 'entity too large' errors.
    Only uses the top result and truncates content aggressively.
    """
    # Sort by score ascending (lower is better for distance) and take only top result
    rag_results = sorted(rag_results, key=lambda x: x.get("score", 0))[:max_chunks]
    
    ctx = []
    for r in rag_results:
        src = r["metadata"].get("source", "?")
        line = r["metadata"].get("start_line", "?")
        # Aggressive truncation - max 400 chars per snippet
        snippet = textwrap.shorten(r["content"].strip(), width=400, placeholder="...")
        ctx.append(f"Source: {src}:{line}\n{snippet}")
    
    ctx_block = "\n\n---\n\n".join(ctx)
    
    # Truncate kernel log to first 500 chars to reduce payload size
    kernel_log_truncated = kernel_log.strip()[:500]
    if len(kernel_log.strip()) > 500:
        kernel_log_truncated += "\n... [truncated for brevity]"
    
    return (
        f"System:\n{system_prompt.strip()}\n\n"
        f"Context (retrieved kernel source snippet):\n{ctx_block}\n\n"
        f"Kernel Log:\n{kernel_log_truncated}\n\n"
        f"User Question:\n{user_query.strip()}\n\n"
        f"Assistant:\n"
    )

# -----------------------------
# Helper: call TGI on localhost:8081
# -----------------------------
TGI_URL = "http://localhost:8081/"

def query_tgi(prompt: str, max_new_tokens=512, temperature=0.1):
    """
    Call the TGI inference endpoint with error handling for large payloads.
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }
    
    try:
        resp = requests.post(TGI_URL, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        return result[0]["generated_text"] if len(result) > 0 else ""
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 413:
            raise Exception("Payload too large for model - try reducing context")
        raise
    except requests.exceptions.Timeout:
        raise Exception("TGI request timed out")
    except Exception as e:
        raise Exception(f"TGI error: {str(e)}")

# -----------------------------
# Endpoint
# -----------------------------
@app.post("/")
def rag_query(request: QueryRequest):
    # 1️⃣ RAG search - Use query to find relevant code
    print("\n" + "="*80)
    print("REQUEST RECEIVED:")
    print("="*80)
    pprint(request.dict())
    print("Vector count:", vs._collection.count())
    
    # Search using the query to find relevant code
    rag_result_raw = vs.similarity_search_with_score(request.query, k=request.max_results)

    print("\n" + "="*80)
    print("RAG RESULTS:")
    print("="*80)
    pprint(rag_result_raw)
    
    rag_results = [
        {"content": doc.page_content, "metadata": doc.metadata, "score": score}
        for doc, score in rag_result_raw
    ]

    # 2️⃣ Build prompt - ONLY use top 3 result to avoid entity too large
    if rag_results:
        # Use only the best result for the prompt to keep size down
        prompt = build_prompt(SYSTEM_PROMPT, rag_results, request.kernel_log, request.query, max_chunks=3)
    else:
        # No RAG results: still include truncated kernel_log
        kernel_log_truncated = request.kernel_log.strip()[:500]
        if len(request.kernel_log.strip()) > 500:
            kernel_log_truncated += "\n... [truncated]"
            
        prompt = (
            f"System:\n{SYSTEM_PROMPT.strip()}\n\n"
            f"No relevant context found from RAG.\n\n"
            f"Kernel Log:\n{kernel_log_truncated}\n\n"
            f"User Question:\n{request.query.strip()}\n\n"
            f"Assistant:\n"
        )

    print("\n" + "="*80)
    print("PROMPT SENT TO QWEN (length: {} chars):".format(len(prompt)))
    print("="*80)
    print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
    print("="*80)

    # 3️⃣ Call Qwen with error handling
    try:
        qwen_answer = query_tgi(prompt, max_new_tokens=512)
        print("\n" + "="*80)
        print("QWEN RESPONSE:")
        print("="*80)
        print(qwen_answer)
        print("="*80 + "\n")
    except Exception as e:
        print(f"Error calling Qwen: {e}")
        qwen_answer = f"⚠️ Error generating analysis: {str(e)}"

    # 4️⃣ Return response in format Streamlit expects
    # Return ALL rag_results for display, but only the top one was sent to the model
    return {
        "matches": rag_results,  # Return all matches for UI display
        "analysis": qwen_answer,  # Add the AI analysis (based on top result only)
        "query": request.query,
        "total_results": len(rag_results),
        "note": "Analysis based on top matching result only to avoid payload size limits"
    }   