# utils/chroma_utils.py

import os
import time
import shutil
import re
import chromadb
import json
import torch
from typing import List, Dict, Optional
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from datetime import datetime

from typing import List
from memory import chroma_client

# ------------------------------------------------------------------------------
#                             Setup Logging
# ------------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "chroma_events.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
except Exception as e:
    print("⚠️ CUDA init failed or not available—falling back to CPU embeddings.")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def log_chroma_event(event_type: str, data: dict):
    """
    Append a JSON‐line to logs/chroma_events.jsonl with fields:
      - timestamp: UTC ISO8601 string
      - event:    name of the event
      - ...plus whatever is in data
    """
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event_type,
        **data
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# === CONFIGURATION ===
CHROMA_DB_PATH            = "./chroma_db"
CHAT_COLLECTION_NAME      = "Pinkys_Brain"
LEXICAL_COLLECTION_NAME   = "lexical_facts"       # Single collection for all users’ facts
GLOBAL_LEX_COLLECTION_NAME = "global_lexical"      # Single collection for global facts

# === CHROMADB CLIENT SETUP ===
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# The “chat” collection stores messages and user_name entries.
try:
    chat_collection = client.get_or_create_collection(
        name=CHAT_COLLECTION_NAME
        # embedding_function=...  # If you later want semantic search, attach embed function
    )
    log_chroma_event("init_collection", {
        "collection": CHAT_COLLECTION_NAME
    })
except Exception as e:
    log_chroma_event("init_collection_error", {
        "collection": CHAT_COLLECTION_NAME,
        "error": str(e)
    })
    raise

# The “lexical_facts” collection stores every user's extracted facts.
try:
    lexical_collection = client.get_or_create_collection(
        name=LEXICAL_COLLECTION_NAME,
        embedding_function=None  # We'll supply embeddings manually
    )
    log_chroma_event("init_collection", {
        "collection": LEXICAL_COLLECTION_NAME
    })
except Exception as e:
    log_chroma_event("init_collection_error", {
        "collection": LEXICAL_COLLECTION_NAME,
        "error": str(e)
    })
    raise

# The “global_lexical” collection stores world‐wide facts.
try:
    global_lex_collection = client.get_or_create_collection(
        name=GLOBAL_LEX_COLLECTION_NAME,
        embedding_function=None
    )
    log_chroma_event("init_collection", {
        "collection": GLOBAL_LEX_COLLECTION_NAME
    })
except Exception as e:
    log_chroma_event("init_collection_error", {
        "collection": GLOBAL_LEX_COLLECTION_NAME,
        "error": str(e)
    })
    raise

# Shared sentence‐transformer embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------------------------------------------------
#                                   UTILS
# ------------------------------------------------------------------------------

def generate_uuid() -> str:
    """Return a new random UUID string."""
    uid = str(uuid4())
    log_chroma_event("generate_uuid", {"uuid": uid})
    return uid


def get_or_create_user_id(request) -> str:
    """
    Fetch user_id from, in order:
      1) JSON body → request.json['user_id']
      2) Query string → request.args['user_id']
      3) Cookie → request.cookies['user_id']
      4) Else → new UUID
    Logs whether existing or generated.
    """
    user_id = None
    if request.is_json:
        user_id = request.json.get("user_id")
    if not user_id:
        user_id = request.args.get("user_id")
    if not user_id:
        user_id = request.cookies.get("user_id")
    if user_id:
        log_chroma_event("get_or_create_user_id_existing", {"user_id": user_id})
    else:
        user_id = generate_uuid()
        log_chroma_event("get_or_create_user_id_generated", {"user_id": user_id})
    return user_id

def extract_text_from_html(html: str) -> str:
    # (If you want to pull this out of the route)
    # … same BeautifulSoup logic …
    pass

def chunk_text(text: str, chunk_size: int=500, overlap: int=50) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def embed_and_store(user_id: str, docs: List[str], source: str):
    embeddings = [embedder.encode(d).tolist() for d in docs]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(docs))]
    chroma_client.add(
        collection_name=user_id,  # or a shared “pinky_web_pages” collection
        embeddings=embeddings,
        documents=docs,
        metadatas=metadatas,
    )

# ------------------------------------------------------------------------------
#                                USER NAME
# ------------------------------------------------------------------------------

def save_user_name(user_id: str, name: str) -> None:
    """
    Persist a user's name into chat_collection:
      - document   = name string
      - metadata   = { "type": "user_name", "user_id": user_id }
      - embeddings = embed(name)
      - id         = "user_name_<user_id>"
    Logs on success or failure.
    """
    doc_id = f"user_name_{user_id}"
    try:
        embedding = embedder.encode(name).tolist()
        chat_collection.add(
            documents=[name],
            metadatas=[{"type": "user_name", "user_id": user_id}],
            ids=[doc_id],
            embeddings=[embedding]
        )
        log_chroma_event("save_user_name", {
            "user_id": user_id,
            "name": name,
            "doc_id": doc_id
        })
    except Exception as e:
        log_chroma_event("save_user_name_error", {
            "user_id": user_id,
            "name": name,
            "error": str(e)
        })


def get_saved_user_name(user_id: str) -> Optional[str]:
    """
    Return the saved “user_name” document string if it exists, else None.
    Logs attempts and outcomes.
    """
    try:
        results = chat_collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if meta.get("type") == "user_name":
                log_chroma_event("get_saved_user_name_found", {
                    "user_id": user_id,
                    "returned_name": doc
                })
                return doc
        log_chroma_event("get_saved_user_name_not_found", {
            "user_id": user_id
        })
    except Exception as e:
        log_chroma_event("get_saved_user_name_error", {
            "user_id": user_id,
            "error": str(e)
        })
    return None


# ------------------------------------------------------------------------------
#                              CHAT MESSAGES
# ------------------------------------------------------------------------------

def save_message_to_chroma(user_id: str, role: str, content: str) -> None:
    """
    Save one chat message to the shared chat_collection:
      - role: "user" or "assistant"
      - content: text
      - metadata: { "type":"chat_message","role":role,"user_id":user_id,"timestamp":ts }
      - embeddings: embed(content)
      - id: "{role}_{timestamp}_{user_id}"
    Logs success (with snippet/length) or failure.
    """
    timestamp = int(time.time() * 1000)
    doc_id = f"{role}_{timestamp}_{user_id}"
    snippet = content[:50]
    try:
        embedding = embedder.encode(content).tolist()
        chat_collection.add(
            documents=[content],
            metadatas=[{
                "type": "chat_message",
                "role": role,
                "user_id": user_id,
                "timestamp": timestamp
            }],
            ids=[doc_id],
            embeddings=[embedding]
        )
        log_chroma_event("save_message_to_chroma", {
            "user_id": user_id,
            "role": role,
            "content_snippet": snippet,
            "content_length": len(content),
            "doc_id": doc_id
        })
    except Exception as e:
        log_chroma_event("save_message_to_chroma_error", {
            "user_id": user_id,
            "role": role,
            "content_snippet": snippet,
            "error": str(e)
        })


def get_relevant_context(user_id: str, query: str, n: int = 5) -> List[Dict]:
    """
    Run a semantic query over chat_collection for messages by this user.
    Returns up to n results as a list of { "role":<role>, "content":<doc> }.
    Logs query inputs and hit count (or error).
    """
    snippet = query[:50]
    try:
        query_emb = embedder.encode(query).tolist()
        results = chat_collection.query(
            query_embeddings=[query_emb],
            n_results=n,
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        hit_count = len(docs)
        log_chroma_event("get_relevant_context", {
            "user_id": user_id,
            "query_snippet": snippet,
            "n": n,
            "hit_count": hit_count
        })
        return [{"role": m.get("role", "unknown"), "content": d} for d, m in zip(docs, metas)]
    except Exception as e:
        log_chroma_event("get_relevant_context_error", {
            "user_id": user_id,
            "query_snippet": snippet,
            "error": str(e)
        })
        return []


def get_chat_history(user_id: str) -> List[Dict]:
    """
    Retrieve the full chat history for user_id, sorted by timestamp.
    Returns a list of { "role":<role>, "content":<doc>, "timestamp":<ts> }.
    Logs number of entries returned or error.
    """
    try:
        results = chat_collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        entries = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if meta.get("type") == "chat_message" and meta.get("role") in ["user", "assistant"]:
                entries.append({
                    "role": meta["role"],
                    "content": doc,
                    "timestamp": meta.get("timestamp", 0)
                })
        # sort by timestamp ascending
        entries.sort(key=lambda x: x["timestamp"])
        log_chroma_event("get_chat_history", {
            "user_id": user_id,
            "entry_count": len(entries)
        })
        return entries
    except Exception as e:
        log_chroma_event("get_chat_history_error", {
            "user_id": user_id,
            "error": str(e)
        })
        return []


# ------------------------------------------------------------------------------
#                              LEXICAL FACTS
# ------------------------------------------------------------------------------

def save_lexical_facts(user_id: str, facts: List[str]) -> None:
    """
    Save a list of simple “facts” for a given user. Each:
      - document: fact string
      - metadata: { "type": "lexical_fact", "user_id": user_id }
      - id: random UUID
      - embedding: embed(fact)
    Uses one “lexical_facts” collection for all users.
    Logs each success or failure.
    """
    for fact in facts:
        snippet = fact[:50]
        doc_id = str(uuid4())
        try:
            embedding = embedder.encode(fact).tolist()
            lexical_collection.add(
                documents=[fact],
                embeddings=[embedding],
                metadatas=[{"type": "lexical_fact", "user_id": user_id}],
                ids=[doc_id]
            )
            log_chroma_event("save_lexical_facts", {
                "user_id": user_id,
                "fact_snippet": snippet,
                "doc_id": doc_id
            })
        except Exception as e:
            log_chroma_event("save_lexical_facts_error", {
                "user_id": user_id,
                "fact_snippet": snippet,
                "error": str(e)
            })


def get_lexical_context(user_id: str, query_text: str, n_results: int = 5) -> List[str]:
    """
    Query “lexical_facts” for semantically similar facts to query_text,
    filtered by user_id. Returns a list of up to n_results fact strings.
    Logs query inputs and result count or error.
    """
    snippet = query_text[:50]
    try:
        query_emb = embedder.encode(query_text).tolist()
        results = lexical_collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            where={"user_id": user_id},
            include=["documents"]
        )
        docs = results["documents"][0] if results["documents"] else []
        log_chroma_event("get_lexical_context", {
            "user_id": user_id,
            "query_snippet": snippet,
            "n_results": n_results,
            "hit_count": len(docs)
        })
        return docs
    except Exception as e:
        log_chroma_event("get_lexical_context_error", {
            "user_id": user_id,
            "query_snippet": snippet,
            "error": str(e)
        })
        return []


# ------------------------------------------------------------------------------
#                             GLOBAL FACTS
# ------------------------------------------------------------------------------

def save_global_fact(fact: str) -> None:
    """
    Save a “global_lexical” fact. No user_id is attached.
    Logs success or failure.
    """
    snippet = fact[:50]
    doc_id = str(uuid4())
    try:
        embedding = embedder.encode(fact).tolist()
        global_lex_collection.add(
            documents=[fact],
            embeddings=[embedding],
            metadatas=[{"type": "global_fact"}],
            ids=[doc_id]
        )
        log_chroma_event("save_global_fact", {
            "fact_snippet": snippet,
            "doc_id": doc_id
        })
    except Exception as e:
        log_chroma_event("save_global_fact_error", {
            "fact_snippet": snippet,
            "error": str(e)
        })


def get_global_context(query_text: str, n_results: int = 5) -> List[str]:
    """
    Semantic search over all global facts. Returns up to n_results fact strings.
    Logs query inputs and result count or error.
    """
    snippet = query_text[:50]
    try:
        query_emb = embedder.encode(query_text).tolist()
        results = global_lex_collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            include=["documents"]
        )
        docs = results["documents"][0] if results["documents"] else []
        log_chroma_event("get_global_context", {
            "query_snippet": snippet,
            "n_results": n_results,
            "hit_count": len(docs)
        })
        return docs
    except Exception as e:
        log_chroma_event("get_global_context_error", {
            "query_snippet": snippet,
            "error": str(e)
        })
        return []
