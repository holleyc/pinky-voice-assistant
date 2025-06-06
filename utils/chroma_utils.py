# utils/chroma_utils.py

import time
import shutil
import chromadb
from typing import List, Dict, Optional
from uuid import uuid4
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
CHROMA_DB_PATH = "./chroma_db"
CHAT_COLLECTION_NAME = "Pinkys_Brain"
LEXICAL_COLLECTION_NAME = "lexical_facts"       # Single collection for all users' facts
GLOBAL_LEX_COLLECTION_NAME = "global_lexical"   # Single collection for global facts

# === CHROMADB CLIENT SETUP ===
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# The “chat” collection stores messages and user_name entries.
chat_collection = client.get_or_create_collection(
    name=CHAT_COLLECTION_NAME,
    # (Optionally) you can attach an embedding function for chat if you want semantic search:
    # embedding_function=embedding_fn  
)

# The “lexical_facts” collection stores every user’s extracted facts.
lexical_collection = client.get_or_create_collection(
    name=LEXICAL_COLLECTION_NAME,
    embedding_function=None  # We'll supply embeddings manually below
)

# The “global_lexical” collection stores world‐wide facts.
global_lex_collection = client.get_or_create_collection(
    name=GLOBAL_LEX_COLLECTION_NAME,
    embedding_function=None
)

# Shared sentence‐transformer embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# === UTILS ===

def generate_uuid() -> str:
    """Return a new random UUID string."""
    return str(uuid4())


def get_or_create_user_id(request) -> str:
    """
    Attempt to fetch user_id from:
      1) JSON body: request.json['user_id']
      2) Query string: request.args['user_id']
      3) Cookie: request.cookies['user_id']
      4) Fallback: generate a new UUID
    """
    user_id = None
    if request.is_json:
        user_id = request.json.get("user_id")
    if not user_id:
        user_id = request.args.get("user_id")
    if not user_id:
        user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = generate_uuid()
        print(f"[get_or_create_user_id] Generated new user_id: {user_id}")
    else:
        print(f"[get_or_create_user_id] Using existing user_id: {user_id}")
    return user_id


# === USER NAME ===

def save_user_name(user_id: str, name: str) -> None:
    """
    Persist a user's name into the chat_collection as:
      document   = the name string
      metadata   = { "type": "user_name", "user_id": <user_id> }
      embeddings = the embedding of `name`
      id         = "user_name_<user_id>"
    """
    try:
        embedding = embedder.encode(name).tolist()
        doc_id = f"user_name_{user_id}"
        chat_collection.add(
            documents=[name],
            metadatas=[{"type": "user_name", "user_id": user_id}],
            ids=[doc_id],
            embeddings=[embedding]
        )
    except Exception as e:
        print(f"[chroma_utils.save_user_name] Error saving user name: {e}")


def get_saved_user_name(user_id: str) -> Optional[str]:
    """
    Return the saved “user_name” document string if it exists,
    otherwise None.
    """
    try:
        results = chat_collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if meta.get("type") == "user_name":
                return doc
    except Exception as e:
        print(f"[chroma_utils.get_saved_user_name] Error retrieving user name: {e}")
    return None


# === CHAT MESSAGES ===

def save_message_to_chroma(user_id: str, role: str, content: str) -> None:
    """
    Save one chat message to the shared chat_collection:
      - role: "user" or "assistant"
      - content: the textual content
      - metadata: includes type="chat_message", user_id, timestamp, role
      - embeddings: embed(content)
      - id: "{role}_{timestamp}_{user_id}"
    """
    timestamp = int(time.time() * 1000)
    doc_id = f"{role}_{timestamp}_{user_id}"
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
    except Exception as e:
        print(f"[chroma_utils.save_message_to_chroma] Error saving message: {e}")


def get_relevant_context(user_id: str, query: str, n: int = 5) -> List[Dict]:
    """
    Run a semantic query over the chat_collection for documents
    belonging to user_id. Returns up to n results as a list of
    { "role": <role>, "content": <doc> }.
    """
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
        return [{"role": m.get("role", "unknown"), "content": d}
                for d, m in zip(docs, metas)]
    except Exception as e:
        print(f"[chroma_utils.get_relevant_context] Error retrieving context: {e}")
        return []


def get_chat_history(user_id: str) -> List[Dict]:
    """
    Retrieve the full chat history for user_id, sorted by timestamp.
    Returns a list of { "role": <role>, "content": <doc>, "timestamp": <ts> }.
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
        entries.sort(key=lambda x: x["timestamp"])
        return entries
    except Exception as e:
        print(f"[chroma_utils.get_chat_history] Error retrieving chat history: {e}")
        return []


# === LEXICAL FACTS ===

def save_lexical_facts(user_id: str, facts: List[str]) -> None:
    """
    Save a list of simple “facts” for a given user. Each fact is:
      - document: the fact string
      - metadata: { "type": "lexical_fact", "user_id": user_id }
      - id: random UUID
      - embedding: embed(fact)
    Uses a single “lexical_facts” collection for all users.
    """
    for fact in facts:
        try:
            embedding = embedder.encode(fact).tolist()
            lexical_collection.add(
                documents=[fact],
                embeddings=[embedding],
                metadatas=[{"type": "lexical_fact", "user_id": user_id}],
                ids=[str(uuid4())]
            )
        except Exception as e:
            print(f"[chroma_utils.save_lexical_facts] Error saving lexical fact: {e}")


def get_lexical_context(user_id: str, query_text: str, n_results: int = 5) -> List[str]:
    """
    Query the “lexical_facts” collection for facts semantically similar
    to query_text, filtered by user_id. Returns a list of up to n_results
    matching fact strings.
    """
    try:
        query_emb = embedder.encode(query_text).tolist()
        results = lexical_collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            where={"user_id": user_id},
            include=["documents"]
        )
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        print(f"[chroma_utils.get_lexical_context] Error querying lexical facts: {e}")
        return []


# === GLOBAL FACTS ===

def save_global_fact(fact: str) -> None:
    """
    Save a “global_lexical” fact. Used for world‐wide or app‐wide knowledge.
    No user_id is attached.
    """
    try:
        embedding = embedder.encode(fact).tolist()
        global_lex_collection.add(
            documents=[fact],
            embeddings=[embedding],
            metadatas=[{"type": "global_fact"}],
            ids=[str(uuid4())]
        )
    except Exception as e:
        print(f"[chroma_utils.save_global_fact] Error saving global fact: {e}")


def get_global_context(query_text: str, n_results: int = 5) -> List[str]:
    """
    Semantic search over all global facts. Returns up to n_results fact strings.
    """
    try:
        query_emb = embedder.encode(query_text).tolist()
        results = global_lex_collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            include=["documents"]
        )
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        print(f"[chroma_utils.get_global_context] Error retrieving global context: {e}")
        return []
