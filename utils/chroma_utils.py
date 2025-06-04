import uuid
import time
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

# utils/chroma_utils.py

from uuid import uuid4
from chromadb import Client

# Ensure this is created once and reused
chroma_client = Client()

def save_lexical_facts(user_id, facts):
    collection = chroma_client.get_or_create_collection(f"{user_id}_lexical")
    for fact in facts:
        collection.add(documents=[fact], ids=[str(uuid4())])

def get_lexical_context(user_id, query_text, n_results=5):
    try:
        collection = chroma_client.get_collection(f"{user_id}_lexical")
        results = collection.query(query_texts=[query_text], n_results=n_results)
        return results["documents"][0] if results["documents"] else []
    except ValueError:
        # Collection might not exist yet
        return []



# === CONFIGURATION ===
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "Pinkys_Brain"

# === INIT ===
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === UTILS ===

def generate_uuid() -> str:
    return str(uuid.uuid4())

def get_or_create_user_id(request) -> str:
    return request.cookies.get("user_id") or request.args.get("user_id") or generate_uuid()

# === USER NAME ===

def save_user_name(user_id: str, name: str):
    embedding = embedder.encode(name).tolist()
    doc_id = f"user_name_{user_id}"
    try:
        collection.add(
            documents=[name],
            metadatas=[{"type": "user_name", "user_id": user_id}],
            ids=[doc_id],
            embeddings=[embedding]
        )
    except Exception as e:
        print(f"[chroma_utils] Error saving user name: {e}")

def get_saved_user_name(user_id: str) -> Optional[str]:
    try:
        results = collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if meta.get("type") == "user_name":
                return doc
    except Exception as e:
        print(f"[chroma_utils] Error retrieving user name: {e}")
    return None

# === MESSAGES ===

def save_message_to_chroma(user_id: str, role: str, content: str):
    timestamp = int(time.time() * 1000)
    doc_id = f"{role}_{timestamp}_{user_id}"
    embedding = embedder.encode(content).tolist()
    try:
        collection.add(
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
        print(f"[chroma_utils] Error saving message: {e}")

def get_relevant_context(user_id: str, query: str, n: int = 5) -> List[Dict]:
    try:
        query_emb = embedder.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=n,
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        return [
            {
                "role": meta.get("role", "unknown"),
                "content": doc
            }
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
    except Exception as e:
        print(f"[chroma_utils] Error retrieving context: {e}")
        return []

def get_chat_history(user_id: str) -> List[Dict]:
    try:
        results = collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        chat_log = [
            {
                "role": meta.get("role", "unknown"),
                "content": doc,
                "timestamp": meta.get("timestamp", 0)
            }
            for doc, meta in zip(results["documents"], results["metadatas"])
            if meta.get("type") == "chat_message" and meta.get("role") in ["user", "assistant"]
        ]
        chat_log.sort(key=lambda x: x["timestamp"])
        return chat_log
    except Exception as e:
        print(f"[chroma_utils] Error retrieving chat history: {e}")
        return []
