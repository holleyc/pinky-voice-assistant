import os
import json
import time
import threading
import re
from typing import Dict, Optional
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# === CONFIGURATION ===
COLLECTION_NAME      = "Pinkys_Brain"
CHROMA_PERSIST_DIR   = "./chroma_persist"
# Profile directory override via env var PROFILE_DIR
PROFILE_DIR_ENV      = "PROFILE_DIR"

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# === CHROMADB SETUP ===
print("[memory.py] Initializing ChromaDB...")
client       = PersistentClient(path=CHROMA_PERSIST_DIR)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection(COLLECTION_NAME)
USER_MEM_COLLECTION   = "user_memory"
GLOBAL_MEM_COLLECTION = "global_memory"
user_mem_col   = client.get_or_create_collection(USER_MEM_COLLECTION, embedding_function=embedding_fn)
global_mem_col = client.get_or_create_collection(GLOBAL_MEM_COLLECTION, embedding_function=embedding_fn)

# === USER PROFILE SYSTEM (per-user files) ===

def _profiles_dir() -> str:
    """
    Return profiles directory, overrideable via PROFILE_DIR env var.
    """
    return os.getenv(PROFILE_DIR_ENV, "user_profiles")


def get_user_profile(user_id: str) -> Dict:
    """
    Returns the profile dict for this user. If no file exists, return blank.
    """
    profiles_dir = _profiles_dir()
    os.makedirs(profiles_dir, exist_ok=True)
    path = os.path.join(profiles_dir, f"{user_id}.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        # corrupt, overwrite
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"facts": {}}, f, indent=2)
        except Exception:
            pass
    return {"facts": {}}


def save_profile_to_disk(user_id: str, profile_data: Dict):
    """
    Write user profile to disk, overwriting existing.
    """
    profiles_dir = _profiles_dir()
    os.makedirs(profiles_dir, exist_ok=True)
    path = os.path.join(profiles_dir, f"{user_id}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)
        print(f"[memory.py] Saved profile for user_id: {user_id}")
    except Exception as e:
        print(f"[memory.py] Error saving profile for {user_id}: {e}")


def update_user_fact(user_id: str, key: str, value):
    """
    Update and persist one fact in user's profile.
    """
    profile = get_user_profile(user_id)
    profile.setdefault("facts", {})[key] = value
    save_profile_to_disk(user_id, profile)

# === VECTOR MEMORY SYSTEM ===

def add_user_memory(user_id: str, text: str, metadata: Optional[Dict] = None):
    """
    Add a user-specific memory vector; uses metadatas=[metadata].
    """
    meta = metadata.copy() if metadata else {}
    meta["user_id"] = user_id
    try:
        user_mem_col.add(
            documents=[text],
            metadatas=[meta],
            ids=[f"{user_id}_{os.urandom(8).hex()}"]
        )
    except Exception as e:
        print(f"[memory.py] add_user_memory error for {user_id}: {e}")


def query_user_memory(user_id: str, query_text: str, n_results: int = 5) -> Dict:
    try:
        return user_mem_col.query(
            query_texts=[query_text], n_results=n_results, where={"user_id": user_id}
        )
    except Exception:
        return {"documents": [[]]}


def add_global_memory(text: str, metadata: Optional[Dict] = None):
    meta = metadata.copy() if metadata else {}
    try:
        global_mem_col.add(
            documents=[text],
            metadatas=[meta],
            ids=[f"global_{os.urandom(8).hex()}"]
        )
    except Exception as e:
        print(f"[memory.py] add_global_memory error: {e}")


def query_global_memory(query_text: str, n_results: int = 5) -> Dict:
    try:
        return global_mem_col.query(query_texts=[query_text], n_results=n_results)
    except Exception:
        return {"documents": [[]]}

# === UTILITIES ===

def safe_extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"[safe_extract_json] Primary decode failed")
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                print(f"[safe_extract_json] Fallback parse failed: {e}")
        return {}

if __name__ == "__main__":
    print("[memory.py] This module is not meant to be run directly.")
