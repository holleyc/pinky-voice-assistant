import os
import json
import time
import threading
import re
from typing import Dict, Optional
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# === CONFIGURATION ===
COLLECTION_NAME = "Pinkys_Brain"
CHROMA_PERSIST_DIR = "./chroma_persist"
PROFILE_STORE_PATH = "./user_profiles.json"

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# === CHROMADB SETUP ===
print("[memory.py] Initializing ChromaDB...")

client = PersistentClient(path=CHROMA_PERSIST_DIR)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Core and auxiliary collections
collection = client.get_or_create_collection(COLLECTION_NAME)
USER_MEM_COLLECTION = "user_memory"
GLOBAL_MEM_COLLECTION = "global_memory"

user_mem_col = client.get_or_create_collection(USER_MEM_COLLECTION, embedding_function=embedding_fn)
global_mem_col = client.get_or_create_collection(GLOBAL_MEM_COLLECTION, embedding_function=embedding_fn)

# === USER PROFILE SYSTEM ===
_user_profiles: Dict[str, Dict[str, str]] = {}

def load_profiles_from_disk():
    """Load user profiles from disk into memory."""
    global _user_profiles
    try:
        with open(PROFILE_STORE_PATH, "r") as f:
            _user_profiles = json.load(f)
        print(f"[memory.py] Loaded profiles: {list(_user_profiles.keys())}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        _user_profiles = {}
        print(f"[memory.py] Failed to load profiles: {e} — Starting fresh.")

def save_profiles_to_disk():
    """Persist all user profiles to disk."""
    try:
        with open(PROFILE_STORE_PATH, "w") as f:
            json.dump(_user_profiles, f, indent=2)
        print("[memory.py] User profiles saved.")
    except Exception as e:
        print(f"[memory.py] Error saving profiles: {e}")

def save_profile_to_disk(user_id: str, profile: Dict):
    """Save a single user profile to the store and persist."""
    _user_profiles[user_id] = profile
    save_profiles_to_disk()

def _autosave_worker(interval: int = 60):
    """Worker thread for periodically saving profiles."""
    while True:
        time.sleep(interval)
        save_profiles_to_disk()
        print("[memory.py] Autosave complete.")

def start_autosave(interval: int = 60):
    """Start autosave thread."""
    threading.Thread(target=_autosave_worker, args=(interval,), daemon=True).start()

def get_user_profile(user_id: str) -> Dict:
    """Retrieve user profile or create a default one."""
    return _user_profiles.get(user_id, {"facts": {}})

def update_user_fact(user_id: str, key: str, value: str):
    """Update a single fact in a user's profile."""
    if user_id not in _user_profiles:
        _user_profiles[user_id] = {"facts": {}}
    _user_profiles[user_id]["facts"][key] = value

# === VECTOR MEMORY SYSTEM ===

def add_user_memory(user_id: str, text: str, metadata: Optional[Dict] = None):
    """Add a user-specific memory vector."""
    metadata = metadata or {}
    metadata["user_id"] = user_id
    user_mem_col.add(
        documents=[text],
        metadatas=[metadata],
        ids=[f"{user_id}_{os.urandom(8).hex()}"]
    )

def query_user_memory(user_id: str, query_text: str, n_results: int = 5) -> Dict:
    """Query a user's vector memory."""
    return user_mem_col.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"user_id": user_id}
    )

def add_global_memory(text: str, metadata: Optional[Dict] = None):
    """Add a global memory vector."""
    metadata = metadata or {}
    global_mem_col.add(
        documents=[text],
        metadatas=[metadata],
        ids=[f"global_{os.urandom(8).hex()}"]
    )

def query_global_memory(query_text: str, n_results: int = 5) -> Dict:
    """Query global memory vectors."""
    return global_mem_col.query(
        query_texts=[query_text],
        n_results=n_results
    )

# === UTILITIES ===

def safe_extract_json(text: str) -> dict:
    """Try to safely extract JSON from a text blob."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                print(f"[safe_extract_json] Nested parse error: {e}")
        print(f"[safe_extract_json] Failed to parse JSON from: {text}")
        return {}

# === INIT ON IMPORT ===
load_profiles_from_disk()
start_autosave(interval=60)
