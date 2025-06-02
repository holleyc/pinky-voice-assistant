import os
import json
import threading
import time
from typing import Dict, Optional
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# ---- CONFIG ----
COLLECTION_NAME = "Pinkys_Brain"
CHROMA_PERSIST_DIR = "./chroma_persist"
PROFILE_STORE_PATH = "./user_profiles.json"  # JSON file path
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# ---- CHROMADB INIT ----
print("Initializing ChromaDB...")
client = PersistentClient(path=CHROMA_PERSIST_DIR)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Main collection (used or extended as needed)
collection = client.get_or_create_collection(COLLECTION_NAME)

# User and Global Memory
USER_MEM_COLLECTION = "user_memory"
GLOBAL_MEM_COLLECTION = "global_memory"

user_mem_col = client.get_or_create_collection(USER_MEM_COLLECTION, embedding_function=embedding_fn)
global_mem_col = client.get_or_create_collection(GLOBAL_MEM_COLLECTION, embedding_function=embedding_fn)

# ---- USER PROFILE STORE ----
_user_profiles: Dict[str, Dict[str, str]] = {}

def load_profiles_from_disk():
    global _user_profiles
    try:
        with open(PROFILE_STORE_PATH, "r") as f:
            _user_profiles = json.load(f)
        print(f"[memory.py] Loaded profiles: {list(_user_profiles.keys())}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        _user_profiles = {}
        print(f"[memory.py] Failed to load profiles: {e} — Starting fresh.")

def save_profiles_to_disk():
    try:
        with open(PROFILE_STORE_PATH, "w") as f:
            json.dump(_user_profiles, f, indent=2)
        print("[memory.py] User profiles saved.")
    except Exception as e:
        print(f"[memory.py] Error saving profiles: {e}")

def _autosave_worker(interval=60):
    while True:
        time.sleep(interval)
        save_profiles_to_disk()
        print("[memory.py] Autosave complete.")

def start_autosave(interval=60):
    threading.Thread(target=_autosave_worker, args=(interval,), daemon=True).start()

# ---- USER PROFILE MANAGEMENT ----
def get_user_profile(user_id: str) -> Dict:
    return _user_profiles.get(user_id, {"facts": {}})

def update_user_fact(user_id: str, key: str, value: str):
    if user_id not in _user_profiles:
        _user_profiles[user_id] = {"facts": {}}
    _user_profiles[user_id]["facts"][key] = value

# ---- MEMORY MANAGEMENT ----
def add_user_memory(user_id: str, text: str, metadata: Optional[Dict] = None):
    metadata = metadata or {}
    metadata["user_id"] = user_id
    user_mem_col.add(
        documents=[text],
        metadatas=[metadata],
        ids=[f"{user_id}_{os.urandom(8).hex()}"]
    )

def query_user_memory(user_id: str, query_text: str, n_results: int = 5) -> Dict:
    return user_mem_col.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"user_id": user_id}
    )

def add_global_memory(text: str, metadata: Optional[Dict] = None):
    metadata = metadata or {}
    global_mem_col.add(
        documents=[text],
        metadatas=[metadata],
        ids=[f"global_{os.urandom(8).hex()}"]
    )

def query_global_memory(query_text: str, n_results: int = 5) -> Dict:
    return global_mem_col.query(
        query_texts=[query_text],
        n_results=n_results
    )

# ---- INIT ----
load_profiles_from_disk()
start_autosave(interval=60)
