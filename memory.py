import os
import json
import time
import threading
import re
import shutil
from typing import Dict, Optional
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# === CONFIGURATION ===
COLLECTION_NAME = "Pinkys_Brain"
CHROMA_PERSIST_DIR = "./chroma_persist"
# (We no longer use PROFILE_STORE_PATH or _user_profiles)

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# === CHROMADB SETUP ===
print("[memory.py] Initializing ChromaDB...")

client = PersistentClient(path=CHROMA_PERSIST_DIR)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(COLLECTION_NAME)
USER_MEM_COLLECTION = "user_memory"
GLOBAL_MEM_COLLECTION = "global_memory"

user_mem_col = client.get_or_create_collection(USER_MEM_COLLECTION, embedding_function=embedding_fn)
global_mem_col = client.get_or_create_collection(GLOBAL_MEM_COLLECTION, embedding_function=embedding_fn)

# === USER PROFILE SYSTEM (per-user files) ===

def get_user_profile(user_id: str) -> Dict:
    """
    Returns the profile dict for this user. If no file exists, return a blank structure.
    """
    profiles_dir = "user_profiles"
    os.makedirs(profiles_dir, exist_ok=True)
    profile_path = os.path.join(profiles_dir, f"{user_id}.json")
    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            return json.load(f)
    else:
        return {"facts": {}}

def save_profile_to_disk(user_id: str, profile_data: Dict):
    """
    Write this user's profile back to disk. Overwrites existing file.
    """
    profiles_dir = "user_profiles"
    os.makedirs(profiles_dir, exist_ok=True)
    profile_path = os.path.join(profiles_dir, f"{user_id}.json")
    with open(profile_path, "w") as f:
        json.dump(profile_data, f, indent=2)
    print(f"[memory.py] Saved profile for user_id: {user_id}")

def update_user_fact(user_id: str, key: str, value):
    """
    Reads the per-user file, updates the `facts` sub-dict, and writes it back.
    """
    profile = get_user_profile(user_id)
    if "facts" not in profile:
        profile["facts"] = {}
    profile["facts"][key] = value
    save_profile_to_disk(user_id, profile)


# === VECTOR MEMORY SYSTEM ===

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

# === UTILITIES ===

def safe_extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"[safe_extract_json] Primary decode failed. Trying fallback...")
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                print(f"[safe_extract_json] Fallback parse failed: {e}")
        print(f"[safe_extract_json] Could not parse JSON from: {text}")
        return {}

# ===================================================================
# (We no longer need load_profiles_from_disk, save_profiles_to_disk, or _user_profiles)
# ===================================================================

def _autosave_worker(interval: int = 60):
    """This thread is no longer needed, since we save per-user on every update."""
    while True:
        time.sleep(interval)
        # We could implement a periodic backup of 'user_profiles/' directory if desired
        print("[memory.py] Autosave (noop) complete.")

def start_autosave(interval: int = 60):
    t = threading.Thread(target=_autosave_worker, args=(interval,), daemon=True)
    t.start()

if __name__ == "__main__":
    print("[memory.py] This module is not meant to be run directly.")
