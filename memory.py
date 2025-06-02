import os
import json
import threading
import time
from typing import List, Dict, Optional

from chromadb import Client

from chromadb.utils import embedding_functions

# ---- CONFIG ----

CHROMA_DIR = "./chroma_data"
os.makedirs(CHROMA_DIR, exist_ok=True)

PROFILE_STORE_PATH = "./user_profiles.json"  # JSON file path

# Initialize Chroma client (persistent on disk)
chroma_client = Client(
    persist_directory="./chroma_persist",
    chroma_db_impl="duckdb+parquet"
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

USER_MEM_COLLECTION = "user_memory"
GLOBAL_MEM_COLLECTION = "global_memory"

try:
    user_mem_col = chroma_client.get_collection(USER_MEM_COLLECTION, embedding_function=embedding_fn)
except:
    user_mem_col = chroma_client.create_collection(USER_MEM_COLLECTION, embedding_function=embedding_fn)

try:
    global_mem_col = chroma_client.get_collection(GLOBAL_MEM_COLLECTION, embedding_function=embedding_fn)
except:
    global_mem_col = chroma_client.create_collection(GLOBAL_MEM_COLLECTION, embedding_function=embedding_fn)

# --- USER PROFILES IN MEMORY ---
_user_profiles: Dict[str, Dict[str, str]] = {}

# --- LOAD/ SAVE USER PROFILES ---


def load_profiles_from_disk():
    global _user_profiles
    try:
        with open(PROFILE_STORE_PATH, "r") as f:
            _user_profiles = json.load(f)
        print(f"[memory.py] Loaded profiles for users: {list(_user_profiles.keys())}")
    except FileNotFoundError:
        _user_profiles = {}
        print("[memory.py] No profile file found. Starting with empty profiles.")
    except json.JSONDecodeError as e:
        _user_profiles = {}
        print(f"[memory.py] JSON decode error loading profiles: {e}. Starting fresh.")


def save_profiles_to_disk():
    try:
        with open(PROFILE_STORE_PATH, "w") as f:
            json.dump(_user_profiles, f, indent=2)
        print("[memory.py] User profiles saved to disk.")
    except Exception as e:
        print(f"[memory.py] Error saving profiles: {e}")


# --- AUTOSAVE THREAD ---

def _autosave_worker(interval=60):
    while True:
        time.sleep(interval)
        save_profiles_to_disk()
        # Optionally: print or log autosave event
        print("[memory.py] Autosave complete.")


def start_autosave(interval=60):
    t = threading.Thread(target=_autosave_worker, args=(interval,), daemon=True)
    t.start()


# --- USER PROFILE FUNCTIONS ---


def get_user_profile(user_id: str) -> Dict:
    """Return user profile dict with facts or empty dict."""
    return _user_profiles.get(user_id, {"facts": {}})


def update_user_fact(user_id: str, key: str, value: str):
    """Add or update a single user fact."""
    if user_id not in _user_profiles:
        _user_profiles[user_id] = {"facts": {}}
    _user_profiles[user_id]["facts"][key] = value
    # save_profiles_to_disk()  # Removed immediate save, rely on autosave


# --- USER MEMORY FUNCTIONS ---


def add_user_memory(user_id: str, text: str, metadata: Optional[Dict] = None):
    if metadata is None:
        metadata = {}

    metadata["user_id"] = user_id

    user_mem_col.add(
        documents=[text],
        metadatas=[metadata],
        ids=[f"{user_id}_{os.urandom(8).hex()}"]
    )
    chroma_client.persist()


def query_user_memory(user_id: str, query_text: str, n_results: int = 5) -> Dict:
    results = user_mem_col.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"user_id": user_id}
    )
    return results


# --- GLOBAL MEMORY FUNCTIONS ---


def add_global_memory(text: str, metadata: Optional[Dict] = None):
    if metadata is None:
        metadata = {}

    global_mem_col.add(
        documents=[text],
        metadatas=[metadata],
        ids=[f"global_{os.urandom(8).hex()}"]
    )
    chroma_client.persist()


def query_global_memory(query_text: str, n_results: int = 5) -> Dict:
    results = global_mem_col.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results


# --- INIT ---

load_profiles_from_disk()
start_autosave(interval=60)  # autosave every 60 seconds
