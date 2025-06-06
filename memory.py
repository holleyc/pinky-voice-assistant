import os
import json
import time
import threading
import re
import shutil
from typing import Dict, Optional
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from datetime import datetime

# ------------------------------------------------------------------------------
#                                 Setup Logging
# ------------------------------------------------------------------------------

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "memory_events.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

def log_memory_event(event_type: str, data: dict):
    """
    Append a JSON‐line record describing a memory operation.
    Each line includes a UTC timestamp, an "event" field, and the provided data fields.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event_type,
        **data
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# === CONFIGURATION ===
COLLECTION_NAME = "Pinkys_Brain"
CHROMA_PERSIST_DIR = "./chroma_persist"

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


# ------------------------------------------------------------------------------
#                         USER PROFILE SYSTEM (per-user files)
# ------------------------------------------------------------------------------

def get_user_profile(user_id: str) -> Dict:
    """
    Returns the profile dict for this user. If no file exists, return a blank structure.
    Logs a 'load_profile' event, noting whether a file was found or a new default was returned.
    """
    profiles_dir = "user_profiles"
    os.makedirs(profiles_dir, exist_ok=True)
    profile_path = os.path.join(profiles_dir, f"{user_id}.json")

    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        log_memory_event("load_profile", {
            "user_id": user_id,
            "status": "loaded_existing",
            "profile_keys": list(profile.get("facts", {}).keys())
        })
        return profile
    else:
        default_profile = {"facts": {}}
        log_memory_event("load_profile", {
            "user_id": user_id,
            "status": "new_profile",
            "profile_keys": []
        })
        return default_profile


def save_profile_to_disk(user_id: str, profile_data: Dict):
    """
    Write this user's profile back to disk. Overwrites existing file.
    Logs a 'save_profile' event with the full 'facts' dictionary.
    """
    profiles_dir = "user_profiles"
    os.makedirs(profiles_dir, exist_ok=True)
    profile_path = os.path.join(profiles_dir, f"{user_id}.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)
    log_memory_event("save_profile", {
        "user_id": user_id,
        "profile_keys": list(profile_data.get("facts", {}).keys()),
        "full_profile": profile_data.get("facts", {})
    })
    print(f"[memory.py] Saved profile for user_id: {user_id}")


def update_user_fact(user_id: str, key: str, value):
    """
    1) Read the existing per-user JSON
    2) Update the fact in that dict
    3) Persist immediately to disk in user_profiles/<user_id>.json
    Logs:
      - 'update_user_fact' before writing (with old_value, new_value)
      - 'save_profile' (implicitly called inside) 
    """
    profile = get_user_profile(user_id)
    old_value = profile.get("facts", {}).get(key)

    if "facts" not in profile:
        profile["facts"] = {}

    profile["facts"][key] = value

    log_memory_event("update_user_fact", {
        "user_id": user_id,
        "fact_key": key,
        "old_value": old_value,
        "new_value": value
    })

    save_profile_to_disk(user_id, profile)


# ------------------------------------------------------------------------------
#                         VECTOR MEMORY SYSTEM (ChromaDB)
# ------------------------------------------------------------------------------

def add_user_memory(user_id: str, text: str, metadata: Optional[Dict] = None):
    """
    Add a user-specific memory vector. Logs 'add_user_memory' with text length and metadata.
    """
    metadata = metadata or {}
    metadata["user_id"] = user_id

    try:
        user_mem_col.add(
            documents=[text],
            metadatas=[metadata],
            ids=[f"{user_id}_{os.urandom(8).hex()}"]
        )
        log_memory_event("add_user_memory", {
            "user_id": user_id,
            "text_snippet": text[:50],
            "text_length": len(text),
            "metadata": metadata
        })
    except Exception as e:
        log_memory_event("add_user_memory_error", {
            "user_id": user_id,
            "error": str(e),
            "text_snippet": text[:50]
        })
        raise


def query_user_memory(user_id: str, query_text: str, n_results: int = 5) -> Dict:
    """
    Query a user's vector memory. Logs 'query_user_memory' with query text and number of hits.
    """
    try:
        results = user_mem_col.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"user_id": user_id}
        )
        hit_count = len(results.get("documents", [[]])[0])
        log_memory_event("query_user_memory", {
            "user_id": user_id,
            "query_text": query_text,
            "n_results": n_results,
            "hit_count": hit_count
        })
        return results
    except Exception as e:
        log_memory_event("query_user_memory_error", {
            "user_id": user_id,
            "query_text": query_text,
            "error": str(e)
        })
        raise


def add_global_memory(text: str, metadata: Optional[Dict] = None):
    """
    Add a global memory vector. Logs 'add_global_memory'.
    """
    metadata = metadata or {}
    try:
        global_mem_col.add(
            documents=[text],
            metadatas=[metadata],
            ids=[f"global_{os.urandom(8).hex()}"]
        )
        log_memory_event("add_global_memory", {
            "text_snippet": text[:50],
            "text_length": len(text),
            "metadata": metadata
        })
    except Exception as e:
        log_memory_event("add_global_memory_error", {
            "error": str(e),
            "text_snippet": text[:50]
        })
        raise


def query_global_memory(query_text: str, n_results: int = 5) -> Dict:
    """
    Query global memory vectors. Logs 'query_global_memory' with query text and number of hits.
    """
    try:
        results = global_mem_col.query(
            query_texts=[query_text],
            n_results=n_results
        )
        hit_count = len(results.get("documents", [[]])[0])
        log_memory_event("query_global_memory", {
            "query_text": query_text,
            "n_results": n_results,
            "hit_count": hit_count
        })
        return results
    except Exception as e:
        log_memory_event("query_global_memory_error", {
            "query_text": query_text,
            "error": str(e)
        })
        raise


# ------------------------------------------------------------------------------
#                                  UTILITIES
# ------------------------------------------------------------------------------

def safe_extract_json(text: str) -> dict:
    """
    Try to safely extract JSON from a text blob. Logs fallback attempts on failure.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log_memory_event("safe_extract_json_fallback", {
            "input_text": text[:200]
        })
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                log_memory_event("safe_extract_json_success", {
                    "input_text": text[:200],
                    "parsed_fragment": parsed
                })
                return parsed
            except Exception as e:
                log_memory_event("safe_extract_json_failure", {
                    "input_text": text[:200],
                    "error": str(e)
                })
        else:
            log_memory_event("safe_extract_json_no_braces", {
                "input_text": text[:200]
            })
        return {}


def _autosave_worker(interval: int = 60):
    """
    This thread is no longer needed for per-user persistence,
    but we keep it in case you want to implement periodic backups of the
    entire 'user_profiles/' directory in the future.
    """
    while True:
        time.sleep(interval)
        # Example: copy 'user_profiles/' folder to a backup folder:
        # shutil.copytree("user_profiles", f"backups/user_profiles_{int(time.time())}", dirs_exist_ok=True)
        log_memory_event("autosave_noop", {"message": "Periodic noop backup placeholder"})


def start_autosave(interval: int = 60):
    t = threading.Thread(target=_autosave_worker, args=(interval,), daemon=True)
    t.start()


if __name__ == "__main__":
    print("[memory.py] This module is not meant to be run directly.")
