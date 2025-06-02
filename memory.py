import os
import json
import chromadb
from chromadb.config import Settings

# ---------------------------
# Initialize ChromaDB Client
# ---------------------------
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="db"))

# Global memory (shared)
global_memory = client.get_or_create_collection(name="global_memory")

# ---------------------------
# User Profiles JSON Storage
# ---------------------------
USER_PROFILES_FILE = "user_profiles.json"

# Load user profiles from disk
if os.path.exists(USER_PROFILES_FILE):
    with open(USER_PROFILES_FILE, "r") as f:
        user_profiles = json.load(f)
else:
    user_profiles = {}

def save_user_profiles():
    with open(USER_PROFILES_FILE, "w") as f:
        json.dump(user_profiles, f, indent=2)

# ---------------------------
# User Profile Management
# ---------------------------
def get_user_profile(user_id):
    return user_profiles.get(user_id, {"facts": {}})

def update_user_fact(user_id, key, value):
    if user_id not in user_profiles:
        user_profiles[user_id] = {"facts": {}}
    user_profiles[user_id]["facts"][key] = value
    save_user_profiles()

# ---------------------------
# User Memory (ChromaDB per user)
# ---------------------------
def get_user_memory_collection(user_id):
    return client.get_or_create_collection(name=f"user_memory_{user_id}")

def add_user_memory(user_id, document, metadata=None):
    metadata = metadata or {}
    user_memory = get_user_memory_collection(user_id)
    doc_id = f"{user_id}_{abs(hash(document))}"
    user_memory.add(documents=[document], metadatas=[metadata], ids=[doc_id])

def query_user_memory(user_id, query, n_results=3):
    user_memory = get_user_memory_collection(user_id)
    return user_memory.query(query_texts=[query], n_results=n_results)

def query_global_memory(query, n_results=3):
    return global_memory.query(query_texts=[query], n_results=n_results)

def add_global_memory(document, metadata=None):
    metadata = metadata or {}
    doc_id = f"global_{abs(hash(document))}"
    global_memory.add(documents=[document], metadatas=[metadata], ids=[doc_id])
