import os
from typing import List, Dict, Optional
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---- CONFIG ----

CHROMA_DIR = "./chroma_data"
os.makedirs(CHROMA_DIR, exist_ok=True)

# Initialize Chroma client (persistent on disk)
chroma_client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMA_DIR
))

# Embedding function (replace with your actual embedding model, e.g. SentenceTransformer)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Collections:
USER_MEM_COLLECTION = "user_memory"
GLOBAL_MEM_COLLECTION = "global_memory"

# Create or get collections
try:
    user_mem_col = chroma_client.get_collection(USER_MEM_COLLECTION, embedding_function=embedding_fn)
except:
    user_mem_col = chroma_client.create_collection(USER_MEM_COLLECTION, embedding_function=embedding_fn)

try:
    global_mem_col = chroma_client.get_collection(GLOBAL_MEM_COLLECTION, embedding_function=embedding_fn)
except:
    global_mem_col = chroma_client.create_collection(GLOBAL_MEM_COLLECTION, embedding_function=embedding_fn)


# --- USER PROFILE STORAGE ---

# Simple in-memory dict for user facts: { user_id: { fact_key: fact_value, ... } }
_user_profiles: Dict[str, Dict[str, str]] = {}


def get_user_profile(user_id: str) -> Dict:
    """Return user profile dict with facts or empty dict."""
    return _user_profiles.get(user_id, {"facts": {}})


def update_user_fact(user_id: str, key: str, value: str):
    """Add or update a single user fact."""
    if user_id not in _user_profiles:
        _user_profiles[user_id] = {"facts": {}}
    _user_profiles[user_id]["facts"][key] = value


# --- USER MEMORY FUNCTIONS ---


def add_user_memory(user_id: str, text: str, metadata: Optional[Dict] = None):
    """Add a text snippet to the user's memory collection with optional metadata."""
    if metadata is None:
        metadata = {}

    # Use user_id as a metadata field to filter later
    metadata["user_id"] = user_id

    user_mem_col.add(
        documents=[text],
        metadatas=[metadata],
        ids=[f"{user_id}_{os.urandom(8).hex()}"]
    )
    chroma_client.persist()


def query_user_memory(user_id: str, query_text: str, n_results: int = 5) -> Dict:
    """Query the user memory collection filtered by user_id and return top docs."""
    results = user_mem_col.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"user_id": user_id}
    )
    return results


# --- GLOBAL MEMORY FUNCTIONS ---


def add_global_memory(text: str, metadata: Optional[Dict] = None):
    """Add a text snippet to the global memory collection."""
    if metadata is None:
        metadata = {}

    global_mem_col.add(
        documents=[text],
        metadatas=[metadata],
        ids=[f"global_{os.urandom(8).hex()}"]
    )
    chroma_client.persist()


def query_global_memory(query_text: str, n_results: int = 5) -> Dict:
    """Query the global memory collection and return top docs."""
    results = global_mem_col.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results
