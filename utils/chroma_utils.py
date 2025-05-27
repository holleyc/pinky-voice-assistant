import uuid
import time
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="Pinkys_Brain")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def generate_uuid():
    return str(uuid.uuid4())

def get_or_create_user_id(request):
    return request.cookies.get("user_id") or request.args.get("user_id") or generate_uuid()

def save_user_name(user_id, name):
    embedding = embedder.encode(name).tolist()
    collection.add(
        documents=[name],
        metadatas=[{"type": "user_name", "user_id": user_id}],
        ids=[f"user_name_{user_id}"],
        embeddings=[embedding]
    )

def get_saved_user_name(user_id):
    results = collection.get(where={"user_id": user_id}, include=["documents", "metadatas"])
    for doc, meta in zip(results["documents"], results["metadatas"]):
        if meta.get("type") == "user_name":
            return doc
    return None

def save_message_to_chroma(user_id, role, content):
    timestamp = int(time.time() * 1000)
    doc_id = f"{role}_{timestamp}_{user_id}"
    embedding = embedder.encode(content).tolist()
    collection.add(
        documents=[content],
        metadatas=[{"role": role, "user_id": user_id, "timestamp": timestamp}],
        ids=[doc_id],
        embeddings=[embedding]
    )

def get_relevant_context(user_id, query, n=5):
    query_emb = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n,
        where={"user_id": user_id},
        include=["documents", "metadatas"]
    )
    return [
        {"role": meta.get("role", "unknown"), "content": doc}
        for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    ]

def get_chat_history(user_id):
    results = collection.get(
        where={"user_id": user_id},
        include=["documents", "metadatas"]
    )
    chat_log = [
        {
            "role": meta["role"],
            "content": doc,
            "timestamp": meta.get("timestamp", 0)
        }
        for doc, meta in zip(results["documents"], results["metadatas"])
        if meta.get("role") in ["user", "assistant"]
    ]
    chat_log.sort(key=lambda x: x["timestamp"])
    return chat_log
