#!/usr/bin/env python3

import uuid
import time
import qrcode
import base64
from io import BytesIO

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template, session, make_response
import requests

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="Pinkys_Brain")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Replace with a secure key in production
OLLAMA_URL = "http://localhost:11434/api/chat"

# Utilities
def generate_uuid():
    return str(uuid.uuid4())

def save_message_to_chroma(role, content, custom_id=None):
    embedding = embedder.encode(content).tolist()
    doc_id = custom_id if custom_id else f"{role}_{int(time.time() * 1000)}"
    collection.add(
        documents=[content],
        metadatas=[{"role": role}],
        ids=[doc_id],
        embeddings=[embedding]
    )

def save_user_name(name):
    user_id = session.get("user_id")
    if not user_id:
        user_id = generate_uuid()
        session["user_id"] = user_id
    embedding = embedder.encode(name).tolist()
    collection.add(
        documents=[name],
        metadatas=[{"type": "user_name", "user_id": user_id}],
        ids=[f"user_name_{user_id}"],
        embeddings=[embedding]
    )

def get_saved_user_name():
    user_id = session.get("user_id")
    if not user_id:
        return None
    results = collection.get(
        where={"user_id": user_id},
        include=["documents", "metadatas"]
    )
    for doc, meta in zip(results['documents'], results['metadatas']):
        if meta.get("type") == "user_name":
            return doc
    return None

def get_relevant_context(query, n=3):
    query_emb = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n,
        include=["documents", "metadatas"]
    )
    context = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        context.append({"role": meta["role"], "content": doc})
    return context

# Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/init", methods=["GET"])
def init_chat():
    user_id = request.cookies.get("user_id") or request.args.get("user_id")
    if not user_id:
        user_id = generate_uuid()
    session["user_id"] = user_id

    user_name = get_saved_user_name()
    message = f"Welcome back, {user_name}!" if user_name else "Hi! What’s your name?"
    resp = make_response(jsonify({"message": message}))
    resp.set_cookie("user_id", user_id, max_age=60 * 60 * 24 * 365 * 5)
    return resp

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if "user_id" not in session:
        session["user_id"] = request.cookies.get("user_id") or generate_uuid()

    user_name = get_saved_user_name()
    if not user_name:
        save_message_to_chroma("user", user_input)
        save_user_name(user_input)
        return jsonify({"response": f"Nice to meet you, {user_input}!"})

    save_message_to_chroma("user", user_input)
    context_messages = get_relevant_context(user_input, n=3)
    messages = context_messages + [{"role": "user", "content": user_input}]

    payload = {
        "model": "pinky",
        "messages": messages,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    assistant_response = response.json().get("message", {}).get("content", "")

    save_message_to_chroma("assistant", assistant_response)
    return jsonify({"response": assistant_response})

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        user_id = session.get("user_id") or generate_uuid()
        session["user_id"] = user_id
        save_user_name(username)
        return render_template("chat.html")
    return render_template("login.html")

@app.route("/qr")
def qr_pair():
    user_id = session.get("user_id") or generate_uuid()
    session["user_id"] = user_id
    url = f"http://localhost:5000/?user_id={user_id}"
    img = qrcode.make(url)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"<h3>Scan to continue chat on mobile</h3><img src='data:image/png;base64,{img_str}'>"

@app.route("/reset", methods=["GET"])
def reset():
    session.clear()
    return jsonify({"message": "Session reset."})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
