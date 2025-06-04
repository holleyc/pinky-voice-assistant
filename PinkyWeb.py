#!/usr/bin/env python3

import os
import json
import tempfile
import requests
import whisper
import re
import atexit
from uuid import uuid4
from flask import (
    Flask, request, jsonify, render_template,
    session, make_response, send_from_directory
)

# --- Internal Module Imports ---
from memory import (
    get_user_profile, update_user_fact,
    query_user_memory, add_user_memory,
    query_global_memory, add_global_memory,
    save_profiles_to_disk, save_profile_to_disk,
    safe_extract_json
)

from utils.qr_utils import generate_qr_base64
from utils.chroma_utils import (
    get_or_create_user_id, save_user_name, get_saved_user_name,
    save_message_to_chroma, get_relevant_context, get_chat_history
)

# Register profile save on exit
atexit.register(save_profiles_to_disk)

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Replace with environment variable in production

model = whisper.load_model("base")
OLLAMA_URL = "http://localhost:11434/api/chat"

# --- Helper Functions ---
def get_user_id():
    # Priority: JSON payload > Cookie > Session > Generate new
    user_id = (
        request.json.get("user_id")
        if request.is_json and "user_id" in request.json
        else request.cookies.get("user_id")
        or session.get("user_id")
    )

    if not user_id:
        user_id = str(uuid4())
        print(f"[get_user_id] Generated new user_id: {user_id}")
    else:
        print(f"[get_user_id] Using existing user_id: {user_id}")

    session["user_id"] = user_id
    return user_id


def chat_with_ollama(messages):
    payload = {"model": "pinky", "messages": messages, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")

import re

def extract_lexical_facts(user_input):
    facts = {}
    # Patterns for name
    name_patterns = [
        r"my name is (\w+)",
        r"i am (\w+)",
        r"call me (\w+)"
    ]
    for pattern in name_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            facts['name'] = match.group(1)
            break

    # Pattern for age
    age_match = re.search(r"my age is (\d+)", user_input, re.IGNORECASE)
    if age_match:
        facts['age'] = int(age_match.group(1))

    # Pattern for favorite color
    color_match = re.search(r"my favorite color is (\w+)", user_input, re.IGNORECASE)
    if color_match:
        facts['favorite_color'] = color_match.group(1)

    # Pattern for vehicle
    vehicle_match = re.search(r"i drive a (\w+)", user_input, re.IGNORECASE)
    if vehicle_match:
        facts['vehicle'] = vehicle_match.group(1)

    return facts


# --- Routes ---
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        save_user_name(get_user_id(), request.form["username"])
        return render_template("chat.html")
    return render_template("login.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No audio uploaded"}), 400

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp:
        audio.save(temp.name)
        try:
            result = model.transcribe(temp.name)
            return jsonify({"text": result["text"]})
        except Exception as e:
            return jsonify({"error": f"Transcription failed: {e}"}), 500

@app.route("/init", methods=["GET"])
def init_chat():
    user_id = get_user_id()
    name = get_saved_user_name(user_id)
    message = f"Welcome back, {name}!" if name else "Hi! What’s your name?"
    response = jsonify({"message": message})
    response.set_cookie("user_id", user_id, max_age=60 * 60 * 24 * 365 * 5)
    return response

@app.route("/profile")
def view_profile():
    user_id = session.get("user_id") or request.cookies.get("user_id")
    if not user_id:
        return jsonify({"error": "User not identified"}), 404
    profile = get_user_profile(user_id)
    return jsonify(profile)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")
        user_id = get_user_id()

        if not user_id:
            return jsonify({"response": "⚠️ User ID not found."}), 400

        print(f"[chat] user_id: {user_id}, input: {user_input}")

        user_profile = get_user_profile(user_id)
        user_name = user_profile.get("facts", {}).get("name")

        add_user_memory(user_id, user_input, metadata={"role": "user"})

        if not user_name:
            user_profile["facts"]["name"] = user_input
            save_profile_to_disk(user_id, user_profile)
            return jsonify({"response": f"Nice to meet you, {user_input}!"})

        user_mem_results = query_user_memory(user_id, user_input, n_results=5)
        global_mem_results = query_global_memory(user_input, n_results=3)

        chat_context = [
            {"role": "system", "content": doc}
            for doc in user_mem_results.get("documents", [[]])[0]
        ] + [
            {"role": "system", "content": doc}
            for doc in global_mem_results.get("documents", [[]])[0]
        ]

        messages = chat_context + [{"role": "user", "content": user_input}]
        response = chat_with_ollama(messages)

        if not response:
            return jsonify({"response": "⚠️ Empty or malformed LLM response."})

        add_user_memory(user_id, response, metadata={"role": "assistant"})

        lexical_data = extract_lexical_facts(user_input)
        print(f"[chat] Extracted lexical facts: {lexical_data}")

        if isinstance(lexical_data, dict) and lexical_data:
            for key, value in lexical_data.items():
                if value:
                    update_user_fact(user_id, key, value)
            save_profile_to_disk(user_id, get_user_profile(user_id))

        # Return both response AND lexical facts
        return jsonify({
            "response": response,
            "lexical_facts": lexical_data or {}
        })

    except requests.ConnectionError:
        return jsonify({"response": "⚠️ Could not connect to Ollama."}), 503
    except requests.HTTPError as e:
        return jsonify({"response": f"⚠️ Ollama HTTP error: {e}"}), 500
    except Exception as e:
        return jsonify({"response": f"⚠️ Unexpected error: {e}"}), 500


@app.route("/ollama_healthcheck")
def ollama_healthcheck():
    try:
        content = chat_with_ollama([{"role": "user", "content": "ping"}])
        return jsonify({"status": "success", "message": content or "No response content"})
    except requests.ConnectionError:
        return jsonify({"status": "error", "message": "❌ Ollama not reachable"}), 503
    except requests.HTTPError as e:
        return jsonify({"status": "error", "message": f"❌ HTTP error: {e}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"❌ Error: {e}"}), 500

@app.route("/qr")
@app.route("/pair")
def qr_pair():
    user_id = get_user_id()
    img_str = generate_qr_base64(user_id)
    if request.path == "/pair":
        return jsonify({"uuid": user_id, "qr_image_base64": img_str})
    return f"<h3>Scan to continue chat on mobile</h3><img src='data:image/png;base64,{img_str}'>"

@app.route("/history")
def history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "No user session found."}), 400
    return render_template("history.html", chat_log=get_chat_history(user_id))

@app.route("/whoami")
def whoami():
    user_id = session.get("user_id") or request.cookies.get("user_id")
    if not user_id:
        return jsonify({"error": "User not identified"}), 404
    session["user_id"] = user_id
    return jsonify({"user_id": user_id, "user_name": get_saved_user_name(user_id)})

@app.route("/reset")
def reset():
    session.clear()
    return jsonify({"message": "Session reset."})

@app.route("/images/<path:filename>")
def serve_images(filename):
    return send_from_directory('images', filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
