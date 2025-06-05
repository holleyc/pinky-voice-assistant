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
    save_profile_to_disk
)

from utils.qr_utils import generate_qr_base64
from utils.chroma_utils import (
    get_or_create_user_id, save_user_name, get_saved_user_name,
    save_message_to_chroma, get_relevant_context, get_chat_history
)

# Register profile save on exit (no-op since we save per-user immediately)
atexit.register(lambda: None)

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = "super_secret_key"  # Use a secure env var in production

# --- Globals ---
model = whisper.load_model("base")
OLLAMA_URL = "http://localhost:11434/api/chat"
USER_PROFILES_DIR = "user_profiles"  # Used by memory.py's get_user_profile / save_profile_to_disk

# --- Helper Functions ---
def get_user_id():
    # Priority: JSON payload > args > cookie > session > generate new
    user_id = None
    if request.is_json:
        user_id = request.json.get("user_id")
    if not user_id:
        user_id = request.args.get("user_id")
    if not user_id:
        user_id = request.cookies.get("user_id") or session.get("user_id")
    if not user_id:
        user_id = get_or_create_user_id(request)
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

def extract_lexical_facts(text):
    # Regex-based extraction for common user facts
    facts = {}

    # 1) Name patterns
    name_match = re.search(r"\b(?:my name is|i am|call me)\s+([A-Za-z]+)\b", text, re.IGNORECASE)
    if name_match:
        facts['name'] = name_match.group(1)
    else:
        single_word = text.strip()
        if re.fullmatch(r"[A-Z][a-z]+", single_word):
            facts['name'] = single_word

    # 2) Age pattern
    age_match = re.search(r"\bmy age is\s+(\d+)\b", text, re.IGNORECASE)
    if age_match:
        facts['age'] = int(age_match.group(1))

    # 3) Favorite color pattern
    color_match = re.search(r"\bmy favorite color is\s+([A-Za-z]+)\b", text, re.IGNORECASE)
    if color_match:
        facts['favorite_color'] = color_match.group(1)

    # 4) Favorite number pattern
    number_match = re.search(r"\bmy favorite number is\s+(\d+)\b", text, re.IGNORECASE)
    if number_match:
        facts['favorite_number'] = int(number_match.group(1))

    # 5) Vehicle pattern (e.g., 'i drive a toyota camry')
    vehicle_match = re.search(r"\bi drive a\s+([A-Za-z\s]+)\b", text, re.IGNORECASE)
    if vehicle_match:
        facts['vehicle'] = vehicle_match.group(1).strip()

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
    # Ensure user profile file exists (blank if new)
    user_profile = get_user_profile(user_id)
    save_profile_to_disk(user_id, user_profile)

    name = get_saved_user_name(user_id)
    message = f"Welcome back, {name}!" if name else "Hi! What’s your name?"
    response = jsonify({"message": message, "user_id": user_id})
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
        data = request.get_json() or {}
        user_input = data.get("message", "").strip()
        user_id = get_user_id()

        if not user_id:
            return jsonify({"response": "⚠️ User ID not found."}), 400

        print(f"[chat] user_id: {user_id}, input: {user_input}")

        user_profile = get_user_profile(user_id)
        user_name = user_profile.get("facts", {}).get("name")

        # 1) Handle retrieval queries: e.g. "what is my favorite color?" or "what is my favorite number?"
        retrieve_match = re.search(r"\bwhat is my ([A-Za-z\s]+)\??", user_input, re.IGNORECASE)
        if retrieve_match:
            key_raw = retrieve_match.group(1).strip().lower()
            key = key_raw.replace(" ", "_")
            value = user_profile.get("facts", {}).get(key)
            if value is not None:
                return jsonify({
                    "response": f"Your {key_raw} is {value}.",
                    "lexical_facts": {}
                })
            else:
                return jsonify({
                    "response": f"I don't know your {key_raw} yet.",
                    "lexical_facts": {}
                })

        # 2) Add to vector memory
        add_user_memory(user_id, user_input, metadata={"role": "user"})

        # 3) If no name yet, treat this input as name
        if not user_name:
            user_profile["facts"]["name"] = user_input
            save_profile_to_disk(user_id, user_profile)
            return jsonify({
                "response": f"Nice to meet you, {user_input}!",
                "lexical_facts": {}
            })

        # 4) Normal LLM chat flow
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
        response_text = chat_with_ollama(messages)

        if not response_text:
            return jsonify({
                "response": "⚠️ Empty or malformed LLM response.",
                "lexical_facts": {}
            })

        add_user_memory(user_id, response_text, metadata={"role": "assistant"})

        # 5) Extract new lexical facts from what the user just said
        lexical_data = extract_lexical_facts(user_input)
        print(f"[chat] Extracted lexical facts: {lexical_data}")

        if isinstance(lexical_data, dict) and lexical_data:
            # Update per-user profile file
            for key, value in lexical_data.items():
                if value is not None:
                    update_user_fact(user_id, key, value)
            # At this point, `update_user_fact` already saved the file

        return jsonify({
            "response": response_text,
            "lexical_facts": lexical_data
        })

    except requests.ConnectionError:
        return jsonify({"response": "⚠️ Could not connect to Ollama."}), 503
    except requests.HTTPError as e:
        return jsonify({
            "response": f"⚠️ Ollama HTTP error: {e}",
            "lexical_facts": {}
        }), 500
    except Exception as e:
        return jsonify({
            "response": f"⚠️ Unexpected error: {e}",
            "lexical_facts": {}
        }), 500

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

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
