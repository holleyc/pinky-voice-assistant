#!/usr/bin/env python3

import os
import json
import tempfile
import requests
import whisper
from uuid import uuid4

from flask import (
    Flask, request, jsonify, render_template,
    session, make_response, send_from_directory
)

from memory import (
    get_user_profile, update_user_fact,
    query_user_memory, add_user_memory,
    query_global_memory, add_global_memory,
    save_profiles_to_disk  # Optional manual save trigger
)

import atexit
atexit.register(save_profiles_to_disk)

from utils.qr_utils import generate_qr_base64
from utils.chroma_utils import (
    get_or_create_user_id, save_user_name, get_saved_user_name,
    save_message_to_chroma, get_relevant_context, get_chat_history
)

# Globals
app = Flask(__name__)
app.secret_key = "super_secret_key"  # Use env variable in production

model = whisper.load_model("base")
OLLAMA_URL = "http://localhost:11434/api/chat"

USER_PROFILES_DIR = "user_profiles"

# --- User Profile Helpers ---

def get_user_profile(user_id):
    if not os.path.exists(USER_PROFILES_DIR):
        os.makedirs(USER_PROFILES_DIR)
    profile_path = os.path.join(USER_PROFILES_DIR, f"{user_id}.json")
    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            return json.load(f)
    else:
        # Create new default profile
        profile = {"facts": {}, "memories": []}
        save_profile_to_disk(user_id, profile)
        return profile

def save_profile_to_disk(user_id, profile):
    if not os.path.exists(USER_PROFILES_DIR):
        os.makedirs(USER_PROFILES_DIR)
    profile_path = os.path.join(USER_PROFILES_DIR, f"{user_id}.json")
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

# --- Helper Functions ---

def get_user_id():
    return session.setdefault("user_id", get_or_create_user_id(request))

def chat_with_ollama(messages):
    payload = {"model": "pinky", "messages": messages, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")

def extract_lexical_facts(text):
    # Replace this with real NLP or LLM-based fact extraction
    return ["Fact 1", "Fact 2"]  # Dummy placeholder


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
    user_id = request.cookies.get("uuid")
    profile = get_user_profile(user_id)
    return jsonify(profile)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")
        user_id = get_user_id()

        print(f"[chat] user_id: {user_id}, input: {user_input}")

        # Load or create user profile
        user_profile = get_user_profile(user_id)
        user_name = user_profile.get("facts", {}).get("name")

        # Save user's message to vector user memory
        add_user_memory(user_id, user_input, metadata={"role": "user"})

        # If no user name known yet, treat first message as name
        if not user_name:
            user_profile["facts"]["name"] = user_input
            save_profile_to_disk(user_id, user_profile)
            return jsonify({"response": f"Nice to meet you, {user_input}!"})

        # Retrieve vector memory (chat history + relevant user data)
        user_mem_results = query_user_memory(user_id, user_input, n_results=5)
        global_mem_results = query_global_memory(user_input, n_results=3)

        chat_context = []
        for doc in user_mem_results.get("documents", [[]])[0]:
            chat_context.append({"role": "system", "content": doc})
        for doc in global_mem_results.get("documents", [[]])[0]:
            chat_context.append({"role": "system", "content": doc})

        messages = chat_context + [{"role": "user", "content": user_input}]

        response = chat_with_ollama(messages)
        if not response:
            return jsonify({"response": "⚠️ Empty or malformed LLM response."})

        # Save assistant reply to user memory
        add_user_memory(user_id, response, metadata={"role": "assistant"})

        # Extract lexical facts and update profile if applicable
        lexical_data = extract_lexical_facts(response)
        print(f"[chat] Lexical facts: {lexical_data}")

        if lexical_data:
            lexical_data = extract_lexical_facts(user_input)
            print(f"[chat] Lexical facts: {lexical_data}")

            if isinstance(lexical_data, dict):
                for key, value in lexical_data.items():
                    user_profile["facts"][key] = value
                save_profile_to_disk(user_id, user_profile)
            else:
                print(f"[chat] Warning: extract_lexical_facts returned non-dict: {type(lexical_data)}")

        return jsonify({"response": response})

    except requests.ConnectionError:
        print("[chat] ERROR: Ollama connection error")
        return jsonify({"response": "⚠️ Could not connect to Ollama."}), 503
    except requests.HTTPError as e:
        print(f"[chat] ERROR: Ollama HTTP error: {e}")
        return jsonify({"response": f"⚠️ Ollama HTTP error: {e}"}), 500
    except Exception as e:
        print(f"[chat] ERROR: Unexpected error: {e}")
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

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
