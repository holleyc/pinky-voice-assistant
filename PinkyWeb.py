#!/usr/bin/env python3

import os
import json
import tempfile
import requests
import whisper
import re
import atexit
from uuid import uuid4
from datetime import datetime

from flask import (
    Flask, request, jsonify, render_template,
    session, send_from_directory
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

# ------------------------------------------------------------------------------
#                                 Setup Logging
# ------------------------------------------------------------------------------

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "interactions.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

def log_event(event_type: str, data: dict):
    """
    Append a JSON‐line record to the log file.
    Each record gets a timestamp and an event type.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event_type,
        **data
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Register a no-op at exit (we don’t need atexit saves now)
atexit.register(lambda: None)

# ------------------------------------------------------------------------------
#                                 Flask App
# ------------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = "super_secret_key"  # In production, use a real secret key

# Load Whisper model once
model = whisper.load_model("base")
OLLAMA_URL = "http://localhost:11434/api/chat"


# ------------------------------------------------------------------------------
#                              Helper Functions
# ------------------------------------------------------------------------------

def get_user_id():
    """
    Determine (or generate) a stable user_id.  Priority:
      1) JSON payload: request.json["user_id"]
      2) Query string: request.args["user_id"]
      3) Cookie: request.cookies["user_id"]
      4) Session: session["user_id"]
      5) Fallback: generate a new UUID via our chroma_util helper
    """
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


def chat_with_ollama(messages: list) -> str:
    """
    Send a non‐streaming request to Ollama. Raises on HTTP errors.
    """
    payload = {"model": "pinky", "messages": messages, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def extract_lexical_facts(text: str) -> dict:
    """
    Regex‐based extraction of common “facts” from a user’s utterance.
    Returns a dict like {"name": "Alice", "age": 30, ...}, or {} if none.
    """
    facts = {}

    # 1) Name: “my name is Alice” or “call me Bob”
    name_match = re.search(r"\b(?:my name is|call me)\s+([A-Z][a-z]+)\b", text)
    if name_match:
        facts["name"] = name_match.group(1)

    # 2) Age: “my age is 48”
    age_match = re.search(r"\bmy age is\s+(\d+)\b", text, re.IGNORECASE)
    if age_match:
        facts["age"] = int(age_match.group(1))

    # 3) Favorite color: “my favorite color is light blue”
    color_match = re.search(r"\bmy favorite color is\s+([A-Za-z\s]+)\b", text, re.IGNORECASE)
    if color_match:
        facts["favorite_color"] = color_match.group(1).strip()

    # 4) Favorite number: “my favorite number is 7”
    number_match = re.search(r"\bmy favorite number is\s+(\d+)\b", text, re.IGNORECASE)
    if number_match:
        facts["favorite_number"] = int(number_match.group(1))

    # 5) Vehicle: “I drive a Toyota Camry TRD”
    vehicle_match = re.search(r"\bi drive a\s+([A-Za-z\s]+)\b", text, re.IGNORECASE)
    if vehicle_match:
        facts["vehicle"] = vehicle_match.group(1).strip()

    return facts


# ------------------------------------------------------------------------------
#                                  Routes
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = get_user_id()
        save_user_name(user_id, request.form["username"])
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
            log_event("transcribe_error", {
                "error": str(e),
                "user_id": session.get("user_id"),
            })
            return jsonify({"error": f"Transcription failed: {e}"}), 500


@app.route("/init", methods=["GET"])
def init_chat():
    user_id = get_user_id()

    # 1) Ensure the user profile file exists
    user_profile = get_user_profile(user_id)
    save_profile_to_disk(user_id, user_profile)

    # 2) Figure out greeting
    name = get_saved_user_name(user_id)
    message = f"Welcome back, {name}!" if name else "Hi! What’s your name?"

    # 3) Log this “init” event
    log_event("init_chat", {
        "user_id": user_id,
        "greeting": message
    })

    response = jsonify({"message": message, "user_id": user_id})
    response.set_cookie("user_id", user_id, max_age=60 * 60 * 24 * 365 * 5)
    return response


@app.route("/profile")
def view_profile():
    user_id = session.get("user_id") or request.cookies.get("user_id")
    if not user_id:
        return jsonify({"error": "User not identified"}), 404

    profile = get_user_profile(user_id)
    log_event("view_profile", {"user_id": user_id, "profile": profile})
    return jsonify(profile)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint. Logs:
     - chat_request (user_input, user_id, timestamp)
     - retrieval attempts (“What is my X?”, “What do I drive?”, etc.)
     - LLM context & response & latency
     - lexical extraction results
     - errors (if any)
    """
    start_ts = datetime.utcnow()
    try:
        data = request.get_json() or {}
        user_input = data.get("message", "").strip()
        user_id = get_user_id()

        if not user_id:
            return jsonify({"response": "⚠️ User ID not found."}), 400

        # 1) Log the incoming request
        log_event("chat_request", {
            "user_id": user_id,
            "user_input": user_input
        })

        user_profile = get_user_profile(user_id)
        user_name = user_profile.get("facts", {}).get("name")

        # -----------------------------
        #  2) Handle retrieval queries
        # -----------------------------
        # “What is my X?”
        m = re.search(r"\bwhat is my ([a-z ]+)\??", user_input, re.IGNORECASE)
        if m:
            key_raw = m.group(1).strip().lower()
            key = key_raw.replace(" ", "_")
            value = user_profile.get("facts", {}).get(key)

            was_found = value is not None
            response_text = (
                f"Your {key_raw} is {value}."
                if was_found
                else f"I don't know your {key_raw} yet."
            )

            # Log retrieval event
            log_event("fact_retrieval", {
                "user_id": user_id,
                "requested_key": key,
                "value_found": value if was_found else None,
                "was_found": was_found
            })

            return jsonify({"response": response_text, "lexical_facts": {}})

        # “What do I drive?”
        if re.search(r"\bwhat do I drive\??", user_input, re.IGNORECASE):
            vehicle = user_profile.get("facts", {}).get("vehicle")
            was_found = bool(vehicle)
            response_text = (
                f"You drive a {vehicle}."
                if was_found
                else "I don't know what you drive yet."
            )
            log_event("fact_retrieval", {
                "user_id": user_id,
                "requested_key": "vehicle",
                "value_found": vehicle if was_found else None,
                "was_found": was_found
            })
            return jsonify({"response": response_text, "lexical_facts": {}})

        # “What do you know about me?” / “Tell me some facts about myself”
        if re.search(r"\b(?:what do you know about me|tell me some facts (?:about )?myself)\b", user_input, re.IGNORECASE):
            facts = user_profile.get("facts", {})
            if facts:
                summary = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in facts.items())
                response_text = f"Here’s what I know about you: {summary}."
            else:
                response_text = "I don't know anything about you yet."
            log_event("profile_summary", {
                "user_id": user_id,
                "profile_facts": facts
            })
            return jsonify({"response": response_text, "lexical_facts": {}})

        # ---------------------------------
        #  3) Add user utterance to vector memory
        # ---------------------------------
        add_user_memory(user_id, user_input, metadata={"role": "user"})
        save_message_to_chroma(user_id, "user", user_input)

        # ---------------------------------
        #  4) If no name yet, treat this as name
        # ---------------------------------
        if not user_name:
            user_profile["facts"]["name"] = user_input
            save_profile_to_disk(user_id, user_profile)

            log_event("fact_extraction", {
                "user_id": user_id,
                "user_input": user_input,
                "extracted": {"name": user_input},
                "profile_before": {},  # blank
                "profile_after": user_profile["facts"]
            })

            return jsonify({"response": f"Nice to meet you, {user_input}!", "lexical_facts": {}})

        # ---------------------------------
        #  5) Normal LLM chat flow
        # ---------------------------------
        #   a) Retrieve vector‐memory context
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

        #   b) Call the LLM
        llm_start = datetime.utcnow()
        try:
            response_text = chat_with_ollama(messages)
        except Exception as e:
            # Log LLM error
            log_event("llm_error", {
                "user_id": user_id,
                "error": str(e),
                "messages": messages
            })
            raise

        llm_end = datetime.utcnow()
        latency_ms = int((llm_end - llm_start).total_seconds() * 1000)

        #   c) Log the LLM call + latency
        log_event("llm_call", {
            "user_id": user_id,
            "messages_sent": messages,
            "response_text": response_text,
            "latency_ms": latency_ms
        })

        #   d) Save assistant reply to vector memory
        add_user_memory(user_id, response_text, metadata={"role": "assistant"})
        save_message_to_chroma(user_id, "assistant", response_text)

        if not response_text:
            return jsonify({"response": "⚠️ Empty or malformed LLM response.", "lexical_facts": {}})

        # ---------------------------------
        #  6) Extract lexical facts from user input
        # ---------------------------------
        lexical_data = extract_lexical_facts(user_input)
        print(f"[chat] Extracted lexical facts: {lexical_data}")

        #   a) Log the extraction attempt
        log_event("fact_extraction", {
            "user_id": user_id,
            "user_input": user_input,
            "extracted": lexical_data,
            "profile_before": user_profile.get("facts", {}).copy()
        })

        #   b) Update user profile if new facts found
        if lexical_data:
            for key, value in lexical_data.items():
                if value is not None and value != "":
                    update_user_fact(user_id, key, value)

            # Log profile after update
            updated_profile = get_user_profile(user_id).get("facts", {}).copy()
            log_event("profile_update", {
                "user_id": user_id,
                "new_facts": lexical_data,
                "profile_after": updated_profile
            })

        # ---------------------------------
        #  7) Return to front‐end
        # ---------------------------------
        return jsonify({"response": response_text, "lexical_facts": lexical_data})

    except requests.ConnectionError:
        log_event("chat_error", {
            "user_id": session.get("user_id"),
            "error": "Could not connect to Ollama"
        })
        return jsonify({"response": "⚠️ Could not connect to Ollama."}), 503

    except requests.HTTPError as e:
        log_event("chat_error", {
            "user_id": session.get("user_id"),
            "error": f"Ollama HTTP error: {e}"
        })
        return jsonify({"response": f"⚠️ Ollama HTTP error: {e}", "lexical_facts": {}}), 500

    except Exception as e:
        log_event("chat_error", {
            "user_id": session.get("user_id"),
            "error": f"Unexpected error: {e}"
        })
        return jsonify({"response": f"⚠️ Unexpected error: {e}", "lexical_facts": {}}), 500


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


# ------------------------------------------------------------------------------
#                                 Run App
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
