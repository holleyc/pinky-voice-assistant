#!/usr/bin/env python3

from flask import (
    Flask, request, jsonify, render_template,
    session, make_response, send_from_directory
)
import requests, tempfile, whisper

from utils.qr_utils import generate_qr_base64
from utils.chroma_utils import (
    get_or_create_user_id, save_user_name, get_saved_user_name,
    save_message_to_chroma, get_relevant_context, get_chat_history
)

# Initialize
app = Flask(__name__)
app.secret_key = "super_secret_key"  # Use env variable in production
model = whisper.load_model("base")
OLLAMA_URL = "http://localhost:11434/api/chat"

# ---------- Helper Functions ----------
def get_user_id():
    return session.setdefault("user_id", get_or_create_user_id(request))

def chat_with_ollama(messages):
    payload = {"model": "pinky", "messages": messages, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")

# ---------- Routes ----------
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

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    user_id = get_user_id()
    user_name = get_saved_user_name(user_id)

    save_message_to_chroma(user_id, "user", user_input)

    if not user_name:
        save_user_name(user_id, user_input)
        return jsonify({"response": f"Nice to meet you, {user_input}!"})

    context = get_relevant_context(user_id, user_input)
    messages = context + [{"role": "user", "content": user_input}]

    try:
        response = chat_with_ollama(messages)
        if not response:
            return jsonify({"response": "⚠️ Empty or malformed LLM response."})
        save_message_to_chroma(user_id, "assistant", response)
        return jsonify({"response": response})
    except requests.ConnectionError:
        return jsonify({"response": "⚠️ Could not connect to Ollama."})
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

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
