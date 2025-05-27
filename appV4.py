#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template, session, make_response
import requests
from utils.qr_utils import generate_qr_base64
from utils.chroma_utils import (
    get_or_create_user_id,
    save_user_name,
    get_saved_user_name,
    save_message_to_chroma,
    get_relevant_context,
    get_chat_history
)

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Use env variable in production

OLLAMA_URL = "http://localhost:11434/api/chat"

# Routes
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/init", methods=["GET"])
def init_chat():
    user_id = get_or_create_user_id(request)
    session["user_id"] = user_id

    user_name = get_saved_user_name(user_id)
    message = f"Welcome back, {user_name}!" if user_name else "Hi! What’s your name?"

    response = jsonify({"message": message})
    response.set_cookie("user_id", user_id, max_age=60 * 60 * 24 * 365 * 5)
    return response

@app.route("/ollama_healthcheck", methods=["GET"])
def ollama_healthcheck():
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "pinky",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False
        })
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        return jsonify({
            "status": "success",
            "message": content or "Ollama responded, but with no content."
        })
    except requests.exceptions.ConnectionError:
        return jsonify({
            "status": "error",
            "message": "❌ Could not connect to Ollama. Is it running on localhost:11434?"
        }), 503
    except requests.exceptions.HTTPError as http_err:
        return jsonify({
            "status": "error",
            "message": f"❌ Ollama HTTP error: {http_err}"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"❌ Unexpected error: {str(e)}"
        }), 500


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    user_id = session.setdefault("user_id", get_or_create_user_id(request))
    user_name = get_saved_user_name(user_id)

    if not user_name:
        save_message_to_chroma(user_id, "user", user_input)
        save_user_name(user_id, user_input)
        return jsonify({"response": f"Nice to meet you, {user_input}!"})

    save_message_to_chroma(user_id, "user", user_input)
    context = get_relevant_context(user_id, user_input)
    messages = context + [{"role": "user", "content": user_input}]

    try:
        payload = {
            "model": "pinky",
            "messages": messages,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        json_data = response.json()
        assistant_msg = json_data.get("message", {}).get("content", "")

        if not assistant_msg:
            print("Ollama response missing 'message.content':", json_data)
            return jsonify({
                "response": "⚠️ LLM response was empty or malformed. Check server logs."
            })

        save_message_to_chroma(user_id, "assistant", assistant_msg)
        return jsonify({"response": assistant_msg})

    except requests.exceptions.ConnectionError:
        print("❌ Ollama server is unreachable at", OLLAMA_URL)
        return jsonify({
            "response": "⚠️ Could not connect to Ollama. Is it running at localhost:11434?"
        })
    except requests.exceptions.HTTPError as http_err:
        print("❌ HTTP error from Ollama:", http_err)
        return jsonify({
            "response": f"⚠️ Ollama returned an HTTP error: {http_err}"
        })
    except Exception as e:
        print("❌ Unexpected error:", e)
        return jsonify({
            "response": f"⚠️ Unexpected error occurred: {str(e)}"
        })



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        user_id = session.setdefault("user_id", get_or_create_user_id(request))
        save_user_name(user_id, username)
        return render_template("chat.html")
    return render_template("login.html")


@app.route("/qr")
def qr_pair():
    user_id = session.setdefault("user_id", get_or_create_user_id(request))
    img_str = generate_qr_base64(user_id)
    return f"<h3>Scan to continue chat on mobile</h3><img src='data:image/png;base64,{img_str}'>"


@app.route("/pair", methods=["GET"])
def pair():
    user_id = session.setdefault("user_id", get_or_create_user_id(request))
    img_str = generate_qr_base64(user_id)
    return jsonify({"uuid": user_id, "qr_image_base64": img_str})


@app.route("/history")
def history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "No user session found."}), 400

    chat_log = get_chat_history(user_id)
    return render_template("history.html", chat_log=chat_log)


@app.route("/whoami", methods=["GET"])
def whoami():
    user_id = session.get("user_id") or request.cookies.get("user_id")
    if not user_id:
        return jsonify({"error": "User not identified"}), 404

    session["user_id"] = user_id
    return jsonify({
        "user_id": user_id,
        "user_name": get_saved_user_name(user_id)
    })


@app.route("/reset", methods=["GET"])
def reset():
    session.clear()
    return jsonify({"message": "Session reset."})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
