import os
import json
import time
import torch
import whisper
import requests
import threading
import numpy as np
import sounddevice as sd
import unicodedata
import re
import time
from collections import defaultdict
import hashlib
from pydub import AudioSegment
from pydub.playback import play as pydub_play
from TTS.api import TTS
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient  # ✅ Use persistent ChromaDB


# ---------- CONFIGURATION ----------
WHISPER_MODEL_NAME = "medium"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TTS_MODEL_NAME = "tts_models/en/vctk/vits"
OLLAMA_URL = "http://localhost:11434/api/generate"
COLLECTION_NAME = "Pinkys_Brain"  # ✅ Shared persistent memory
CHROMA_PERSIST_DIR = "./chroma_db"
RECORD_DURATION = 5
DEFAULT_SPEAKER = "p294"

USER_PROFILES_DIR = "./user_profiles"

USE_GPU = torch.cuda.is_available()
device = "cuda" if USE_GPU else "cpu"

audio_lock = threading.Lock()
tts_cache = {}

# ---------- CONVERSATION MEMORY ----------
conversation_history = []  # 🧠 In-memory multi-turn context
MAX_HISTORY_LENGTH = 6     # Keep last 3 user-assistant pairs


# ---------- INIT TTS ----------
os.environ["TTS_DEVICE"] = device  # ✅ Set this BEFORE initializing TTS

print("Loading TTS model...")


# ---------- INIT MODELS ----------
#os.environ["TTS_DEVICE"] = device
from TTS.api import TTS
tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=False)
tts_sample_rate = tts.synthesizer.output_sample_rate

#tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=False)
#TTS_SAMPLE_RATE = tts.synthesizer.output_sample_rate

try:
    init_audio = tts.tts("Initializing voice.", speaker=DEFAULT_SPEAKER)
    sd.play(init_audio, samplerate=tts_sample_rate)
    sd.wait()
except Exception as e:
    print(f"TTS initialization error: {e}")


# ---------- INIT MODELS ----------
if USE_GPU:
    torch.cuda.empty_cache()

print("Initializing ChromaDB...")
client = PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)  # Default init, replaced later

print("Loading embedder...")
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
except Exception as e:
    print(f"Embedder error, falling back to CPU: {e}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

print("Loading Whisper...")
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
    print(f"Using Whisper device: {device}")

except Exception as e:
    print(f"Whisper fallback to CPU: {e}")
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device="cpu")
    device = "cpu"
    print(f"Using Whisper device: {device}")


# ---------- AUDIO FUNCTIONS ----------
def record_audio(duration=RECORD_DURATION, sample_rate=16000):
    print("🎙️ Listening...")
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        return np.squeeze(audio)
    except Exception as e:
        print(f"Audio recording error: {e}")
        return None

def clean_text(text):
    # Normalize and remove combining characters
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Encode to ASCII (strip emojis, unusual phonemes, etc.)
    text = text.encode("ascii", errors="ignore").decode()

    # Optional: aggressively filter only safe characters
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()

def get_tts_audio(text):
    cleaned = clean_text(text)
    key = hash_text(cleaned)
    if key in tts_cache:
        return tts_cache[key]
    try:
        audio = tts.tts(cleaned, speaker="p280", speed=0.70)
        tts_cache[key] = audio
        return audio
    except Exception as e:
        print(f"TTS synthesis error: {e}")
        return None

def split_sentences(text, max_length=200):
    sentences = re.split(r'[.!?]', text)
    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > max_length:
            # Hard split long sentences
            for i in range(0, len(sentence), max_length):
                chunks.append(sentence[i:i+max_length])
        elif sentence:
            chunks.append(sentence)
    return chunks


def play_audio(text):
    with audio_lock:
        try:
            print("🔊 Speaking...")
            audio = get_tts_audio(text)
            if audio is None:
                return
            sd.play(audio, samplerate=tts_sample_rate)
            sd.wait()  # Wait for playback to finish
        except Exception as e:
            print(f"TTS playback error: {e}")
            # Fallback with pydub
            try:
                cleaned = clean_text(text)
                tmp_path = "tmp_fallback.wav"
                tts.tts_to_file(cleaned, tmp_path)
                fallback_audio = AudioSegment.from_wav(tmp_path)
                pydub_play(fallback_audio)
            except Exception as e2:
                print(f"Fallback playback failed: {e2}")



# ---------- SPEECH RECOGNITION ----------
def recognize_speech(audio):
    print("🧠 Transcribing...")
    normalized = audio / np.max(np.abs(audio))
    result = whisper_model.transcribe(normalized, fp16=(device == "cuda"))
    return result.get("text", "").strip()

# ---------- LLM COMMUNICATION ----------
def identify_user():
    print("👤 Who are you?")
    audio = record_audio()
    raw_name = recognize_speech(audio).strip().lower()
    print(f"👋 Welcome, {raw_name}!")

    # Extract only the name from common patterns
    name_match = re.search(r"(?:my name is|i am|it's|this is)?\s*([a-zA-Z0-9_-]+)", raw_name)
    if name_match:
        name = name_match.group(1)
    else:
        name = re.sub(r'\W+', '_', raw_name)  # fallback: replace invalid chars

    name = name[:100]  # limit length for safety
    print(f"✅ Sanitized user ID: {name}")

    return name


def sanitize_collection_name(name):
    # Replace spaces and apostrophes with underscores, keep only valid characters
    name = name.lower()
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    name = name.strip("._-")  # Make sure it starts and ends with alphanumeric
    name = name[:100]  # Avoid overly long names
    return name or "default_user"

def load_user_profile(user_id):
    os.makedirs(USER_PROFILES_DIR, exist_ok=True)
    profile_path = os.path.join(USER_PROFILES_DIR, f"{user_id}.json")

    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            return json.load(f)
    else:
        profile = {
            "name": user_id,
            "preferences": {},
            "collection_name": f"profile_{user_id}"
        }
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)
        return profile



def update_user_profile(profile):
    path = os.path.join(USER_PROFILES_DIR, f"{profile['name']}.json")
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)

# Replace initial collection init with dynamic user profile logic
user_id = identify_user()
safe_user_id = sanitize_collection_name(user_id)
user_profile = load_user_profile(user_id)

user_collection_name = user_profile.get("collection_name", f"profile_{safe_user_id}")
collection = client.get_or_create_collection(user_collection_name)

# Modify ask_ollama to include user profile context
def ask_ollama(context, query, model="pinky"):
    history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in conversation_history])
    profile_str = f"User: {user_profile['name']}\nPreferences: {user_profile['preferences']}\n"
    prompt = f"{profile_str}Context:\n{context}\n\n{history_text}\nUser: {query}\nAssistant:"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("response", "[No response key found]")
    except requests.exceptions.RequestException as e:
        print(f"LLM error: {e}")
        return "Sorry, I couldn't connect to the local model."



# ---------- MEMORY ----------
def search_memory(query, top_k=3):
    embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    if results["documents"]:
        return "\n".join(results["documents"][0])
    return ""

def save_memory(entry):
    clean_entry = clean_text(entry)
    if not clean_entry.strip():
        return
    embedding = embedder.encode([clean_entry])[0].tolist()
    doc_id = str(time.time())
    try:
        collection.add(documents=[clean_entry], embeddings=[embedding], ids=[doc_id])
    except Exception as e:
        print(f"Error saving memory: {e}")


# ---------- MAIN LOOP ----------
def assistant_loop():
    print("🧠 Voice Assistant is active. Say 'exit' to quit.\n")
    while True:
        audio = record_audio()
        if audio is None:
            continue

        query = recognize_speech(audio)
        if not query:
            continue

        print(f"👤 You said: {query}")
        if "exit" in query.lower():
            print("👋 Exiting...")
            break

        if "clear memory" in query.lower() or "new topic" in query.lower():
            print("🧹 Clearing conversation history...")
            conversation_history.clear()
            play_audio("Okay, let's start fresh.")
            continue

        # Optional: Handle profile updates through query
        if "my favorite color is" in query.lower():
            color = query.split("my favorite color is")[-1].strip()
            user_profile["preferences"]["favorite_color"] = color
            update_user_profile(user_profile)
            play_audio(f"Okay {user_profile['name']}, I will remember that your favorite color is {color}.")
            continue

        context = search_memory(query)
        response = ask_ollama(context, query)
        print(f"🤖 Assistant: {response}")

        # Multi-turn memory storage
        conversation_history.append((query, response))
        if len(conversation_history) > MAX_HISTORY_LENGTH:
            conversation_history.pop(0)

        save_memory(f"{query} {response}")
        play_audio(response)

        time.sleep(1)

# ---------- ENTRY ----------
if __name__ == "__main__":
    assistant_loop()
