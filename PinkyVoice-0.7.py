# Full script with integrated persistent rotating error logging and context
import os
import json
import time
import torch
import requests
import threading
import numpy as np
import sounddevice as sd
import unicodedata
import re
import hashlib
import logging
import spacy
from logging.handlers import RotatingFileHandler
from pydub import AudioSegment
from pydub.playback import play as pydub_play
from TTS.api import TTS
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# ---------- LOGGING SETUP ----------
LOG_FILE = "assistant_error.log"
logger = logging.getLogger("assistant_logger")
logger.setLevel(logging.ERROR)

handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_error_with_context(exception, user_input=None, output=None):
    logger.error(
        "Exception occurred:\n%s\nUser Input: %s\nOutput: %s",
        str(exception),
        json.dumps(user_input, indent=2) if user_input else "N/A",
        json.dumps(output, indent=2) if output else "N/A"
    )

# ---------- CONFIGURATION ----------
WHISPER_MODEL_NAME = "small"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TTS_MODEL_NAME = "tts_models/en/vctk/vits"
OLLAMA_URL = "http://localhost:11434/api/generate"
COLLECTION_NAME = "Pinkys_Brain"
CHROMA_PERSIST_DIR = "./chroma_db"
RECORD_DURATION = 5
DEFAULT_SPEAKER = "p294"
USER_PROFILES_DIR = "./user_profiles"
nlp = spacy.load("en_core_web_sm")

USE_GPU = torch.cuda.is_available()
device = "cuda" if USE_GPU else "cpu"
audio_lock = threading.Lock()
tts_cache = {}
conversation_history = []
MAX_HISTORY_LENGTH = 6

os.environ["TTS_DEVICE"] = device

print("Loading TTS model...")
if "tts" not in globals():
    tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=False)
tts_sample_rate = tts.synthesizer.output_sample_rate

try:
    init_audio = tts.tts("Initializing voice.", speaker=DEFAULT_SPEAKER)
    sd.play(init_audio, samplerate=tts_sample_rate)
    sd.wait()
except Exception as e:
    log_error_with_context(e, user_input="Initializing voice")

if USE_GPU:
    torch.cuda.empty_cache()

print("Initializing ChromaDB...")
client = PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

print("Loading embedder...")
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
except Exception as e:
    log_error_with_context(e)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

print("Loading Faster-Whisper model...")
try:
    whisper_model = WhisperModel(
        WHISPER_MODEL_NAME,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )
except Exception as e:
    log_error_with_context(e)
    exit(1)

def record_audio(duration=RECORD_DURATION, sample_rate=16000):
    print("🎙️ Listening...")
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        return np.squeeze(audio)
    except Exception as e:
        log_error_with_context(e, user_input="record_audio")
        return None

def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", "", text)
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
        log_error_with_context(e, user_input=text)
        return None

def split_sentences(text, max_length=200):
    sentences = re.split(r'[.!?]', text)
    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > max_length:
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
            sd.wait()
        except Exception as e:
            log_error_with_context(e, user_input=text)
            try:
                cleaned = clean_text(text)
                tmp_path = "tmp_fallback.wav"
                tts.tts_to_file(cleaned, tmp_path)
                fallback_audio = AudioSegment.from_wav(tmp_path)
                pydub_play(fallback_audio)
            except Exception as e2:
                log_error_with_context(e2, user_input=text)

def recognize_speech(audio):
    print("🧠 Transcribing...")
    normalized = audio / np.max(np.abs(audio))
    segments, _ = whisper_model.transcribe(normalized, beam_size=5)
    full_text = " ".join(segment.text for segment in segments)
    return full_text.strip()

def identify_user():
    print("👤 Who are you?")
    audio = record_audio()
    raw_name = recognize_speech(audio).strip().lower()
    print(f"👋 Welcome, {raw_name}!")
    name_match = re.search(r"(?:my name is|i am|it's|this is)?\s*([a-zA-Z0-9_-]+)", raw_name)
    name = name_match.group(1) if name_match else re.sub(r'\W+', '_', raw_name)
    name = name[:100]
    print(f"✅ Sanitized user ID: {name}")
    return name

def sanitize_collection_name(name):
    name = name.lower()
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    name = name.strip("._-")
    return name[:100] or "default_user"

def load_user_profile(user_id):
    os.makedirs(USER_PROFILES_DIR, exist_ok=True)
    path = os.path.join(USER_PROFILES_DIR, f"{user_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    profile = {"name": user_id, "preferences": {}, "collection_name": f"profile_{user_id}"}
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    return profile

def update_user_profile(profile):
    path = os.path.join(USER_PROFILES_DIR, f"{profile['name']}.json")
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)

user_id = identify_user()
safe_user_id = sanitize_collection_name(user_id)
user_profile = load_user_profile(user_id)
user_collection_name = user_profile.get("collection_name", f"profile_{safe_user_id}")
collection = client.get_or_create_collection(user_collection_name)

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
        log_error_with_context(e, user_input=query)
        return "Sorry, I couldn't connect to the local model."

def search_memory(query, top_k=3):
    embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    if results["documents"]:
        context = "\n".join(results["documents"][0])
        print("🔍 Retrieved memory context:\n", context)
        return context
    return ""

def lexical_features(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    keywords = list(set(tokens))
    return {
        "tokens": tokens,
        "entities": entities,
        "keywords": keywords
    }

def save_memory(entry):
    clean_entry = clean_text(entry)
    if not clean_entry.strip():
        return
    
    # Perform lexical analysis
    features = lexical_features(clean_entry)

    # Optionally include keywords in the embedding process
    embedding_input = clean_entry + " " + " ".join(features["keywords"])
    embedding = embedder.encode([clean_entry])[0].tolist()
    doc_id = str(time.time())
    try:
        collection.add(
            documents=[clean_entry], 
            embeddings=[embedding], 
            ids=[doc_id],
            metadatas=[features]  # Save lexical metadata with each document
        )
    except Exception as e:
        log_error_with_context(e, user_input=clean_entry)

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
        if "my favorite color is" in query.lower():
            color = query.split("my favorite color is")[-1].strip()
            user_profile["preferences"]["favorite_color"] = color
            update_user_profile(user_profile)
            play_audio(f"Okay {user_profile['name']}, I will remember that your favorite color is {color}.")
            continue
        context = search_memory(query)
        response = ask_ollama(context, query)
        print(f"🤖 Assistant: {response}")
        conversation_history.append((query, response))
        if len(conversation_history) > MAX_HISTORY_LENGTH:
            conversation_history.pop(0)
        save_memory(f"{query} {response}")
        play_audio(response)
        time.sleep(1)

if __name__ == "__main__":
    assistant_loop()
