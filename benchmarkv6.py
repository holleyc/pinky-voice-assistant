import os
import time
import sounddevice as sd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from chromadb import PersistentClient
import whisper
import hashlib
import re
import unicodedata
import torch
import csv
from datetime import datetime
import pandas as pd
import json
import soundfile as sf
import librosa

# ---------- CONFIG ----------
WHISPER_MODEL_NAME = "tiny"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
TTS_MODEL_NAME = "tts_models/en/vctk/vits"
OLLAMA_URL = "http://localhost:11434/api/generate"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "Pinkys_Brain"

USE_GPU = torch.cuda.is_available()
device = "cuda" if USE_GPU else "cpu"

embedding_time = None
embedding = None
add_time = None
query_time = None
llm_time = None
tts_time = None
audio_duration_estimate = 1.0  # Avoid division by zero
whisper_time = None
whisper_text = ""

# ---------- INIT MODELS ----------
os.environ["TTS_DEVICE"] = device
from TTS.api import TTS
tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=False)
tts_sample_rate = tts.synthesizer.output_sample_rate

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
client = PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)
whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=device)

import re
import unicodedata

def clean_text(text):
    # Normalize unicode (e.g., fancy quotes to standard quotes)
    text = unicodedata.normalize("NFKC", text)

    # Remove unwanted characters but keep useful punctuation
    # Allowed: alphanumeric, whitespace, common punctuation
    text = re.sub(r"[^\w\s.,!?;:'\"()\-]", '', text)

    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def parse_ollama_stream(stream_output):
    parts = []
    for line in stream_output.splitlines():
        try:
            chunk = json.loads(line)
            if 'response' in chunk:
                parts.append(chunk['response'])
        except json.JSONDecodeError:
            continue
    return ''.join(parts).strip()

def ask_ollama(context, query, model="pinky"):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": True},
            timeout=60,
            stream=True
        )
        response.raise_for_status()
        stream_output = ""
        for chunk in response.iter_lines():
            if chunk:
                stream_output += chunk.decode("utf-8") + "\n"
        return parse_ollama_stream(stream_output)
    except requests.exceptions.RequestException as e:
        print(f"LLM error: {e}")
        return "[Error connecting to local model]"


def run_tts(text):
    global tts_time
    print("\n--- TTS Benchmark (VITS) ---")
    cleaned = clean_text(text)
    try:
        start = time.time()
        segments = re.split(r'(?<=[.?!])\s+', cleaned)
        total_audio = []

        for segment in segments:
            if segment.strip():
                audio = tts.tts(segment.strip(), speaker="p280", speed=0.70)
                audio = audio / np.max(np.abs(audio))  # normalize
                sd.play(audio, samplerate=tts_sample_rate)
                sd.wait()
                time.sleep(0.3)  # breathing pause
        tts_time = time.time() - start
        print(f"Synthesis Time: {tts_time:.2f}s")

    except Exception as e:
        print("TTS synthesis failed:", str(e))

def benchmark_whisper(audio):
    global whisper_time, whisper_text
    print("\n--- Whisper Benchmark ---")
    start = time.time()
    result = whisper_model.transcribe(audio, fp16=(device == "cuda"))
    whisper_time = time.time() - start
    whisper_text = result.get("text", "[No transcript]")
    print(f"Transcription Time: {whisper_time:.2f}s")
    print(f"Transcription: {whisper_text}")

def benchmark_embedding(text):
    global embedding_time, embedding
    print("\n--- Embedding Benchmark ---")
    start = time.time()
    embedding = embedder.encode([text])[0]
    embedding_time = time.time() - start
    print(f"Embedding Time: {embedding_time:.4f}s")
    print(f"Vector Length: {len(embedding)}")

def benchmark_memory(text):
    global add_time, query_time
    print("\n--- ChromaDB Benchmark ---")
    embedding_local = embedder.encode([text])[0]
    start_add = time.time()
    collection.add(documents=[text], embeddings=[embedding_local.tolist()], ids=[str(time.time())])
    add_time = time.time() - start_add
    start_query = time.time()
    result = collection.query(query_embeddings=[embedding_local.tolist()], n_results=1)
    query_time = time.time() - start_query
    print(f"Add Time: {add_time:.2f}s, Query Time: {query_time:.2f}s")
    print("Top Match:", result['documents'][0][0] if result['documents'] else "[None]")

def benchmark_ollama(query):
    global llm_time
    print("\n--- LLM Benchmark ---")
    start = time.time()
    response = ask_ollama("", query)
    llm_time = time.time() - start
    print(f"LLM Response Time: {llm_time:.2f}s")
    print("Response:", response)

def log_benchmark_to_csv(data, filename="benchmark_results.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def compare_benchmarks(filename="benchmark_results.csv", num_recent=5):
    if not os.path.isfile(filename):
        print("No benchmark file found to compare.")
        return
    df = pd.read_csv(filename, parse_dates=["timestamp"])
    if df.empty:
        print("Benchmark log is empty.")
        return
    print("\n--- Benchmark Comparison ---")
    recent = df.tail(num_recent)
    for col in df.columns:
        if col == "timestamp" or df[col].dtype == object:
            continue
        values = recent[col].dropna()
        if not values.empty:
            print(f"{col}: avg={values.mean():.3f}, min={values.min():.3f}, max={values.max():.3f}, latest={values.iloc[-1]:.3f}")

def benchmark_end_to_end(text):
    global tts_time
    print("\n--- End-to-End Benchmark ---")
    start_context = time.time()
    embedding_local = embedder.encode([text])[0]
    results = collection.query(query_embeddings=[embedding_local.tolist()], n_results=1)
    context = "\n".join(results["documents"][0]) if results["documents"] else ""
    context_time = time.time() - start_context
    start_llm = time.time()
    response = ask_ollama(context, text)
    llm_time = time.time() - start_llm
    start_tts = time.time()
    run_tts(response)
    tts_time = time.time() - start_tts
    print(f"Memory: {context_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s")

# ---------- CLI ----------
if __name__ == "__main__":
    sample_text = "Who are you?"
    benchmark_embedding(sample_text)
    benchmark_memory(sample_text)
    benchmark_ollama(sample_text)
    run_tts(sample_text)
    benchmark_end_to_end(sample_text)
    fake_audio = np.random.rand(16000 * 5).astype(np.float32)
    benchmark_whisper(fake_audio)
    benchmark_data = {
        "timestamp": datetime.now().isoformat(),
        "embedding_time": embedding_time,
        "vector_length": len(embedding) if embedding is not None else 0,
        "chroma_add_time": add_time,
        "chroma_query_time": query_time,
        "llm_response_time": llm_time,
        "tts_synthesis_time": tts_time,
        "tts_real_time_factor": tts_time / audio_duration_estimate if tts_time else None,
        "whisper_transcription_time": whisper_time,
        "whisper_text": whisper_text[:100],
    }
    log_benchmark_to_csv(benchmark_data)
    print("\n--- Benchmark data logged to benchmark_results.csv ---")
    compare_benchmarks()
