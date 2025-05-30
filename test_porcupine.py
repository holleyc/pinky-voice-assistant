import pvporcupine
import sounddevice as sd
import numpy as np

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    pcm = indata[:, 0].astype(np.int16)
    result = porcupine.process(pcm)
    if result >= 0:
        print("Wake word detected!")

# Create Porcupine wake word engine with built-in keyword "picovoice"
porcupine = pvporcupine.create(keywords=["picovoice"])

try:
    with sd.InputStream(channels=1, samplerate=porcupine.sample_rate, callback=audio_callback):
        print("Listening for wake word 'picovoice'... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    porcupine.delete()
