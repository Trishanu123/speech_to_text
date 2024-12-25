import sounddevice as sd
import numpy as np
import whisper
import queue

# Load the Whisper model
model = whisper.load_model("medium")  # Choose "base", "small", "medium", or "large"

# Parameters for audio capture
SAMPLE_RATE = 16000  # Whisper works best with 16kHz audio
DURATION = 5  # Duration of each recording in seconds
CHANNELS = 1  # Mono audio
audio_queue = queue.Queue()  # Queue to hold audio data

def record_audio():
    """Records audio and adds it to the queue."""
    print("Recording audio...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
    sd.wait()  # Wait for the recording to finish
    audio_queue.put(audio_data)
    print("Recording completed!")

def transcribe_audio():
    """Fetches audio from the queue and transcribes it."""
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            # Flatten the audio data and normalize
            audio_data = np.squeeze(audio_data)
            audio_data = (audio_data / np.max(np.abs(audio_data))).astype(np.float32)
            print("Transcribing...")
            result = model.transcribe(audio_data, fp16=False)
            print("You said:", result["text"])

try:
    print("Press Ctrl+C to stop the program.")
    while True:
        record_audio()  # Record audio
        transcribe_audio()  # Transcribe the recorded audio
except KeyboardInterrupt:
    print("\nProgram stopped.")
