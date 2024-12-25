import whisper

# Load the Whisper model
model = whisper.load_model("medium")  # Replace "base" with other models like "small", "medium", "large" as needed

# Path to the MP3 file
audio_file = "test.mp3"  # Replace with the path to your MP3 file

# Transcribe the MP3 file
print("Transcribing audio...")
result = model.transcribe(audio_file)

# Output the text
print("Transcription:")
print(result["text"])
