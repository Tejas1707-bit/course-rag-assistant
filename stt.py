import whisper
import json

# Load Whisper model
model = whisper.load_model("turbo")

# Transcribe audio
result = model.transcribe(
    "audios/02_B.mp3",
    task="translate",
    word_timestamps=True
)

chunks = []

# Create chunks
for segment in result["segments"]:

    text = segment["text"].strip()

    # Skip empty segments
    if not text:
        continue

    chunks.append({
        "start": round(segment["start"], 2),
        "end": round(segment["end"], 2),
        "text": text
    })

# Save chunks as JSON
with open("chunks.json", "w", encoding="utf-8") as json_file:
    json.dump(chunks, json_file, indent=4, ensure_ascii=False)

print(f"✅ {len(chunks)} chunks saved successfully to chunks.json")