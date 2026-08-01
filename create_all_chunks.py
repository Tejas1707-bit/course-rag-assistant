import whisper
import json
import os

# Load Whisper model
model = whisper.load_model("turbo")   # or "large-v2"

# Create output folder
os.makedirs("jsons", exist_ok=True)

# Read all audio files
audios = sorted(os.listdir("audios"))

for audio in audios:

    if "_" not in audio or not audio.endswith(".mp3"):
        continue

    number = audio.split("_")[0]
    title = audio.split("_")[1].replace(".mp3", "")

    print(f"Processing: {audio}")

    result = model.transcribe(
        audio=f"audios/{audio}",
        task="translate",
        word_timestamps=True
    )

    chunks = []

    for segment in result["segments"]:

        text = segment["text"].strip()

        if not text:
            continue

        chunks.append({
            "number": number,
            "title": title,
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "text": text
        })

    chunks_with_metadata = {
        "number": number,
        "title": title,
        "full_text": result["text"],
        "chunks": chunks
    }

    output_file = f"jsons/{number}_{title}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            chunks_with_metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved -> {output_file}")

print("\n✅ All audio files processed successfully.")