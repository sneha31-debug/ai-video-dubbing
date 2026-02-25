"""
Step 2: Transcribe the extracted audio using OpenAI Whisper.

Outputs a JSON file with the transcript text and per-segment timestamps.
These timestamps are used to align the Hindi audio during lip sync.

Usage:
    python scripts/02_transcribe.py
"""

import whisper
import json
import os
import argparse
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def transcribe(audio_path: str, model_size: str, language: str, device: str) -> dict:
    print(f"Loading Whisper model: {model_size} (device: {device})")
    model = whisper.load_model(model_size, device=device)

    print(f"Transcribing: {audio_path}")
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,   # word-level timestamps for precise sync
        verbose=False
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Step 2: Whisper transcription")
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    audio_path = args.audio or os.path.join(cfg["output"]["audio_dir"], "clip_audio.wav")
    output_dir = cfg["output"]["transcripts_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}. Run 01_extract_clip.py first.")

    result = transcribe(
        audio_path=audio_path,
        model_size=cfg["whisper"]["model_size"],
        language=cfg["whisper"]["language"],
        device=cfg["whisper"]["device"]
    )

    # Save full result (text + segments + word timestamps)
    transcript_path = os.path.join(output_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Save plain text for quick reference
    text_path = os.path.join(output_dir, "transcript.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())

    print("\nStep 2 complete.")
    print(f"  Transcript JSON : {transcript_path}")
    print(f"  Plain text      : {text_path}")
    print(f"\nTranscribed text:\n  {result['text'].strip()}")


if __name__ == "__main__":
    main()
