"""
Step 1: Extract a video clip and its audio from the source video.

Usage:
    python scripts/01_extract_clip.py
    python scripts/01_extract_clip.py --input input/source.mp4 --start 15 --duration 15
"""

import subprocess
import os
import argparse
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def extract_clip(input_video: str, output_clip: str, start: int, duration: int):
    os.makedirs(os.path.dirname(output_clip), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-ss", str(start),
        "-t", str(duration),
        "-c", "copy",       # stream copy — no re-encode, instant
        output_clip
    ]
    print(f"Extracting clip [{start}s -> {start+duration}s] from: {input_video}")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Clip saved: {output_clip}")


def extract_audio(input_clip: str, output_wav: str):
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_clip,
        "-vn",              # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",     # 16kHz — required by Whisper
        "-ac", "1",         # mono
        output_wav
    ]
    print(f"Extracting audio from: {input_clip}")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Audio saved: {output_wav}")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Extract clip and audio")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_video = args.input or cfg["source_video"]
    start = args.start if args.start is not None else cfg["clip"]["start_time"]
    duration = args.duration if args.duration is not None else cfg["clip"]["duration"]

    output_clip = os.path.join(cfg["output"]["clips_dir"], "clip.mp4")
    output_wav = os.path.join(cfg["output"]["audio_dir"], "clip_audio.wav")

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Source video not found: {input_video}")

    extract_clip(input_video, output_clip, start, duration)
    extract_audio(output_clip, output_wav)

    print("\nStep 1 complete.")
    print(f"  Clip : {output_clip}")
    print(f"  Audio: {output_wav}")


if __name__ == "__main__":
    main()
