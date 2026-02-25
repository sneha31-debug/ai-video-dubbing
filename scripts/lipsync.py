"""
Step 5: Lip sync the video clip to the Hindi audio using Wav2Lip.

Wav2Lip requires:
  - Wav2Lip/ repo cloned (via setup.sh)
  - Wav2Lip/checkpoints/wav2lip.pth downloaded

Usage:
    python scripts/05_lipsync.py
"""

import subprocess
import os
import argparse
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_wav2lip(
    face_video: str,
    audio_path: str,
    output_path: str,
    checkpoint: str,
    resize_factor: int = 1,
    face_det_batch_size: int = 16,
    wav2lip_batch_size: int = 128
):
    if not os.path.exists("Wav2Lip/inference.py"):
        raise FileNotFoundError(
            "Wav2Lip repo not found. Run setup.sh first or:\n"
            "  git clone https://github.com/Rudrabha/Wav2Lip.git"
        )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Wav2Lip checkpoint not found: {checkpoint}\n"
            "Download wav2lip.pth from: https://github.com/Rudrabha/Wav2Lip#getting-the-weights"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "python", "Wav2Lip/inference.py",
        "--checkpoint_path", checkpoint,
        "--face", face_video,
        "--audio", audio_path,
        "--outfile", output_path,
        "--resize_factor", str(resize_factor),
        "--face_det_batch_size", str(face_det_batch_size),
        "--wav2lip_batch_size", str(wav2lip_batch_size),
        "--nosmooth"   # disable temporal smoothing for sharper frames
    ]

    print(f"Running Wav2Lip...")
    print(f"  Face  : {face_video}")
    print(f"  Audio : {audio_path}")
    print(f"  Output: {output_path}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("Wav2Lip inference failed. Check the output above for details.")

    print(f"\nLip-synced video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 5: Wav2Lip lip sync")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    face_video = os.path.join(cfg["output"]["clips_dir"], "clip.mp4")
    audio_path = os.path.join(cfg["output"]["audio_dir"], "hindi_audio.wav")
    output_path = os.path.join(cfg["output"]["clips_dir"], "lipsynced.mp4")

    if not os.path.exists(face_video):
        raise FileNotFoundError(f"Clip not found: {face_video}. Run 01_extract_clip.py first.")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Hindi audio not found: {audio_path}. Run 04_voice_clone.py first.")

    run_wav2lip(
        face_video=face_video,
        audio_path=audio_path,
        output_path=output_path,
        checkpoint=cfg["wav2lip"]["checkpoint"],
        resize_factor=cfg["wav2lip"]["resize_factor"],
        face_det_batch_size=cfg["wav2lip"]["face_det_batch_size"],
        wav2lip_batch_size=cfg["wav2lip"]["wav2lip_batch_size"]
    )

    print("\nStep 5 complete.")
    print(f"  Lip-synced video: {output_path}")


if __name__ == "__main__":
    main()
