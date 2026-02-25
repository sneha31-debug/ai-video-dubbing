"""
Step 6: Face restoration using GFPGAN.

Wav2Lip produces blurry faces. GFPGAN sharpens and restores them.
This is critical for the 40% visual fidelity score.

Requires:
  - GFPGAN/ repo cloned and installed (via setup.sh)
  - GFPGANv1.4.pth downloaded to GFPGAN/experiments/pretrained_models/

Usage:
    python scripts/06_face_restore.py
"""

import subprocess
import os
import shutil
import argparse
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_gfpgan(input_video: str, output_path: str, model_path: str, upscale: int = 2):
    if not os.path.isdir("GFPGAN"):
        raise FileNotFoundError("GFPGAN repo not found. Run setup.sh first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"GFPGAN model not found: {model_path}\n"
            "Download from: https://github.com/TencentARC/GFPGAN/releases/tag/v1.3.4"
        )

    # GFPGAN only processes images, so we:
    # 1. Extract frames from lip-synced video
    # 2. Pass frames dir to GFPGAN
    # 3. Reassemble video with restored frames + original audio

    frames_dir = "output/clips/frames_raw"
    restored_dir = "output/clips/frames_restored"
    os.makedirs(frames_dir, exist_ok=True)

    # Extract frames at 25fps
    print("Extracting frames for face restoration...")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        "-vf", "fps=25",
        os.path.join(frames_dir, "frame_%04d.png")
    ], check=True, capture_output=True)

    # Run GFPGAN on frame directory
    print(f"Running GFPGAN (upscale={upscale})...")
    result = subprocess.run([
        "python", "GFPGAN/inference_gfpgan.py",
        "-i", frames_dir,
        "-o", restored_dir,
        "-v", "1.4",
        "--upscale", str(upscale),
        "--only_center_face"   # only restore the main speaking face
    ], capture_output=False)

    if result.returncode != 0:
        raise RuntimeError("GFPGAN inference failed.")

    # Restored frames land in restored_dir/restored_imgs/
    restored_frames = os.path.join(restored_dir, "restored_imgs")

    # Get FPS and audio from original lip-synced video
    fps_result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", input_video
    ], capture_output=True, text=True)
    fps_str = fps_result.stdout.strip()
    num, den = fps_str.split("/")
    fps = float(num) / float(den)

    # Extract audio from lip-synced video
    temp_audio = "output/audio/lipsynced_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-acodec", "pcm_s16le", temp_audio
    ], check=True, capture_output=True)

    # Reassemble: restored frames + lip-synced audio
    print(f"Assembling final video: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(restored_frames, "frame_%04d.png"),
        "-i", temp_audio,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        output_path
    ], check=True, capture_output=True)

    # Cleanup temp frame dirs
    shutil.rmtree(frames_dir, ignore_errors=True)

    print(f"Face-restored video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 6: GFPGAN face restoration")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_video = os.path.join(cfg["output"]["clips_dir"], "lipsynced.mp4")
    output_path = os.path.join(cfg["output"]["final_dir"], cfg["output"]["final_filename"])

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Lip-synced video not found: {input_video}. Run 05_lipsync.py first.")

    run_gfpgan(
        input_video=input_video,
        output_path=output_path,
        model_path=cfg["gfpgan"]["model_path"],
        upscale=cfg["gfpgan"]["upscale"]
    )

    print("\nStep 6 complete.")
    print(f"  Final output: {output_path}")


if __name__ == "__main__":
    main()
