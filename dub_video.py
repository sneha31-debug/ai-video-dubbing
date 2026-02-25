"""
Main pipeline orchestrator for Supernan AI Video Dubbing.

Runs all 6 steps in sequence:
  1. Extract clip + audio (FFmpeg)
  2. Transcribe audio      (Whisper)
  3. Translate to Hindi    (IndicTrans2 / googletrans)
  4. Clone voice to Hindi  (Coqui XTTS v2)
  5. Lip sync video        (Wav2Lip)
  6. Restore faces         (GFPGAN)

Usage:
    python dub_video.py
    python dub_video.py --input input/video.mp4 --start 15 --duration 15
    python dub_video.py --skip-lipsync        # useful for quick translation checks
    python dub_video.py --only-step 2         # run a single step
"""

import argparse
import sys
import os
import time
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def override_config(cfg, args):
    if args.input:
        cfg["source_video"] = args.input
    if args.start is not None:
        cfg["clip"]["start_time"] = args.start
    if args.duration is not None:
        cfg["clip"]["duration"] = args.duration
    return cfg


def run_step(step_num: int, label: str, fn, skip: bool = False):
    if skip:
        print(f"\n[Step {step_num}] {label} — SKIPPED")
        return
    print(f"\n{'='*55}")
    print(f"[Step {step_num}/6] {label}")
    print(f"{'='*55}")
    t0 = time.time()
    fn()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Supernan AI Video Dubbing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dub_video.py
  python dub_video.py --input input/video.mp4 --start 15 --duration 15
  python dub_video.py --only-step 3
  python dub_video.py --skip-lipsync --skip-restore
        """
    )
    parser.add_argument("--input", type=str, help="Path to source video")
    parser.add_argument("--start", type=int, help="Clip start time in seconds")
    parser.add_argument("--duration", type=int, help="Clip duration in seconds")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--only-step", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Run only a single step (1-6)")
    parser.add_argument("--skip-lipsync", action="store_true", help="Skip Wav2Lip (step 5)")
    parser.add_argument("--skip-restore", action="store_true", help="Skip GFPGAN (step 6)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    cfg = load_config(args.config)
    cfg = override_config(cfg, args)

    # Write back overrides so all sub-scripts pick them up via config
    with open(args.config, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    only = args.only_step

    # Import step functions lazily (avoids loading heavy models until needed)
    def step1():
        from scripts.extract_clip import extract_clip, extract_audio
        clip_path = os.path.join(cfg["output"]["clips_dir"], "clip.mp4")
        wav_path = os.path.join(cfg["output"]["audio_dir"], "clip_audio.wav")
        extract_clip(cfg["source_video"], clip_path, cfg["clip"]["start_time"], cfg["clip"]["duration"])
        extract_audio(clip_path, wav_path)

    def step2():
        from scripts.transcribe import transcribe
        import json
        audio_path = os.path.join(cfg["output"]["audio_dir"], "clip_audio.wav")
        result = transcribe(audio_path, cfg["whisper"]["model_size"],
                            cfg["whisper"]["language"], cfg["whisper"]["device"])
        out_dir = cfg["output"]["transcripts_dir"]
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "transcript.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(result["text"].strip())

    def step3():
        import subprocess
        subprocess.run(["python", "scripts/03_translate.py", "--config", args.config], check=True)

    def step4():
        import subprocess
        subprocess.run(["python", "scripts/04_voice_clone.py", "--config", args.config], check=True)

    def step5():
        import subprocess
        subprocess.run(["python", "scripts/05_lipsync.py", "--config", args.config], check=True)

    def step6():
        import subprocess
        subprocess.run(["python", "scripts/06_face_restore.py", "--config", args.config], check=True)

    steps = {
        1: ("Extract clip & audio", step1, False),
        2: ("Transcribe (Whisper)", step2, False),
        3: ("Translate to Hindi (IndicTrans2)", step3, False),
        4: ("Voice clone (XTTS v2)", step4, False),
        5: ("Lip sync (Wav2Lip)", step5, args.skip_lipsync),
        6: ("Face restoration (GFPGAN)", step6, args.skip_restore),
    }

    total_start = time.time()

    if only:
        label, fn, skip = steps[only]
        run_step(only, label, fn, skip=False)
    else:
        for num, (label, fn, skip) in steps.items():
            run_step(num, label, fn, skip=skip)

    total = time.time() - total_start
    final_output = os.path.join(cfg["output"]["final_dir"], cfg["output"]["final_filename"])

    print(f"\n{'='*55}")
    print(f"Pipeline complete in {total:.1f}s")
    if os.path.exists(final_output):
        print(f"Final output: {final_output}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
