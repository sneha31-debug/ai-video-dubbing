"""
Step 4: Generate Hindi audio using Coqui XTTS v2 voice cloning.

The original speaker's voice is cloned from the extracted clip audio,
then used to speak the Hindi translated text. If the resulting audio
is longer than the clip duration, it is time-stretched to fit.

Usage:
    python scripts/04_voice_clone.py
"""

import os
import argparse
import yaml

from utils.audio_utils import get_audio_duration, time_stretch_audio


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def generate_hindi_audio(hindi_text: str, speaker_wav: str, output_path: str, model_name: str, language: str):
    from TTS.api import TTS
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS v2 on: {device}")

    tts = TTS(model_name).to(device)
    tts.tts_to_file(
        text=hindi_text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=output_path
    )
    print(f"Hindi audio saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 4: Voice cloning with XTTS v2")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    translation_path = os.path.join(cfg["output"]["translations_dir"], "translation.txt")
    speaker_wav = os.path.join(cfg["output"]["audio_dir"], "clip_audio.wav")
    raw_hindi_wav = os.path.join(cfg["output"]["audio_dir"], "hindi_audio_raw.wav")
    final_hindi_wav = os.path.join(cfg["output"]["audio_dir"], "hindi_audio.wav")

    if not os.path.exists(translation_path):
        raise FileNotFoundError(f"Translation not found: {translation_path}. Run 03_translate.py first.")
    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(f"Speaker audio not found: {speaker_wav}. Run 01_extract_clip.py first.")

    with open(translation_path, encoding="utf-8") as f:
        hindi_text = f.read().strip()

    print(f"Hindi text to synthesize:\n  {hindi_text}\n")

    generate_hindi_audio(
        hindi_text=hindi_text,
        speaker_wav=speaker_wav,
        output_path=raw_hindi_wav,
        model_name=cfg["tts"]["model_name"],
        language=cfg["tts"]["language"]
    )

    # Time-stretch Hindi audio to match original clip duration
    if cfg["audio"]["time_stretch"]:
        original_duration = get_audio_duration(speaker_wav)
        hindi_duration = get_audio_duration(raw_hindi_wav)

        print(f"Original duration : {original_duration:.2f}s")
        print(f"Hindi audio length: {hindi_duration:.2f}s")

        if abs(hindi_duration - original_duration) > 0.5:
            time_stretch_audio(
                input_path=raw_hindi_wav,
                output_path=final_hindi_wav,
                target_duration=original_duration,
                max_stretch_factor=cfg["audio"]["max_stretch_factor"]
            )
        else:
            import shutil
            shutil.copy(raw_hindi_wav, final_hindi_wav)
            print("Durations close enough — no stretch needed.")
    else:
        import shutil
        shutil.copy(raw_hindi_wav, final_hindi_wav)

    print("\nStep 4 complete.")
    print(f"  Raw Hindi audio     : {raw_hindi_wav}")
    print(f"  Stretched Hindi audio: {final_hindi_wav}")


if __name__ == "__main__":
    main()
