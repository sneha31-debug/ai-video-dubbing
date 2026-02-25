"""
Step 4: Generate Hindi audio using edge-tts (Microsoft Neural TTS).

Uses Microsoft's free edge-tts library (hi-IN-SwaraNeural voice) to
synthesize the Hindi translated text. Works on any Python version.
If the resulting audio is longer than the clip duration, it is
time-stretched to fit.

Usage:
    python scripts/voice_clone.py
    pip install edge-tts  (already in requirements.txt)
"""

import os
import argparse
import asyncio
import yaml

from utils.audio_utils import get_audio_duration, time_stretch_audio


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


async def _synthesize_edge_tts(text: str, voice: str, output_mp3: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_mp3)


def generate_hindi_audio(hindi_text: str, output_path: str, voice: str = "hi-IN-SwaraNeural"):
    """Synthesize Hindi text to audio using Microsoft edge-tts."""
    import soundfile as sf
    import numpy as np

    # edge-tts outputs MP3; convert to WAV via pydub
    mp3_path = output_path.replace(".wav", "_tmp.mp3")
    asyncio.run(_synthesize_edge_tts(hindi_text, voice, mp3_path))

    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(22050).set_channels(1)
    audio.export(output_path, format="wav")
    os.remove(mp3_path)
    print(f"Hindi audio saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 4: Hindi TTS with edge-tts")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    translation_path = os.path.join(cfg["output"]["translations_dir"], "translation.txt")
    speaker_wav = os.path.join(cfg["output"]["audio_dir"], "clip_audio.wav")
    raw_hindi_wav = os.path.join(cfg["output"]["audio_dir"], "hindi_audio_raw.wav")
    final_hindi_wav = os.path.join(cfg["output"]["audio_dir"], "hindi_audio.wav")

    if not os.path.exists(translation_path):
        raise FileNotFoundError(f"Translation not found: {translation_path}. Run translate.py first.")

    with open(translation_path, encoding="utf-8") as f:
        hindi_text = f.read().strip()

    print(f"Hindi text to synthesize:\n  {hindi_text}\n")

    voice = cfg.get("tts", {}).get("voice", "hi-IN-SwaraNeural")
    print(f"Using voice: {voice}")

    generate_hindi_audio(
        hindi_text=hindi_text,
        output_path=raw_hindi_wav,
        voice=voice,
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
