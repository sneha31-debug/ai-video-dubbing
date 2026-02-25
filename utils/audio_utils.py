import librosa
import soundfile as sf
import numpy as np


def get_audio_duration(path: str) -> float:
    y, sr = librosa.load(path, sr=None)
    return librosa.get_duration(y=y, sr=sr)


def time_stretch_audio(input_path: str, output_path: str, target_duration: float, max_stretch_factor: float = 1.3) -> bool:
    """
    Stretch or compress audio to match target_duration using librosa.
    Returns True if stretch was applied, False if out of bounds.
    """
    y, sr = librosa.load(input_path, sr=None)
    current_duration = librosa.get_duration(y=y, sr=sr)

    if current_duration == 0:
        raise ValueError(f"Audio file is empty: {input_path}")

    stretch_factor = current_duration / target_duration  # >1 = speed up, <1 = slow down

    if stretch_factor > max_stretch_factor or stretch_factor < (1 / max_stretch_factor):
        print(f"Warning: stretch factor {stretch_factor:.2f} exceeds max {max_stretch_factor}. Applying anyway.")

    y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
    sf.write(output_path, y_stretched, sr)
    print(f"Audio stretched: {current_duration:.2f}s -> {target_duration:.2f}s (factor: {stretch_factor:.3f})")
    return True


def resample_audio(input_path: str, output_path: str, target_sr: int = 16000):
    y, sr = librosa.load(input_path, sr=target_sr)
    sf.write(output_path, y, target_sr)
    print(f"Resampled {input_path} to {target_sr}Hz -> {output_path}")


def trim_silence(input_path: str, output_path: str, top_db: int = 20):
    y, sr = librosa.load(input_path, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    sf.write(output_path, y_trimmed, sr)
    print(f"Silence trimmed: {output_path}")
