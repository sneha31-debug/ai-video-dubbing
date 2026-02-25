import subprocess
import os


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def get_video_fps(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    fps_str = result.stdout.strip()  # e.g. "30000/1001"
    num, den = fps_str.split("/")
    return float(num) / float(den)


def get_frame_count(video_path: str) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


def extract_frames(video_path: str, output_dir: str, fps: int = None):
    os.makedirs(output_dir, exist_ok=True)
    fps_filter = f"fps={fps}" if fps else "fps=25"
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", fps_filter,
        os.path.join(output_dir, "frame_%04d.png"),
        "-y"
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Frames extracted to: {output_dir}")


def frames_to_video(frames_dir: str, audio_path: str, output_path: str, fps: float = 25.0):
    frame_pattern = os.path.join(frames_dir, "frame_%04d.png")
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-i", audio_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        output_path,
        "-y"
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Video assembled: {output_path}")
