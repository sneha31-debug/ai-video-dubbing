# AI Video Dubbing Pipeline — Supernan Intern Challenge

End-to-end Python pipeline that takes an English training video and outputs a Hindi-dubbed clip with voice cloning and lip sync.

---

## Pipeline Overview

| Step | Task | Tool | Cost |
|------|------|------|------|
| 1 | Video/audio extraction | FFmpeg | Free |
| 2 | Speech transcription | OpenAI Whisper (small) | Free |
| 3 | English → Hindi translation | IndicTrans2 / googletrans | Free |
| 4 | Hindi voice cloning | Coqui XTTS v2 | Free |
| 5 | Lip sync | Wav2Lip | Free |
| 6 | Face restoration | GFPGAN | Free |

**Estimated cost per minute of video: Rs. 0** (Google Colab T4 Free Tier)

---

## Project Structure

```
ai-video-dubbing/
├── dub_video.py              # Main pipeline orchestrator
├── config.yaml               # All configuration (paths, model names, timestamps)
├── requirements.txt
├── setup.sh                  # One-command environment setup
├── scripts/
│   ├── 01_extract_clip.py
│   ├── 02_transcribe.py
│   ├── 03_translate.py
│   ├── 04_voice_clone.py
│   ├── 05_lipsync.py
│   └── 06_face_restore.py
├── utils/
│   ├── audio_utils.py        # Time-stretching, audio helpers
│   └── video_utils.py
├── input/                    # Place source video here
├── output/                   # All intermediate and final outputs
└── notebooks/
    └── ai_dubbing_colab.ipynb
```

---

## Quick Start

### Option A: Google Colab (Recommended — Free T4 GPU)

1. Open `notebooks/ai_dubbing_colab.ipynb` in Google Colab
2. Runtime > Change runtime type > **T4 GPU**
3. Run all cells

### Option B: Local Setup

**Prerequisites:** Python 3.9+, FFmpeg (`brew install ffmpeg` on Mac)

```bash
git clone https://github.com/your-username/ai-video-dubbing.git
cd ai-video-dubbing
chmod +x setup.sh && ./setup.sh
```

**Run the pipeline:**

```bash
# Full pipeline
python dub_video.py --input input/source_video.mp4 --start 15 --duration 15

# Or run individual steps
python scripts/01_extract_clip.py
python scripts/02_transcribe.py
python scripts/03_translate.py
python scripts/04_voice_clone.py
python scripts/05_lipsync.py
python scripts/06_face_restore.py
```

---

## Architecture

```
Input Video (MP4)
      |
[FFmpeg]        --> clip.mp4 + clip_audio.wav
      |
[Whisper]       --> English transcript + timestamps
      |
[IndicTrans2]   --> Hindi translated text
      |
[XTTS v2]       --> Hindi audio (voice-cloned WAV)
      |               ^ time-stretched to match clip duration via librosa
[Wav2Lip]       --> Lip-synced video
      |
[GFPGAN]        --> Face-restored final output
      |
Output: hindi_dubbed_final.mp4
```

---

## Design Decisions

**Whisper:** Runs on CPU, provides word-level timestamps for precise sync. `small` model is the best balance of speed and accuracy for 15-second clips.

**IndicTrans2 over Google Translate:** Purpose-built for Indian languages — handles natural Hindi idioms. Falls back to `googletrans` automatically if VRAM is insufficient.

**Coqui XTTS v2:** Best free voice cloning available — needs only 3–6 seconds of reference audio and supports Hindi natively.

**Wav2Lip over VideoReTalking:** Lighter and faster on Colab free tier. GFPGAN post-processing compensates for the blurriness Wav2Lip introduces.

**Audio time-stretching:** Hindi speech is typically ~15–20% longer than English. `librosa.effects.time_stretch` compresses the Hindi audio to exactly match the original duration without degrading voice quality.

---

## Scaling to 500 Hours of Video

```
Input Videos (S3)
      |
Job Queue (Redis / SQS)
      |
  Worker Nodes x N          # e.g., 8x A100 80GB spot instances
  (Whisper + XTTS + Wav2Lip per worker)
      |
Output Store (S3)
      |
GFPGAN batch post-processing (parallel GPU)
```

- Cost estimate: ~$2–3 per hour of video on A100 spot instances
- 500 hours overnight: ~16 A100 instances x 8 hours = ~$200–300 total
- Long audio is silence-split into 30-second chunks, processed in parallel, then stitched with FFmpeg

---

## Known Limitations

- Wav2Lip produces blurry faces; GFPGAN mitigates but does not fully eliminate this
- XTTS on CPU is slow (~5 min for 15 seconds) — use Colab GPU
- Time-stretching beyond 25% can slightly alter voice quality
- IndicTrans2 needs ~6 GB VRAM; falls back to googletrans automatically on low-VRAM environments
- Pipeline assumes a single speaker

---

## What I'd Improve with More Time

- Switch to VideoReTalking for sharper lip sync
- Add multi-speaker diarization (pyannote.audio)
- Implement async parallel processing with FastAPI + Celery
- Deploy a Gradio demo on Hugging Face Spaces

---

## Submission

Output video: `output/final/hindi_dubbed_final.mp4`  
Email: ganesh@supernan.app | Subject: `Supernan AI Intern – [Your Name]`