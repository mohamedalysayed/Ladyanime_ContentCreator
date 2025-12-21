# 🎬 LadyAnime Video Engine 

LadyAnime Video Engine is a local-first video processing tool designed to turn long anime episodes into clean recaps and labeled YouTube-ready clips.

This repository contains the MVP implementation, focused on correctness, determinism, and extensibility.

---

## Features

### Recap Generator
- Input: full episode video (.mkv / .mp4) + subtitles (.srt)
- Automatically:
  - Parses subtitles
  - Detects scene-like segments using timing and silence gaps
  - Cuts segments with audio-safe FFmpeg settings
  - Concatenates them into a recap video

### LadyAnime Label Overlay
- Optional animated MP4 label overlay
- Can be applied to:
  - Final recap video
  - Individual segments
- Adjustable main video height to control label visibility (top/bottom)

### Automatic Transcription (Optional)
- If subtitles are missing:
  - Uses Faster-Whisper
  - Generates subtitles.srt automatically
- Can be disabled depending on mode (e.g. Shorts)

### Local Gradio UI
- Upload video & subtitles
- Fine-tune segmentation parameters
- Real-time progress bar (per segment)
- Preview output video
- Debug-friendly outputs (segments.json)

---

## Design Principles

- Local-first (no cloud, no APIs)
- Deterministic output
- FFmpeg-safe (VLC / YouTube compatible)
- AI-ready structured outputs
- Modular architecture

This is a system meant to be extended, not a black box.

---

## Project Structure

```text
ladyanime_mvp/
├── app/
│   ├── main.py
│   ├── gui.py
│   ├── pipeline.py
│   ├── ffmpeg_tools.py
│   ├── srt_parser.py
│   ├── segment_matcher.py
│   ├── transcribe.py
│   └── config.py
│
├── data/
│   ├── input/
│   ├── output/
│   └── shorts_label/
│       └── shortsLabel_LA.mp4
│
├── requirements.txt
├── README.md
└── run.sh
```

---

## Requirements

### System
- Linux (tested on Ubuntu)
- Python 3.10+
- FFmpeg

Install FFmpeg:
sudo apt install ffmpeg

### Python Dependencies
Installed via requirements.txt:
- gradio
- rich
- pysrt
- faster-whisper
- numpy

---

## Installation & Usage

### 1. Clone Repository
git clone https://github.com/<your-username>/ladyanime_mvp.git
cd ladyanime_mvp

### 2. Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### 3. Launch UI
python -m app.gui

Open in browser:
http://127.0.0.1:7860

---

## Gradio Interface

### Recap Generator Tab
- Upload episode video
- Upload subtitles (optional)
- Controls:
  - Segment length
  - Silence gap
  - Minimum segment length
  - Maximum segments
  - Reuse existing segments
  - Apply LadyAnime label
  - Adjust main video height

### Outputs
- recap.mp4 or recap_labeled.mp4
- segments/seg_XXX.mp4
- segments_labeled/seg_XXX.mp4
- segments.json

---

## segments.json

Example:
[
  {
    "start_s": 120.5,
    "end_s": 152.3,
    "score": 1
  }
]

This file is designed for future AI scene selection.

---

## Label Overlay

- Location:
  data/shorts_label/shortsLabel_LA.mp4
- Resolution: 1080x1920
- Looped automatically if shorter than video
- Main video is scaled down and centered
- Label remains visible top & bottom

---

## MVP Limitations

- Subtitle-based segmentation only
- No semantic AI yet
- No face tracking
- No batch episode processing

These are intentional MVP boundaries.

---

## Roadmap

- YouTube Shorts mode (9:16, 15s/30s/60s)
- Auto zoom & smart crop
- AI scene ranking
- Desktop application (PySide6)

---

## License

MIT License

---

LadyAnime Video Engine is the technical foundation for future creator tools.
You are early.

