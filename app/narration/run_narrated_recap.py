# run_narrated_recap.py
from pathlib import Path
import subprocess
import json

from .summarizer import summarize_full_episode
from .voice import VoiceSynthesizer


def get_video_duration(video: Path) -> float:
    import subprocess, json
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def run_ai_narrated_recap(
    recap_video: Path,
    episode_video: Path,     # kept for future season logic
    episode_srt: Path,
    segments_json: Path,     # ignored for narration
    out_dir: Path,
) -> tuple[Path, str]:

    out_dir.mkdir(parents=True, exist_ok=True)

    silent_video = out_dir / "recap_silent.mp4"
    narration_wav = out_dir / "narration.wav"
    final_video = out_dir / "recap_narrated.mp4"

    # --------------------------------------------------
    # 1) Strip original audio
    # --------------------------------------------------
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(recap_video),
            "-c:v", "copy",
            "-an",
            str(silent_video),
        ],
        check=True,
    )

    recap_duration = get_video_duration(silent_video)

    # --------------------------------------------------
    # 2) Build ONE summary from FULL SRT
    # --------------------------------------------------
    original_text, summary_text = summarize_full_episode(
        episode_srt=episode_srt,
        recap_duration_sec=recap_duration,
    )

    narration_panel_text = (
        "ORIGINAL:\n"
        + original_text
        + "\n\n"
        + "-" * 48
        + "\n\n"
        + "SUMMARY:\n"
        + summary_text
    )

    # --------------------------------------------------
    # 3) Generate ONE narration WAV
    # --------------------------------------------------
    voice = VoiceSynthesizer()
    voice.synthesize(
        text=summary_text,
        duration=recap_duration,
        out_wav=narration_wav,
    )

    # --------------------------------------------------
    # 4) Overlay narration
    # --------------------------------------------------
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(silent_video),
            "-i", str(narration_wav),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-movflags", "+faststart",
            str(final_video),
        ],
        check=True,
    )

    return final_video, narration_panel_text
