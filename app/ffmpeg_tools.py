from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import List

from rich.console import Console

from .segment_matcher import Segment

console = Console()


def _run(cmd: List[str]) -> None:
    """
    Run a subprocess command and show stderr on failure.
    """
    console.print(f"[bold cyan]$ {' '.join(cmd)}[/bold cyan]")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        console.print("[bold red]FFmpeg error:[/bold red]")
        if proc.stdout:
            console.print(proc.stdout)
        if proc.stderr:
            console.print(proc.stderr)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def write_segments_json(path: Path, segments: List[Segment]) -> None:
    payload = [asdict(s) for s in segments]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def cut_segments(
    video_path: Path,
    out_dir: Path,
    segments: List[Segment],
    progress_cb=None,
) -> List[Path]:
    """
    Cut segments into VLC / YouTube / MP4-safe files.

    Progress:
    - Calls progress_cb(i, total, segment) BEFORE each cut
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    total = len(segments)

    for i, seg in enumerate(segments, start=1):

        # 🔹 Progress BEFORE starting this segment
        if progress_cb:
            progress_cb(i - 1, total, seg)

        out_path = out_dir / f"seg_{i:03d}.mp4"
        duration = max(0.001, seg.end_s - seg.start_s)

        cmd = [
            "ffmpeg",
            "-y",

            "-ss", f"{seg.start_s:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(video_path),

            "-map", "0:v:0",
            "-map", "0:a:m:language:eng?",
            "-map", "0:a:0?",

            "-sn",
            "-dn",
            "-map_chapters", "-1",

            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.0",
            "-preset", "veryfast",
            "-crf", "23",

            "-c:a", "aac",
            "-b:a", "192k",
            "-ac", "2",
            "-ar", "48000",

            "-af", "aresample=async=1:first_pts=0",
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",

            str(out_path),
        ]

        _run(cmd)
        outputs.append(out_path)

        # 🔹 Progress AFTER finishing this segment
        if progress_cb:
            progress_cb(i, total, seg)

    return outputs

def overlay_label(
    video_in: Path,
    label_mp4: Path,
    video_out: Path,
    video_height: int = 1300,
):
    """
    Shorts composition:
    LABEL (top)
    VIDEO (center)
    LABEL (bottom)
    """

    canvas_w = 1080
    # canvas_h = 1920 - video_height 
    canvas_h = 1920
    video_height =  min(video_height, canvas_h - 2)

    top_h = (canvas_h - video_height) // 2
    bottom_h = canvas_h - video_height - top_h

    filter_complex = (
        # Background label (full 9:16)
        "[0:v]scale=1080:1920,setsar=1[label];"

        # Scale main video to FIT inside label
        f"[1:v]scale=1080:{video_height}:"
        "force_original_aspect_ratio=decrease,"
        "setsar=1[video];"

        # Center overlay
        "[label][video]overlay="
        "(W-w)/2:"
        "(H-h)/2"
        "[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", str(label_mp4),
        "-i", str(video_in),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "1:a?",
        "-shortest",  # 🔑 THIS IS THE FIX
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        str(video_out),
    ]

    subprocess.run(cmd, check=True)

def overlay_label_to_segments(
    segment_files: List[Path],
    label_mp4: Path,
    out_dir: Path,
    video_height: int,
) -> List[Path]:
    """
    Apply LadyAnime label overlay to EVERY segment.

    Returns:
        List of labeled segment paths (same filenames).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_segments: List[Path] = []

    for seg in segment_files:
        out_path = out_dir / seg.name

        overlay_label(
            video_in=seg,
            label_mp4=label_mp4,
            video_out=out_path,
            video_height=video_height,
        )

        labeled_segments.append(out_path)

    return labeled_segments

def concat_segments(segment_files: List[Path], output_path: Path) -> None:
    """
    Concatenate already-clean MP4 segments.
    Fast stream copy is now SAFE because all segments are identical.
    """
    if not segment_files:
        raise ValueError("No segment files to concatenate.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_path.parent / "concat_list.txt"

    content = "\n".join(f"file '{p.as_posix()}'" for p in segment_files)
    list_file.write_text(content, encoding="utf-8")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]

    _run(cmd)
