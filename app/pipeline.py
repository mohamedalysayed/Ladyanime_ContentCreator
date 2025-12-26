from __future__ import annotations
from .ffmpeg_tools import get_video_duration
from pathlib import Path
from typing import List, Callable, Optional

from rich.console import Console
from rich.panel import Panel

from .config import AppConfig
from .ffmpeg_tools import cut_segments, concat_segments, write_segments_json
from .srt_parser import load_srt
from .segment_matcher import Segment, build_segments_from_subtitles
import math
import subprocess
# from typing import List
from .ffmpeg_tools import overlay_label
from .ffmpeg_tools import rhythmic_cut

console = Console()

def run_mvp(
    config: AppConfig,
    progress_cb: Optional[Callable[[int, int, Segment], None]] = None,
    max_block_sec: float = 32.0,
    silence_gap_sec: float = 5.0,
    min_segment_sec: float = 18.0,
    segment_files_override: Optional[List[Path]] = None, 
) -> Path:
    """
    Run the end-to-end LadyAnime MVP pipeline.

    Pipeline steps:
    ---------------
    1) Validate inputs (video + subtitles)
    2) Parse subtitles (.srt)
    3) Build time-based segments from subtitles
    4) Cut video segments via FFmpeg (audio-safe, VLC-safe)
    5) Concatenate segments into final recap.mp4
    6) Save segments.json for debugging and future AI logic

    Parameters:
    -----------
    config : AppConfig
        Central configuration (paths, filenames, limits).

    progress_cb : optional callable
        Called during segment cutting as:
            progress_cb(current_index, total_segments, segment)

        Used by Gradio to update progress bars.
        Safe to be None (CLI mode).

    max_block_sec : float
        Target segment duration (e.g. ~30s).

    silence_gap_sec : float
        Silence duration that forces a new segment.

    min_segment_sec : float
        Minimum allowed segment length (prevents micro-clips).

    Returns:
    --------
    Path
        Path to final recap video.
    """

    # ------------------------------------------------------------------
    # 0) Prepare output directory and validate inputs
    # ------------------------------------------------------------------
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not config.video_path.exists():
        raise FileNotFoundError(f"Missing input video: {config.video_path}")

    if not config.subtitles_path.exists():
        raise FileNotFoundError(f"Missing input subtitles: {config.subtitles_path}")

    console.print(
        Panel.fit(
            "LadyAnime MVP — Subtitle → Segments → Recap",
            style="bold magenta",
        )
    )

    # ------------------------------------------------------------------
    # 1) Parse subtitles
    # ------------------------------------------------------------------
    lines = load_srt(str(config.subtitles_path))
    console.print(f"[green]Loaded[/green] {len(lines)} subtitle lines")

    if not lines:
        raise RuntimeError("Subtitle file parsed successfully but contains no usable entries.")

    subtitle_tuples = [(l.start_s, l.end_s, l.text) for l in lines]

    # ------------------------------------------------------------------
    # 2) Build time-based segments
    # ------------------------------------------------------------------
    segments: List[Segment] = build_segments_from_subtitles(
        subtitle_lines=subtitle_tuples,
        max_block_sec=max_block_sec,
        silence_gap_sec=silence_gap_sec,
        min_segment_sec=min_segment_sec,
        max_segments=config.max_segments,
    )

    if not segments:
        console.print("[yellow]No segments produced.[/yellow] Check subtitle timing/format.")
        write_segments_json(config.output_dir / "segments.json", [])
        return config.output_dir / f"{config.output_basename}.mp4"

    console.print(f"[green]Selected[/green] {len(segments)} segments")

    # ------------------------------------------------------------------
    # 3) Save segments.json (debug + AI later)
    # ------------------------------------------------------------------
    segments_json = config.output_dir / "segments.json"
    write_segments_json(segments_json, segments)
    console.print(f"[cyan]Wrote[/cyan] {segments_json}")

    # ------------------------------------------------------------------
    # 4) Cut segments (audio-safe)
    # ------------------------------------------------------------------
    seg_dir = config.output_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    seg_files = cut_segments(
        video_path=config.video_path,
        out_dir=seg_dir,
        segments=segments,
        progress_cb=progress_cb,
    )

    if not seg_files:
        raise RuntimeError("Segment cutting finished but produced no output files.")

    console.print(f"[green]Cut[/green] {len(seg_files)} segment files")

    # ------------------------------------------------------------------
    # 5) Concatenate final recap
    # ------------------------------------------------------------------
    out_video = config.output_dir / f"{config.output_basename}.mp4"
    final_segments = segment_files_override or seg_files
    concat_segments(final_segments, out_video)


    console.print(f"[bold green]DONE[/bold green] → {out_video}")

    return out_video

def run_rhythmic_recap(
    config: AppConfig,
    intro_skip_sec: float,
    outro_skip_sec: float,
    keep_sec: float,
    skip_sec: float,
    speed_factor: float = 1.0,
    emit_recap: bool = True,
    progress_cb=None,
    return_timeline: bool = False,
) -> Path | None:

    if speed_factor <= 0:
        speed_factor = 1.0

    intro_skip_sec /= speed_factor
    keep_sec /= speed_factor
    skip_sec /= speed_factor

    seg_dir = config.output_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(config.video_path)
    start_t = max(0.0, intro_skip_sec)
    end_t   = max(start_t, duration - outro_skip_sec)
    #active_duration = end_t - start_t

    segments, timeline = rhythmic_cut(
        video_path=config.video_path,
        out_dir=seg_dir,
        start_offset=start_t,
        end_offset=end_t,
        keep_sec=keep_sec,
        skip_sec=skip_sec,
        progress_cb=progress_cb,
    )

    if not segments:
        raise RuntimeError("No rhythmic segments produced.")

    if not emit_recap:
        return (None, timeline) if return_timeline else None

    out_video = config.output_dir / f"{config.output_basename}.mp4"
    concat_segments(segments, out_video)

    if return_timeline:
        return out_video, timeline

    return out_video

def run_shorts(
    video_path: Path,
    out_dir: Path,
    clip_duration: int,
    add_label: bool,
    video_height: int,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    max_shorts: Optional[int] = None,
):
    """
    Create shorts WITHOUT subtitles.
    Cuts fixed-duration clips and optionally applies label.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Missing input video: {video_path}")

    # --------------------------------------------------
    # 1) Probe video duration  ✅ FIRST
    # --------------------------------------------------
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    duration = float(subprocess.check_output(probe_cmd).decode().strip())

    # --------------------------------------------------
    # 2) Compute number of shorts
    # --------------------------------------------------
    total_clips = max(1, math.floor(duration / clip_duration))

    if max_shorts is not None:
        total_clips = min(total_clips, max_shorts)

    generated: list[Path] = []

    # --------------------------------------------------
    # 3) Cut fixed-length shorts
    # --------------------------------------------------
    for i in range(total_clips):
        if progress_cb:
            progress_cb(i, total_clips)

        start = i * clip_duration
        out_clip = out_dir / f"short_{i+1:03d}.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-t", str(clip_duration),
            "-i", str(video_path),
            "-vf", "setsar=1",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ac", "2",
            str(out_clip),
        ]   
        subprocess.run(cmd, check=True)
        generated.append(out_clip)

    if not generated:
        raise RuntimeError("No shorts were generated.")

    # --------------------------------------------------
    # 4) Optional LadyAnime label
    # --------------------------------------------------
    if add_label:
        label_mp4 = (
            video_path.parents[2]
            / "data"
            / "shorts_label"
            / "shortsLabel_LA.mp4"
        )

        labeled_dir = out_dir / "labeled"
        labeled_dir.mkdir(exist_ok=True)

        labeled: list[Path] = []

        for clip in generated:
            out_labeled = labeled_dir / clip.name

            overlay_label(
                video_in=clip,
                label_mp4=label_mp4,
                video_out=out_labeled,
                video_height=video_height,
            )

            labeled.append(out_labeled)

        generated = labeled

        # Replace originals with labeled versions
        for f in out_dir.glob("short_*.mp4"):
            f.unlink()

        for f in labeled:
            f.rename(out_dir / f.name)

    # --------------------------------------------------
    # 5) Return ONE path for Gradio
    # --------------------------------------------------
    return generated[0]

