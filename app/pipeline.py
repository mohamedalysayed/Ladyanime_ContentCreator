from __future__ import annotations

from pathlib import Path
from typing import List, Callable, Optional

from rich.console import Console
from rich.panel import Panel

from .config import AppConfig
from .ffmpeg_tools import cut_segments, concat_segments, write_segments_json
from .srt_parser import load_srt
from .segment_matcher import Segment, build_segments_from_subtitles

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
