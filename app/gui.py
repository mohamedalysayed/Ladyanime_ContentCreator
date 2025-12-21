from __future__ import annotations

from .transcribe import transcribe_to_srt
import shutil
from pathlib import Path
from typing import Optional

import gradio as gr

from .config import AppConfig
from .pipeline import run_mvp
#from .segment_matcher import Segment
from .ffmpeg_tools import overlay_label_to_segments

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _project_root() -> Path:
    # app/ -> project root
    return Path(__file__).resolve().parents[1]


def _ensure_dirs() -> tuple[Path, Path]:
    root = _project_root()
    input_dir = root / "data" / "input"
    output_dir = root / "data" / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def _copy_to_inputs(video_file: str, srt_file: str) -> tuple[str, str]:
    """
    Copy user-uploaded files into data/input with stable names expected by AppConfig.
    We preserve the video extension (mkv/mp4/etc).
    """
    input_dir, _ = _ensure_dirs()

    vsrc = Path(video_file)
    ssrc = Path(srt_file)

    vdst = input_dir / f"video{vsrc.suffix.lower()}"
    sdst = input_dir / "subtitles.srt"

    shutil.copy2(vsrc, vdst)
    shutil.copy2(ssrc, sdst)

    return vdst.name, sdst.name


# ----------------------------------------------------------------------
# Gradio worker with GLOBAL progress bar
# ----------------------------------------------------------------------
def detect_existing_segments(output_dir: Path) -> list[Path]:
    """
    Detect already-created segment files.
    """
    seg_dir = output_dir / "segments"
    if not seg_dir.exists():
        return []

    return sorted(seg_dir.glob("seg_*.mp4"))

def _run_recap(
    video: Optional[str],
    subtitles: Optional[str],
    max_block_sec: float,
    silence_gap_sec: float,
    min_segment_sec: float,
    max_segments: int,
    use_existing: bool,
    add_label: bool,
    video_height: int,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Gradio handler: runs recap pipeline and returns output paths.
    Shows a single global progress bar.
    """

    if not video:
        raise gr.Error("Please upload an episode video (.mkv / .mp4).")
    
    if add_label and subtitles is None:
        raise gr.Error(
            "Subtitles are required for Recap. "
            "Transcription is disabled when creating labeled shorts."
        )

    if not subtitles:
        progress(0.07, desc="No subtitles provided — transcribing audio…")

        video_filename, _ = _copy_to_inputs(video, video)  # temp copy

        srt_path = (
            _project_root()
            / "data"
            / "input"
            / "subtitles.srt"
        )

        transcribe_to_srt(
            video_path=_project_root() / "data" / "input" / video_filename,
            srt_out=srt_path,
        )

        subtitles = str(srt_path)

    # ------------------------------------------------------------
    # Stage 1: Preparation (0–10%)
    # ------------------------------------------------------------
    progress(0.05, desc="Preparing files…")

    video_filename, srt_filename = _copy_to_inputs(video, subtitles)

    cfg = AppConfig(
        video_filename=video_filename,
        subtitles_filename=srt_filename,
        max_segments=int(max_segments),
        output_basename="recap",
    )
    
    existing_segments = detect_existing_segments(cfg.output_dir)
    active_segment_files: list[Path] = []
    progress(0.12, desc="Analyzing subtitles & building segments…")

    # ------------------------------------------------------------
    # Progress callback from pipeline (segment cutting)
    # ------------------------------------------------------------
    #def on_progress(i: int, total: int, seg: Segment):
        # Global progress allocation:
        # 10% → parsing + segmentation
        # 10–90% → cutting segments
    #    base = 0.10
    #    span = 0.80

    #    frac = base + span * (i / max(total, 1))

    #    progress(
    #        frac,
    #        desc=f"Cutting segment {i}/{total}  ({seg.start_s:.1f}s → {seg.end_s:.1f}s)",
    #    )

    def on_progress(i: int, total: int, seg: Segment):
        base = 0.25
        span = 0.65
        frac = base + span * (i / max(total, 1))

        progress(
            frac,
            desc=f"Creating segment {i} / {total}  ({seg.start_s:.1f}s → {seg.end_s:.1f}s)",
        )

    # ------------------------------------------------------------
    # Stage 2: Run pipeline
    # ------------------------------------------------------------
    #progress(0.10, desc="Analyzing subtitles…")

    if use_existing and existing_segments:
        active_segment_files = existing_segments
        out_video = cfg.output_dir / "recap.mp4"

        progress(
            0.20,
            desc=f"Using {len(active_segment_files)} existing segments",
        )

        progress(
            0.20,
            desc=f"Using {len(segments)} existing segments",
        )
    else:
        progress(0.15, desc="Generating segments…")

        out_video = run_mvp(
            cfg,
            progress_cb=on_progress,
            max_block_sec=max_block_sec,
            silence_gap_sec=silence_gap_sec,
            min_segment_sec=min_segment_sec,
        )

        active_segment_files = detect_existing_segments(cfg.output_dir)

        if add_label and active_segment_files:
            label_path = (
                _project_root()
                / "data"
                / "shorts_label"
                / "shortsLabel_LA.mp4"
            )

            labeled_dir = cfg.output_dir / "segments_labeled"
            labeled_dir.mkdir(exist_ok=True)

            progress(0.75, desc="Applying LadyAnime label to segments…")

            labeled_dir = cfg.output_dir / "segments_labeled"
            labeled_dir.mkdir(exist_ok=True)

            progress(0.75, desc="Applying LadyAnime label to segments…")

            labeled_segments = overlay_label_to_segments(
                segment_files=detect_existing_segments(cfg.output_dir),
                label_mp4=(
                    _project_root()
                    / "data"
                    / "shorts_label"
                    / "shortsLabel_LA.mp4"
                ),
                out_dir=labeled_dir,
                video_height=video_height,
            )

            # 🔥 REPLACE original segments with labeled ones
            for f in (cfg.output_dir / "segments").glob("seg_*.mp4"):
                f.unlink()

            for f in labeled_segments:
                f.rename(cfg.output_dir / "segments" / f.name)

        if add_label:
            from .ffmpeg_tools import overlay_label

            labeled_out = cfg.output_dir / "recap_labeled.mp4"

            overlay_label(
                video_in=Path(out_video),
                label_mp4=(
                    _project_root()
                    / "data"
                    / "shorts_label"
                    / "shortsLabel_LA.mp4"
                ),
                video_out=labeled_out,
                video_height=video_height,
            )

            out_video = labeled_out  # THIS IS CRITICAL


        from .ffmpeg_tools import overlay_label

    # ------------------------------------------------------------
    # Stage 3: Finalization (90–100%)
    # ------------------------------------------------------------
    progress(0.95, desc="Finalizing recap…")

    segments_json = cfg.output_dir / "segments.json"

    progress(1.0, desc="Done ✔")

    return str(out_video), str(segments_json)

# ----------------------------------------------------------------------
# UI definition
# ----------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LadyAnime Video Engine") as demo:
        gr.Markdown(
            """
# 🎬 LadyAnime Video Generation Engine

A multi-mode local tool for:
- Recaps
- YouTube Shorts
- Transcription
- AI-assisted scene selection

Starting with **Recap Generator**.
"""
        )

        with gr.Tabs():

            # ======================================================
            # TAB 1: Recap Generator
            # ======================================================
            with gr.Tab("Recap Generator"):

                with gr.Row():
                    video_in = gr.File(
                        label="Episode video (.mkv / .mp4)",
                        file_count="single",
                    )
                    srt_in = gr.File(
                        label="Subtitles (.srt)",
                        file_count="single",
                    )

                with gr.Accordion("Segment Settings", open=True):
                    max_block = gr.Slider(
                        10, 60, value=32, step=1,
                        label="Target segment length (sec)",
                    )
                    silence_gap = gr.Slider(
                        1, 15, value=5, step=1,
                        label="Silence gap → new scene (sec)",
                    )
                    min_seg = gr.Slider(
                        3, 30, value=18, step=1,
                        label="Minimum segment length (sec)",
                    )
                    max_segs = gr.Slider(
                        5, 120, value=40, step=1,
                        label="Maximum number of segments",
                    )
                    
                    use_existing = gr.Checkbox(
                        label="Use existing segments if available",
                        value=True,
                    )

                    add_label = gr.Checkbox(
                        label="Apply LadyAnime label overlay to segments",
                        value=False,
                    )

                    video_height = gr.Slider(
                        900,
                        1600,
                        value=1300,
                        step=50,
                        label="Main video height (label visibility)",
                        info="Lower = more LadyAnime label visible (top & bottom)",
                    )

                run_btn = gr.Button("🚀 Generate Recap", variant="primary")

                with gr.Row():
                    out_video = gr.Video(label="Recap Preview")
                    out_json = gr.File(label="segments.json (debug / AI input)")

                run_btn.click(
                    fn=_run_recap,
                    inputs=[
                        video_in,
                        srt_in,
                        max_block,
                        silence_gap,
                        min_seg,
                        max_segs,
                        use_existing,
                        add_label,
                        video_height,
                    ],
                    outputs=[out_video, out_json],
                )

                gr.Markdown(
                    """
**Tip:**  
Language does not matter (Spanish, Japanese, etc).  
Segmentation is based on timestamps and silence gaps.
"""
                )

            # ======================================================
            # TAB 2: YouTube Shorts (next)
            # ======================================================
            with gr.Tab("YouTube Shorts (Next)"):
                gr.Markdown(
                    """
### Planned
- 9:16 vertical shorts
- 15s / 30s / 60s
- Auto crop & zoom
- **LadyAnime label overlay (MP4 with transparency supported)**
"""
                )

            # ======================================================
            # TAB 3: Transcription
            # ======================================================
            with gr.Tab("Transcription (Whisper)"):
                gr.Markdown(
                    """
### Planned
- Video → SRT
- Language auto-detect
- Whisper models (tiny → small)
"""
                )

            # ======================================================
            # TAB 4: AI Brain
            # ======================================================
            with gr.Tab("AI Brain (Later)"):
                gr.Markdown(
                    """
### Planned
- Read `segments.json`
- Rank best scenes
- Emotional / epic / viral tuning
"""
                )

        gr.Markdown(
            """
---
### Notes
- Everything runs locally
- Output is written to `data/output/`
- This Gradio UI is intentionally temporary and flexible
"""
        )

    return demo

def main():
    ui = build_ui()
    ui.launch()


if __name__ == "__main__":
    main()
