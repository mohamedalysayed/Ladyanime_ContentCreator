from __future__ import annotations
import threading
import time
from .transcribe import transcribe_to_srt
import shutil
from pathlib import Path
from typing import Optional
import subprocess
import json
import gradio as gr
from .config import AppConfig
from .pipeline import run_mvp, run_shorts
from .segment_matcher import Segment
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


def _copy_to_inputs(
    video_file: str,
    srt_file: Optional[str],
) -> tuple[str, Optional[str]]:
    """
    Copy user-uploaded files into data/input with stable names expected by AppConfig.

    - Video is always required
    - Subtitles are OPTIONAL (shorts mode)
    - Preserves video extension (mkv/mp4/etc)
    """
    input_dir, _ = _ensure_dirs()

    # ---- video (always) ----
    vsrc = Path(video_file)
    vdst = input_dir / f"video{vsrc.suffix.lower()}"
    shutil.copy2(vsrc, vdst)

    # ---- subtitles (optional) ----
    if srt_file:
        ssrc = Path(srt_file)
        sdst = input_dir / "subtitles.srt"
        shutil.copy2(ssrc, sdst)
        return vdst.name, sdst.name

    # shorts mode → no subtitles
    return vdst.name, None

def clean_output_dir(output_dir: Path) -> None:
    """
    Remove old generated outputs before starting a new run.
    Safe: only touches known output artifacts.
    """
    if not output_dir.exists():
        return

    for item in output_dir.iterdir():
        if item.is_dir() and item.name in {
            "segments",
            "segments_labeled",
            "shorts",
        }:
            shutil.rmtree(item, ignore_errors=True)

        elif item.is_file() and item.suffix in {".mp4", ".json", ".txt"}:
            item.unlink(missing_ok=True)

def list_videos(dir_path: Path) -> list[str]:
    if not dir_path.exists():
        return []
    return [str(p) for p in sorted(dir_path.glob("*.mp4"))]

def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])

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

def run_transcription(
    video: str,
    model: str,
    language: Optional[str],
    progress=gr.Progress(track_tqdm=False),
):
    input_dir, _ = _ensure_dirs()

    video_path = Path(video)
    srt_path = input_dir / "transcription.srt"

    duration = get_video_duration(video_path)

    transcription_done = False
    transcription_text = ""

    def _worker():
        nonlocal transcription_done, transcription_text
        transcription_text = transcribe_to_srt(
            video_path=video_path,
            srt_out=srt_path,
            model_name=model,
            language=language or None,
        )
        transcription_done = True

    thread = threading.Thread(target=_worker)
    thread.start()

    # ---- smooth fake progress ----
    start = time.time()
    max_wait = duration * 0.9  # realistic cap

    while not transcription_done:
        elapsed = time.time() - start
        frac = min(elapsed / max_wait, 0.95)

        progress(
            0.15 + frac * 0.75,
            desc="🎧 Transcribing audio (AI is working)…",
        )
        time.sleep(0.25)

    thread.join()

    progress(1.0, desc="✅ Transcription completed")

    return str(srt_path), transcription_text

def _run_recap(
    video: Optional[str],
    subtitles: Optional[str],
    max_block_sec: float,
    silence_gap_sec: float,
    min_segment_sec: float,
    max_segments: int,
    clean_before_run: bool,
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
    
    #if mode == "recap" and not subtitles:
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

    if clean_before_run:
        clean_output_dir(cfg.output_dir)

    existing_segments = detect_existing_segments(cfg.output_dir)
    active_segment_files: list[Path] = []
    progress(0.12, desc="Analyzing subtitles & building segments…")

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
    if use_existing and existing_segments:
        active_segment_files = existing_segments
        out_video = cfg.output_dir / "recap.mp4"

        progress(0.20, desc=f"Using {len(active_segment_files)} existing segments")

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

        #if mode == "recap" and add_label and active_segment_files:
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

            # REPLACE original segments with labeled ones
            for f in (cfg.output_dir / "segments").glob("seg_*.mp4"):
                f.unlink()

            for f in labeled_segments:
                f.rename(cfg.output_dir / "segments" / f.name)

        # Apply label ONLY for recap (shorts already handled in run_shorts)
        if add_label:
        #if add_label and mode == "recap":
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

            out_video = labeled_out

    # ------------------------------------------------------------
    # Stage 3: Finalization (90–100%)
    # ------------------------------------------------------------
    progress(0.95, desc="Finalizing recap…")

    segments_json = cfg.output_dir / "segments.json"

    progress(1.0, desc="Done ✔")

    segments_dir = cfg.output_dir / "segments"
    segment_videos = list_videos(segments_dir)

    return str(out_video), str(segments_json), segment_videos

def _run_shorts_ui(
    video: Optional[str],
    clip_duration: int,
    max_shorts: int,
    clean_before_run: bool,
    add_label: bool,
    video_height: int,
    progress=gr.Progress(track_tqdm=False),
):
    if not video:
        raise gr.Error("Please upload a video.")

    progress(0.05, desc="Preparing files…")

    video_filename, _ = _copy_to_inputs(video, None)

    cfg = AppConfig(
        video_filename=video_filename,
        subtitles_filename=None,
        max_segments=int(max_shorts),
        output_basename="shorts",
    )

    if clean_before_run:
        clean_output_dir(cfg.output_dir)

    def on_shorts_progress(i: int, total: int):
        base = 0.10
        span = 0.80
        frac = base + span * (i / max(total, 1))
        progress(frac, desc=f"Creating short {i+1} / {total}")

    out_clip = run_shorts(
        video_path=cfg.video_path,
        out_dir=cfg.output_dir / "shorts",
        clip_duration=int(clip_duration),
        add_label=add_label,
        video_height=video_height,
        progress_cb=on_shorts_progress,
        max_shorts=int(max_shorts),
    )

    progress(1.0, desc="Done ✔")

    shorts_dir = cfg.output_dir / "shorts"
    shorts_videos = list_videos(shorts_dir)

    return str(out_clip), shorts_videos

# ----------------------------------------------------------------------
# UI definition
# ----------------------------------------------------------------------
def list_shorts_with_status():
    """
    Returns a table of shorts + upload status.
    """
    from tools.upload_shorts import _load_state, SHORTS_DIR

    state = _load_state()
    uploaded = set(state["uploaded_files"].keys())

    rows = []
    for p in sorted(SHORTS_DIR.glob("*.mp4")):
        status = "✅ Uploaded" if p.name in uploaded else "⏳ Pending"
        rows.append([p.name, status])

    return rows


def run_youtube_uploader(dry_run: bool = False):
    """
    Runs the uploader script and captures stdout.
    """
    cmd = ["python3", "tools/upload_shorts.py"]
    if dry_run:
        cmd.append("--dry-run")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output = []
    for line in proc.stdout:
        output.append(line.rstrip())

    return "\n".join(output)

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LadyAnime Video Engine") as demo:
        gr.Image(
            value=str(_project_root() / "data" / "assets" / "ladyAnime_banner.jpg"),
            show_label=False,
            container=False,
            height=260,
        )
        gr.Markdown("---")
        # gr.Markdown(
        #     "### 🎬 Automated Recaps & YouTube Shorts Engine",
        # )
        gr.Markdown(
            """
# LadyAnime Video Generation Engine

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
                        10, 120, value=20, step=1,
                        label="Target segment length (sec)",
                    )
                    silence_gap = gr.Slider(
                        1, 15, value=5, step=1,
                        label="Silence gap → new scene (sec)",
                    )
                    min_seg = gr.Slider(
                        3, 60, value=19, step=1,
                        label="Minimum segment length (sec)",
                    )
                    max_segs = gr.Slider(
                        1, 400, value=200, step=1,
                        label="Maximum number of segments",
                    )
                    clean_before_run = gr.Checkbox(
                        label="Clean previous outputs before generating",
                        value=True,
                    )
                    
                    use_existing = gr.Checkbox(
                        label="Use existing segments if available",
                        value=False,
                    )

                    add_label = gr.Checkbox(
                        label="Apply LadyAnime Shorts label overlay to shorts/recap",
                        value=True,
                    )

                    video_height = gr.Slider(
                        1000,
                        1400,
                        value=1200,
                        step=25,
                        label="Main video height (label visibility)",
                        info="Lower = more LadyAnime label visible (top & bottom)",
                    )

                run_btn = gr.Button("Generate Recap", variant="primary")

                with gr.Row():
                    out_video = gr.Video(label="Recap Preview")
                    out_json = gr.File(label="segments.json (debug / AI input)")

                segments_gallery = gr.Gallery(
                    label="Generated Segments / Shorts",
                    columns=3,
                    height=320,
                    preview=True,
                )

                run_btn.click(
                    fn=_run_recap,
                    inputs=[video_in, srt_in, max_block, silence_gap, min_seg, max_segs, clean_before_run, use_existing, add_label, video_height],
                    outputs=[out_video, out_json, segments_gallery],
                )

#                 gr.Markdown(
#                     """
# **Tip:**  
# Language does not matter (Spanish, Japanese, etc).  
# Segmentation is based on timestamps and silence gaps.
# """
#                )

            # ======================================================
            # TAB 2: YouTube Shorts
            # ======================================================
            with gr.Tab("YouTube Shorts"):

                with gr.Row():
                    shorts_video = gr.File(
                        label="Source video (.mp4 / .mkv)",
                        file_count="single",
                    )

                clip_duration = gr.Slider(
                    5, 60, value=30, step=5,
                    label="Clip duration (seconds)",
                )

                max_shorts = gr.Slider(
                    1, 100, value=10, step=1,
                    label="Maximum number of shorts",
                )

                clean_shorts = gr.Checkbox(
                    label="Clean previous shorts before generating",
                    value=True,
                )

                add_label_shorts = gr.Checkbox(
                    label="Apply LadyAnime label overlay",
                    value=True,
                )

                video_height_shorts = gr.Slider(
                    1000,
                    1400,
                    value=1200,
                    step=25,
                    label="Main video height (label visibility)",
                )

                run_shorts_btn = gr.Button("Generate Shorts", variant="primary")

                with gr.Row():
                    shorts_preview = gr.Video(label="Short Preview")

                shorts_gallery = gr.Gallery(
                    label="Generated Shorts",
                    columns=3,
                    height=320,
                    preview=True,
                )

                run_shorts_btn.click(
                    fn=_run_shorts_ui,
                    inputs=[
                        shorts_video,
                        clip_duration,
                        max_shorts,
                        clean_shorts,
                        add_label_shorts,
                        video_height_shorts,
                    ],
                    outputs=[
                        shorts_preview,
                        shorts_gallery,
                    ],
                )


            # ======================================================
            # TAB 3: Transcription
            # ======================================================
            with gr.Tab("Transcription"):

                gr.Markdown(
                    """
            ### 🎧 AI Transcription Engine
            High-quality speech-to-text powered by **Faster-Whisper**.

            • Multi-language support  
            • Clean `.srt` subtitles  
            • Full transcript preview  
            """
                )

                video_in = gr.File(
                    label="Video to transcribe",
                    file_count="single",
                )

                with gr.Row():
                    model = gr.Dropdown(
                        ["tiny", "base", "small", "medium"],
                        value="small",
                        label="Whisper model",
                    )

                    language = gr.Textbox(
                        label="Language (optional)",
                        placeholder="e.g. en, ja — leave empty for auto-detect",
                    )

                run_btn = gr.Button("📝 Transcribe", variant="primary")

                gr.Markdown("### 📄 Output")

                srt_out = gr.File(
                    label="Download subtitles (.srt)",
                    interactive=False,
                )

                transcript_text = gr.Textbox(
                    label="Full Transcription (select & copy)",
                    placeholder="Transcribed text will appear here…",
                    lines=16,
                    max_lines=24,
                )

                gr.Markdown(
                    "💡 Tip: You can select the text above and copy it directly."
                )

                run_btn.click(
                    fn=run_transcription,
                    inputs=[video_in, model, language],
                    outputs=[srt_out, transcript_text],
                )

            # ======================================================
            # TAB 4: AI Brain
            # ======================================================
            with gr.Tab("AI Brain"):
                gr.Markdown(
                    """
### Planned
- Read `segments.json`
- Rank best scenes
- Emotional / epic / viral tuning
"""
                )

            # ======================================================
            # TAB 5: YouTube Upload
            # ======================================================
            with gr.Tab("YouTube Uploads"):

                gr.Markdown(
                    """
### 📤 Automated YouTube Shorts Uploader

This tool uploads **2 shorts per day** to the **LadyAnime** channel.

• OAuth authenticated ✅  
• Smart scheduling (12:00 / 18:00)  
• Upload state tracking  
• Safe re-runs (no duplicates)  
"""
                )

                refresh_btn = gr.Button("🔄 Refresh Shorts Status")

                shorts_table = gr.Dataframe(
                    headers=["File", "Status"],
                    datatype=["str", "str"],
                    interactive=False,
                    row_count=(1, "dynamic"),
                )

                refresh_btn.click(
                    fn=list_shorts_with_status,
                    outputs=shorts_table,
                )

                gr.Markdown("---")

                with gr.Row():
                    upload_now_btn = gr.Button("🚀 Upload Now", variant="primary")
                    dry_run_btn = gr.Button("🧪 Dry Run (Preview Schedule)")

                uploader_log = gr.Textbox(
                    label="Uploader Log",
                    lines=14,
                    max_lines=20,
                    interactive=False,
                )

                upload_now_btn.click(
                    fn=lambda: run_youtube_uploader(dry_run=False),
                    outputs=uploader_log,
                )

                dry_run_btn.click(
                    fn=lambda: run_youtube_uploader(dry_run=True),
                    outputs=uploader_log,
                )

    return demo

def main():
    ui = build_ui()
    ui.launch()


if __name__ == "__main__":
    main()
