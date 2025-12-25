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
from .pipeline import run_mvp, run_shorts, run_rhythmic_recap
from .segment_matcher import Segment
from .ffmpeg_tools import overlay_label_to_segments


class ProgressController:
    def __init__(self, progress: gr.Progress, prefix: str = ""):
        self.progress = progress
        self.prefix = prefix

    def set(self, frac: float, msg: str):
        if STOP_EVENT.is_set():
            raise gr.Error("🛑 Processing stopped by user.")
        self.progress(max(0.0, min(frac, 1.0)), desc=f"{self.prefix}{msg}")

    def map(self, base: float, span: float, frac: float, msg: str):
        self.set(base + span * frac, msg)

# ----------------------------------------------------------------------
# Global stop signal (thread-safe)
# ----------------------------------------------------------------------
STOP_EVENT = threading.Event()
def guarded_progress_cb(progress, prefix: str = ""):
    def _cb(frac: float, msg: str):
        if STOP_EVENT.is_set():
            raise gr.Error("🛑 Processing stopped by user.")
        progress_call(progress, frac, f"{prefix}{msg}")
    return _cb

def progress_call(progress_fn, frac: float, msg: str):
    """
    Call progress function safely regardless of whether it expects:
    - progress(frac, desc="...")
    OR
    - progress(frac, "...")
    """
    try:
        progress_fn(frac, desc=msg)
    except TypeError:
        progress_fn(frac, msg)

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

def render_segments_with_separators(items):
    placeholder = str(_project_root() / "data/assets/blank.jpg")
    rendered = []

    for it in items:
        if isinstance(it, str) and it.startswith("__EPISODE__::"):
            rendered.append((placeholder, it.replace("__EPISODE__::", "")))
        else:
            rendered.append(it)

    return rendered

def concat_videos_ffmpeg(videos: list[Path], output: Path):
    txt = output.parent / "concat_list.txt"
    with txt.open("w") as f:
        for v in videos:
            f.write(f"file '{v.as_posix()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(txt),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)

def speed_up_video(input_v: Path, factor: float) -> Path:
    if factor == 1.0:
        return input_v

    out = input_v.with_name(input_v.stem + f"_x{factor}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_v),
        "-filter_complex",
        f"[0:v]setpts={1/factor}*PTS[v];[0:a]atempo={factor}[a]",
        "-map", "[v]",
        "-map", "[a]",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out

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
    #progress=gr.Progress(track_tqdm=False),
    progress=None,
):
    if progress is None:
        def progress(*args, **kwargs):
            pass
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
        if STOP_EVENT.is_set():
            progress(1.0, desc="🛑 Transcription stopped")
            return None, ""
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

def _process_single_video(
    video: str,
    subtitles: Optional[str],
    base_output_dir: Path,
    mode: str,
    intro_skip: float,
    outro_skip: float,
    keep_sec: float,
    skip_sec: float,
    concat_final: bool,
    speed_factor: float, 
    max_block_sec: float,
    silence_gap_sec: float,
    min_segment_sec: float,
    max_segments: int,
    clean_before_run: bool,
    use_existing: bool,
    add_label: bool,
    video_height: int,
    progress,
):
    RHYTHMIC_MODE = "Rhythmic (2s in / 2s out)"
    video_name = Path(video).stem
    safe_name = video_name.replace(" ", "_")

    #progress(0.0, desc=f"📼 Processing {video_name}")
    progress_call(progress, 0.0, f"📼 Processing {video_name}")

    if STOP_EVENT.is_set():
        raise gr.Error("🛑 Processing stopped by user.")

    # ---------- rhythmic ----------
    if mode == RHYTHMIC_MODE:
        video_filename, _ = _copy_to_inputs(video, None)

        # cfg = AppConfig(
        #     video_filename=video_filename,
        #     subtitles_filename=None,
        #     max_segments=9999,
        #     output_basename="recap",
        #     output_subdir=safe_name,
        # )

        cfg = AppConfig(
            video_filename=video_filename,
            subtitles_filename=None,
            max_segments=9999,
            output_basename=f"{safe_name}_recap",
            output_subdir=safe_name,
        )

        out = run_rhythmic_recap(
            config=cfg,
            intro_skip_sec=intro_skip,
            outro_skip_sec=outro_skip,
            keep_sec=keep_sec,
            skip_sec=skip_sec,
            speed_factor=speed_factor,
            emit_recap=concat_final,
            progress_cb=guarded_progress_cb(
                progress,
                prefix=f"{video_name}: "
            ),
        )

        return cfg, out

    # ---------- subtitle-based ----------
    if not subtitles:
        video_filename, _ = _copy_to_inputs(video, None)
        srt_path = _project_root() / "data" / "input" / "subtitles.srt"
        transcribe_to_srt(
            video_path=_project_root() / "data" / "input" / video_filename,
            srt_out=srt_path,
        )
        subtitles = str(srt_path)

    video_filename, srt_filename = _copy_to_inputs(video, subtitles)

    # cfg = AppConfig(
    #     video_filename=video_filename,
    #     subtitles_filename=srt_filename,
    #     max_segments=int(max_segments),
    #     output_basename="recap",
    #     output_subdir=safe_name,
    #     output_root=base_output_dir,   # 👈 NEW
    # )

    cfg = AppConfig(
        video_filename=video_filename,
        subtitles_filename=srt_filename,
        max_segments=int(max_segments),
        output_basename=f"{safe_name}_recap",
        output_subdir=safe_name,
        output_root=base_output_dir,
    )

    if clean_before_run:
        clean_output_dir(cfg.output_dir)

    def guarded_progress(i: int, t: int, s: Segment):
        if STOP_EVENT.is_set():
            raise gr.Error("🛑 Processing stopped by user.")
        #progress(i / max(t, 1), f"{video_name}: segment {i}/{t}")
        progress_call(progress, i / max(t, 1), f"{video_name}: segment {i}/{t}")

    # Reuse existing segments if requested
    if use_existing:
        existing = detect_existing_segments(cfg.output_dir)
        if existing:
            progress_call(
                progress,
                0.05,
                f"{video_name}: using existing {len(existing)} segments",
            )

            # If concat_final → reuse existing recap if available
            recap = cfg.output_dir / f"{cfg.output_basename}.mp4"
            if recap.exists():
                return cfg, recap

            # Otherwise, just reuse segments (viewer still works)
            return cfg, None

    out_video = run_mvp(
        cfg,
        progress_cb=guarded_progress,
        max_block_sec=max_block_sec,
        silence_gap_sec=silence_gap_sec,
        min_segment_sec=min_segment_sec,
    )

    if add_label and out_video:
        from .ffmpeg_tools import overlay_label
        labeled = cfg.output_dir / "recap_labeled.mp4"
        overlay_label(out_video, _project_root() / "data/shorts_label/shortsLabel_LA.mp4", labeled, video_height)
        out_video = labeled

    return cfg, out_video

def _run_recap(
    video,
    subtitles,
    output_dir,
    mode,
    intro_skip,
    outro_skip,
    keep_sec,
    skip_sec,
    concat_final,
    agglomerate_season,
    speed_factor, 
    max_block_sec,
    silence_gap_sec,
    min_segment_sec,
    max_segments,
    clean_before_run,
    use_existing,
    add_label,
    video_height,
    progress=gr.Progress(track_tqdm=False),
):
    from pathlib import Path
    if output_dir:
        base_output_dir = Path(output_dir)
    else:
        base_output_dir = _project_root() / "data" / "output"

    all_recaps = []

    STOP_EVENT.clear()

    pc = ProgressController(progress)
    pc.set(0.02, "Preparing job…")

    if not video or (isinstance(video, list) and len(video) == 0):
        raise gr.Error(
            "🚫 No videos selected.\n\n"
            "Please upload one or more episode videos (.mp4 or .mkv) "
            "before clicking **Generate Recap**."
        )

    videos = video if isinstance(video, list) else [video]
    total = len(videos)
    per_video_span = 0.80 / total   # progress range: 15% → 95%

    last_out_video = None
    last_segments = []

    for idx, v in enumerate(videos, start=1):

        if STOP_EVENT.is_set():
            progress(1.0, desc="🛑 Processing stopped")
            #return None, None, []
            return None, []
        
        video_base = 0.15 + (idx - 1) * per_video_span
        #pc.set(video_base, f"Processing {Path(v).stem}")
        episode_label = f"S1 · Episode {idx}/{total}"
        episode_name = Path(v).stem

        pc.set(
            video_base,
            f"{episode_label} — {episode_name}"
        )
        
        def on_segment(frac, msg):
            pc.map(
                base=video_base,
                span=per_video_span * 0.85,
                frac=frac,
                msg=f"{episode_label} — {msg}",
            )

        cfg, out_video = _process_single_video(
            v,
            subtitles,
            base_output_dir,
            mode,
            intro_skip,
            outro_skip,
            keep_sec,
            skip_sec,
            concat_final,
            speed_factor,
            max_block_sec,
            silence_gap_sec,
            min_segment_sec,
            max_segments,
            clean_before_run,
            use_existing,
            add_label,
            video_height,
            on_segment,
        )

        if out_video:
            all_recaps.append(Path(out_video))

        last_out_video = (
            str(out_video) if out_video and Path(out_video).is_file() else None
        )

        json_path = cfg.output_dir / "segments.json"
        last_json = str(json_path) if json_path.exists() else None

        #last_segments = list_videos(cfg.output_dir / "segments")
        gallery_label = f"🧩 {Path(v).stem}"
        last_segments.append(f"__EPISODE__::{gallery_label}")
        last_segments.extend(list_videos(cfg.output_dir / "segments"))

    #progress(1.0, desc="✅ All videos processed")
    pc.set(1.0, "✅ All videos processed")

    if agglomerate_season and len(all_recaps) > 1:
        season_out = base_output_dir / "Season_Recap.mp4"
        concat_videos_ffmpeg(all_recaps, season_out)
        last_out_video = season_out

    # Apply speed ONCE, at the very end
    # if last_out_video and speed_factor != 1.0:
    #     last_out_video = str(
    #         speed_up_video(Path(last_out_video), speed_factor)
    #     )

    if mode != "Rhythmic (2s in / 2s out)" and last_out_video and speed_factor != 1.0:
        last_out_video = str(
            speed_up_video(Path(last_out_video), speed_factor)
        )

    #return last_out_video, last_segments
    return last_out_video, render_segments_with_separators(last_segments)    

def _run_shorts_ui(
    video: Optional[str],
    clip_duration: int,
    max_shorts: int,
    clean_before_run: bool,
    add_label: bool,
    video_height: int,
    progress=None,
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

def _read_uploaded_json_pretty() -> dict:
    from tools.upload_shorts import STATE_FILE
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}

def _pending_and_uploaded_tables():
    from tools.upload_shorts import _load_state, SHORTS_DIR
    state = _load_state()
    uploaded = set(state["uploaded_files"].keys())

    pending = []
    uploaded_rows = []

    for p in sorted(SHORTS_DIR.glob("*.mp4")):
        if p.name in uploaded:
            meta = state["uploaded_files"][p.name]
            uploaded_rows.append([p.name, meta.get("title",""), meta.get("publish_at",""), meta.get("video_id","")])
        else:
            # title default = base_title (can be edited in UI)
            pending.append([p.name, state.get("base_title","")])

    return pending, uploaded_rows, state.get("base_title","")

def _compute_fresh_until_label(pending_count: int) -> str:
    from tools.upload_shorts import fresh_until_date
    if pending_count <= 0:
        return "Fresh-until: (no pending videos)"
    dt = fresh_until_date(pending_count)
    if not dt:
        return "Fresh-until: (n/a)"
    # show date clearly in Zurich time
    return f"Fresh-until: {dt.strftime('%Y-%m-%d %H:%M')} Europe/Zurich"

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
    with gr.Blocks(
        title="LadyAnime Video Engine",
        css="""
        /* Horizontal scrolling gallery for segments */
        #segments-gallery {
            overflow-x: auto;
            white-space: nowrap;
        }

        #segments-gallery .gallery-item {
            display: inline-block;
            width: 240px;
            margin-right: 10px;
        }
        """
    ) as demo:
        gr.Image(
            value=str(_project_root() / "data" / "assets" / "ladyAnime_banner.jpg"),
            show_label=False,
            container=False,
            height=260,
        )
        gr.Markdown("---")

        with gr.Row():
            # LEFT COLUMN — Main Engine
            with gr.Column(scale=3):
                gr.Markdown(
                    """
        # LadyAnime Video Generation Engine

        A multi-mode local tool for:
        - 🎬 **Recaps**
        - 📱 **YouTube Shorts**
        - 🎧 **Transcription**
        - 🤖 **AI-assisted scene selection**

        Starting with **Recap Generator** below ⬇️
        """
                )

            # RIGHT COLUMN — Companion Tool
            with gr.Column(scale=2):
                gr.Markdown(
                    """
        ## AI Companion Tool 

        🧠 **Summaries & Text-to-Speech Generator**

        A lightweight AI assistant for:
        - Generate summaries
        - Voice-over generation
        - Script preparation

        👉 **[Open Tool HERE ↗](https://lady-anime-agent-front-end-vwa5.vercel.app/)**
        """
                )

        with gr.Tabs():

            # ======================================================
            # TAB 1: Recap Generator
            # ======================================================
            with gr.Tab("Recap Generator"):

                mode = gr.Radio(
                    ["Subtitle-based", "Rhythmic (2s in / 2s out)"],
                    value="Rhythmic (2s in / 2s out)",
                    label="Recap Mode",
                )

                with gr.Row():
                    video_in = gr.File(
                        label="Episode videos (.mkv / .mp4)",
                        file_count="multiple",
                    )
                    srt_in = gr.File(
                        label="Subtitles (.srt)",
                        file_count="single",
                    )

                def _toggle_recap_mode(selected_mode):
                    is_subtitle = selected_mode == "Subtitle-based"
                    is_rhythmic = not is_subtitle

                    return (
                        gr.update(visible=is_subtitle),   # srt_in
                        gr.update(visible=is_rhythmic),   # rhythmic accordion
                        gr.update(visible=is_subtitle),   # subtitle accordion
                    )

                with gr.Accordion("Rhythmic Recap Settings", open=False, visible=False) as rhythmic_accordion:
                    intro_skip = gr.Slider(
                        0, 300, value=90, step=1,
                        label="Intro skip (seconds)",
                        info="Skip opening / recap / OP",
                    )
                    outro_skip = gr.Slider(
                        0, 300, value=60, step=1,
                        label="Outro skip (seconds)",
                        info="Skip ending / ED / preview",
                    )

                    keep_sec = gr.Slider(
                        0.5, 10, value=2, step=0.5,
                        label="Keep duration (seconds)",
                    )

                    skip_sec = gr.Slider(
                        0.5, 10, value=2, step=0.5,
                        label="Skip duration (seconds)",
                    )

                    concat_final = gr.Checkbox(
                        label="Concatenate segments into final recap video",
                        value=True,
                    )

                    agglomerate_season = gr.Checkbox(
                        label="Merge all episode recaps into ONE season recap",
                        value=False,
                    )

                    speed_factor = gr.Slider(
                        0.75, 2.0,
                        value=1.0,
                        step=0.25,
                        label="Playback speed",
                        info="Applied during rhythmic cutting (not post-processed)",
                    )

                with gr.Accordion("Segment Settings", open=False) as subtitle_controls:
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

                mode.change(
                    fn=_toggle_recap_mode,
                    inputs=mode,
                    outputs=[
                        srt_in,
                        rhythmic_accordion,
                        subtitle_controls,
                    ],
                )
                # Force correct visibility on initial load
                demo.load(
                    fn=lambda: _toggle_recap_mode("Rhythmic (2s in / 2s out)"),
                    outputs=[srt_in, rhythmic_accordion, subtitle_controls],
                )

                output_dir_picker = gr.Textbox(
                    label="Output folder (optional)",
                    placeholder="Leave empty to use default: data/output",
                )

                with gr.Row():
                    run_btn = gr.Button("Generate", variant="primary")
                    stop_btn = gr.Button("🛑 Stop", variant="secondary")
    
                def _stop_processing():
                    STOP_EVENT.set()
                    return "🛑 Processing stopped by user."

                stop_btn.click(
                    fn=_stop_processing,
                    outputs=[],
                )

                with gr.Row():
                    out_video = gr.Video(label="Recap Preview")

                segments_gallery = gr.Gallery(
                    label="Generated Segments / Shorts",
                    elem_id="segments-gallery",
                    columns=999,     # force single row
                    height=260,
                    preview=True,
                )

                run_btn.click(
                    fn=_run_recap,

                    inputs=[
                        video_in,
                        srt_in,
                        output_dir_picker,
                        mode,
                        intro_skip,
                        outro_skip,
                        keep_sec,
                        skip_sec,
                        concat_final,
                        agglomerate_season,
                        speed_factor, 
                        max_block,
                        silence_gap,
                        min_seg,
                        max_segs,
                        clean_before_run,
                        use_existing,
                        add_label,
                        video_height,
                    ],
                    outputs=[out_video, segments_gallery],
                )

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
            # TAB 4: YouTube Upload
            # ======================================================
            with gr.Tab("YouTube Uploads"):

                gr.Markdown(
                    """
            ### 📤 LadyAnime Shorts Uploader (Safe Mode)

            ✅ It allows your to select videos to upload  
            ✅ Schedules **2/day at 00:00 + 12:00 (Europe/Zurich)**  
            ✅ State tracking (no duplicates)  
            ✅ Atomic saves + retry handling  
            """
                )

                # ---- top controls ----
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Lists", variant="secondary")
                    view_json_btn = gr.Button("📄 View uploaded.json", variant="secondary")

                fresh_until_lbl = gr.Markdown("Fresh-until: (refresh to compute)")

                base_title = gr.Textbox(
                    label="Base Title (applied to all shorts unless edited per-file)",
                    placeholder="e.g. Frieren Episode 1 — Best Moments",
                )

                description_box = gr.Textbox(
                    label="Default Description",
                    lines=3,
                    value="#ladyAnime #anime #Shorts\n",
                )

                tags_box = gr.Textbox(
                    label="Tags (comma-separated)",
                    value="LadyAnime, anime, Shorts",
                )

                privacy_box = gr.Radio(
                    ["private", "unlisted", "public"],
                    value="public",
                    label="Video Privacy",
                )

                gr.Markdown(
                    "⚠️ Recommendation: You can also keep uploads **Private** and switch to Public after review."
                )

                # ---- Pending + Uploaded ----
                with gr.Row():
                    pending_df = gr.Dataframe(
                        headers=["File", "Title (editable)"],
                        datatype=["str", "str"],
                        row_count=(1, "dynamic"),
                        interactive=True,
                        label="Pending (select via checkbox list below)",
                    )

                    uploaded_df = gr.Dataframe(
                        headers=["File", "Title", "publishAt", "video_id"],
                        datatype=["str", "str", "str", "str"],
                        row_count=(1, "dynamic"),
                        interactive=False,
                        label="Uploaded / Scheduled",
                    )

                pending_select = gr.CheckboxGroup(label="Select pending shorts to upload", choices=[])

                with gr.Row():
                    # Left: safe preview
                    with gr.Column(scale=1, min_width=160):
                        dry_run_btn = gr.Button(
                            "🧪 Dry Run (Preview)",
                            variant="secondary",
                        )

                    # Right: upload actions grouped
                    with gr.Column(scale=2):
                        with gr.Row():
                            upload_all_btn = gr.Button(
                                "📅 Upload ALL Remaining",
                                variant="secondary",
                            )
                            upload_selected_btn = gr.Button(
                                "🚀 Upload Selected",
                                variant="primary",
                                interactive=False,  
                            )
                def _toggle_upload_selected(selected):
                    return gr.update(interactive=bool(selected))

                pending_select.change(
                    fn=_toggle_upload_selected,
                    inputs=pending_select,
                    outputs=upload_selected_btn,
                )

                uploader_log = gr.Textbox(label="Uploader Log", lines=14, max_lines=20, interactive=False)

                # JSON viewer
                state_json = gr.JSON(label="uploaded.json (live view)")

                def _refresh_ui():
                    pending, uploaded_rows, bt = _pending_and_uploaded_tables()
                    pending_files = [row[0] for row in pending]
                    fresh = _compute_fresh_until_label(len(pending_files))

                    return (
                        pending,
                        uploaded_rows,
                        gr.update(choices=pending_files, value=[]),
                        fresh,
                        bt,
                    )
                
                refresh_btn.click(
                    fn=_refresh_ui,
                    outputs=[pending_df, uploaded_df, pending_select, fresh_until_lbl, base_title],
                )

                def _view_json():
                    return _read_uploaded_json_pretty()

                view_json_btn.click(fn=_view_json, outputs=state_json)

                # ---- Upload handlers with progress bar support ----
                def _parse_tags(s: str) -> list[str]:
                    return [t.strip() for t in (s or "").split(",") if t.strip()]
                
                def _collect_per_file_titles(pending_table) -> dict[str, str]:
                    """
                    pending_table can be:
                    - pandas.DataFrame (Gradio Dataframe output)
                    - list[list[Any]] (sometimes, depending on gradio version)
                    - None
                    Returns: {filename: title}
                    """
                    out: dict[str, str] = {}

                    if pending_table is None:
                        return out

                    # If it's a pandas DataFrame
                    try:
                        import pandas as pd  # local import is fine
                        if isinstance(pending_table, pd.DataFrame):
                            if pending_table.empty:
                                return out
                            rows = pending_table.values.tolist()
                        else:
                            rows = pending_table
                    except Exception:
                        rows = pending_table

                    # If it's still not a list, bail safely
                    if not isinstance(rows, list):
                        return out

                    for row in rows:
                        if not row or len(row) < 1:
                            continue
                        filename = str(row[0]).strip() if row[0] is not None else ""
                        if not filename:
                            continue
                        title = ""
                        if len(row) >= 2 and row[1] is not None:
                            title = str(row[1]).strip()
                        out[filename] = title

                    return out

                def _upload(selected, pending_table, bt, desc, tags, privacy, schedule_all, dry_run, progress=gr.Progress(track_tqdm=False)):

                    from tools.upload_shorts import upload_many

                    if (not schedule_all) and (not selected) and (not dry_run):
                        return "⚠️ No videos selected. Select shorts from the checkbox list first.", *(_refresh_ui()[0:2]), _refresh_ui()[3], _refresh_ui()[4], _read_uploaded_json_pretty()

                    per_file_titles = _collect_per_file_titles(pending_table)

                    def cb(frac: float, msg: str):
                        # smooth progress illusion
                        visual_frac = 0.1 + 0.8 * frac
                        progress(visual_frac, desc=msg)

                    res = upload_many(
                        selected_files=selected if not schedule_all else None,
                        schedule_all_remaining=schedule_all,
                        base_title=bt or "",
                        per_file_titles=per_file_titles,
                        description=desc or "#ladyAnime #anime #Shorts\n",
                        tags=_parse_tags(tags),
                        privacy_status=privacy,
                        made_for_kids=False,
                        dry_run=dry_run,
                        progress_cb=cb,
                    )

                    # refresh after upload
                    pending, uploaded_rows, _, fresh, bt2 = _refresh_ui()
                    state_view = _read_uploaded_json_pretty()

                    return res["log"], pending, uploaded_rows, fresh, bt2, state_view
                
                def dry_run_handler(selected, pending_table, bt, desc, tags, privacy, progress=gr.Progress(track_tqdm=False)):
                    return _upload(
                        selected=selected,
                        pending_table=pending_table,
                        bt=bt,
                        desc=desc,
                        tags=tags,
                        privacy=privacy,
                        schedule_all=(not selected),
                        dry_run=True,
                        progress=progress,
                    )

                upload_selected_btn.click(
                    fn=lambda selected, pending_table, bt, desc, tags, privacy, progress=gr.Progress(track_tqdm=False):
                        _upload(
                            selected,
                            pending_table,
                            bt,
                            desc,
                            tags,
                            privacy,
                            schedule_all=False,
                            dry_run=False,
                            progress=progress,
                        ),
                    inputs=[pending_select, pending_df, base_title, description_box, tags_box, privacy_box],
                    outputs=[uploader_log, pending_df, uploaded_df, fresh_until_lbl, base_title, state_json],
                )

                upload_all_btn.click(
                    fn=lambda pending_table, bt, desc, tags, privacy, progress=gr.Progress(track_tqdm=False):
                        _upload(
                            [],
                            pending_table,
                            bt,
                            desc,
                            tags,
                            privacy,
                            schedule_all=True,
                            dry_run=False,
                            progress=progress,
                        ),
                    inputs=[pending_df, base_title, description_box, tags_box, privacy_box],
                    outputs=[uploader_log, pending_df, uploaded_df, fresh_until_lbl, base_title, state_json],
                )

                dry_run_btn.click(
                    fn=dry_run_handler,
                    inputs=[pending_select, pending_df, base_title, description_box, tags_box, privacy_box],
                    outputs=[uploader_log, pending_df, uploaded_df, fresh_until_lbl, base_title, state_json],
                )

            # ======================================================
            # TAB 5: AI Brain
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
    return demo

def main():
    ui = build_ui()
    ui.launch()

if __name__ == "__main__":
    main()
