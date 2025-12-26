
import json
import gradio as gr
from pathlib import Path
from typing import Optional
import threading
import time
from .transcribe import transcribe_to_srt
from .config import AppConfig
from .pipeline import run_mvp, run_shorts, run_rhythmic_recap
from .segment_matcher import Segment
from .ui_progress import ProgressController
from .ui_helpers import *
from .ui_progress import *
from app.narration.run_narrated_recap import run_ai_narrated_recap

# ----------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------
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
    input_dir, _ = ensure_dirs()

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

def detect_existing_segments(output_dir: Path) -> list[Path]:
    """
    Detect already-created segment files.
    """
    seg_dir = output_dir / "segments"
    if not seg_dir.exists():
        return []

    return sorted(seg_dir.glob("seg_*.mp4"))

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
        video_filename, _ = copy_to_inputs(video, None)

        cfg = AppConfig(
            video_filename=video_filename,
            subtitles_filename=None,
            max_segments=9999,
            output_basename=f"{safe_name}_recap",
            output_subdir=safe_name,
        )

        out, rhythmic_timeline = run_rhythmic_recap(
            config=cfg,
            intro_skip_sec=intro_skip,
            outro_skip_sec=outro_skip,
            keep_sec=keep_sec,
            skip_sec=skip_sec,
            speed_factor=speed_factor,
            emit_recap=concat_final,
            return_timeline=True,   # 👈 NEW
            progress_cb=guarded_progress_cb(
                progress,
                prefix=f"{video_name}: "
            ),
        )

        # 🔹 WRITE segments.json for narration
        segments_json = cfg.output_dir / "segments.json"
        segments_json.write_text(
            json.dumps(rhythmic_timeline, indent=2),
            encoding="utf-8",
        )

        return cfg, out


    # ---------- subtitle-based ----------
    if not subtitles:
        video_filename, _ = copy_to_inputs(video, None)
        srt_path = project_root() / "data" / "input" / "subtitles.srt"
        transcribe_to_srt(
            video_path=project_root() / "data" / "input" / video_filename,
            srt_out=srt_path,
        )
        subtitles = str(srt_path)

    video_filename, srt_filename = copy_to_inputs(video, subtitles)

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
        overlay_label(out_video, project_root() / "data/shorts_label/shortsLabel_LA.mp4", labeled, video_height)
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
    ai_narration,   
    progress=gr.Progress(track_tqdm=False),
):
    from pathlib import Path
    if output_dir:
        base_output_dir = Path(output_dir)
    else:
        base_output_dir = project_root() / "data" / "output"

    all_recaps = []
    narration_text = ""
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
            return None, [], ""
        
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

        # --------------------------------------------------
        # AUTO / REUSED TRANSCRIPTION (per episode)
        # --------------------------------------------------
        if ai_narration and not subtitles:
            episode_dir = base_output_dir / Path(v).stem
            episode_dir.mkdir(parents=True, exist_ok=True)

            srt_out = episode_dir / "episode_full.srt"

            if srt_out.exists():
                # ✅ Reuse existing SRT
                subtitles = str(srt_out)
                pc.set(video_base, "📄 Reusing existing episode transcript")

            else:
                # ❌ No SRT yet → transcribe once
                pc.set(video_base, "🎧 Transcribing episode for AI narration…")

                transcribe_to_srt(
                    video_path=Path(v),
                    srt_out=srt_out,
                    model_name="small",
                    language=None,
                )

                subtitles = str(srt_out)
                pc.set(video_base + 0.05, "✅ Transcription ready")


        # # --------------------------------------------------
        # # Decide which SRT to use for THIS episode narration
        # # --------------------------------------------------
        # episode_srt_path: Optional[str] = subtitles  # do NOT overwrite `subtitles` globally

        # if ai_narration:
        #     if episode_srt_path:
        #         pc.set(video_base, "📄 Using provided subtitles for AI narration")
        #     else:
        #         pc.set(video_base, "🎧 Transcribing episode for AI narration…")

        #         episode_dir = base_output_dir / Path(v).stem
        #         episode_dir.mkdir(parents=True, exist_ok=True)

        #         srt_out = episode_dir / "episode_full.srt"

        #         transcribe_to_srt(
        #             video_path=Path(v),
        #             srt_out=srt_out,
        #             model_name="small",
        #             language=None,  # auto-detect English/Spanish etc.
        #         )

        #         episode_srt_path = str(srt_out)
        #         pc.set(video_base + 0.05, "✅ Transcription ready")

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

        # --------------------------------------------------
        # AI NARRATED RECAP PIPELINE
        # --------------------------------------------------
        if ai_narration and last_out_video:
            segments_json = cfg.output_dir / "segments.json"

            if not segments_json.exists():
                raise gr.Error(
                    "AI narration failed.\n\n"
                    "segments.json was not generated.\n"
                    "This usually means segmentation did not run."
                )

            pc.set(video_base + per_video_span * 0.9, "🎙️ Generating AI narration…")

            narrated_video, narration_text = run_ai_narrated_recap(
                recap_video=Path(last_out_video),
                episode_video=Path(v),
                episode_srt=Path(subtitles),
                segments_json=segments_json,
                out_dir=cfg.output_dir / "narrated",
            )

            last_out_video = str(narrated_video)

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

    MAX_GALLERY_ITEMS = 80
    last_segments = last_segments[-MAX_GALLERY_ITEMS:]

    #return last_out_video, last_segments
    return (
        last_out_video,
        render_segments_with_separators(last_segments),
        narration_text,
    )

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

    video_filename, _ = copy_to_inputs(video, None)

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