
import gradio as gr
import json
import subprocess
from pathlib import Path

from .ui_handlers import (
    _run_recap,
    _run_shorts_ui,
    run_transcription,
)
from .ui_progress import STOP_EVENT
from .ui_helpers import project_root, split_narration_text

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
            value=str(project_root() / "data" / "assets" / "ladyAnime_banner.jpg"),
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

        **Summaries & Text-to-Speech Generator**

        A lightweight AI assistant for:
        - Generate summaries
        - Voice-over generation
        - Script preparation

        **[Open Tool HERE ↗](https://lady-anime-agent-front-end-vwa5.vercel.app/)**
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
                        0, 500, value=90, step=1,
                        label="Intro skip (seconds)",
                        info="Skip opening / recap / OP",
                    )
                    outro_skip = gr.Slider(
                        0, 500, value=60, step=1,
                        label="Outro skip (seconds)",
                        info="Skip ending / ED / preview",
                    )

                    keep_sec = gr.Slider(
                        0.5, 100, value=2, step=0.5,
                        label="Keep duration (seconds)",
                    )

                    skip_sec = gr.Slider(
                        0.5, 100, value=2, step=0.5,
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
                        0.5, 2.0,
                        value=1.0,
                        step=0.25,
                        label="Playback speed",
                        info="Applied during rhythmic cutting (not post-processed)",
                    )

                    ai_narration = gr.Checkbox(
                        label="Generate AI Narrated Recap",
                        value=False,
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

                # narration_text_box = gr.Textbox(
                #     label="AI Narration — Original vs Summary",
                #     lines=20,
                #     visible=False,
                # )

                narration_text_hidden = gr.Textbox(
                    visible=False,
                    interactive=False,
                )

                with gr.Row():
                    original_box = gr.Textbox(
                        label="📝 Original Dialogue (Extracted)",
                        lines=18,
                        interactive=False,
                    )

                    summary_box = gr.Textbox(
                        label="🎙 AI Narration (Summary)",
                        lines=18,
                        interactive=False,
                    )


                # def _toggle_narration_panel(flag):
                #     return gr.update(visible=flag)

                # ai_narration.change(
                #     fn=_toggle_narration_panel,
                #     inputs=ai_narration,
                #     outputs=narration_text_box,
                # )
                def _toggle_narration_panel(flag):
                    return gr.update(visible=flag)

                ai_narration.change(
                    fn=_toggle_narration_panel,
                    inputs=ai_narration,
                    outputs=narration_text_hidden,
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
                        ai_narration,
                    ],
                    outputs=[out_video, segments_gallery, narration_text_hidden],
                ).then(
                    split_narration_text,
                    inputs=narration_text_hidden,
                    outputs=[original_box, summary_box],
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
    ui.queue()
    ui.launch(
        max_threads=1
    )

if __name__ == "__main__":
    main()
