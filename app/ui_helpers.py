
# ui_helpers.py
from pathlib import Path
import shutil
import subprocess
import json
from typing import Optional


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

# def _project_root() -> Path:
#     # app/ -> project root
#     return Path(__file__).resolve().parents[1]

def ensure_dirs() -> tuple[Path, Path]:
    root = project_root()
    input_dir = root / "data" / "input"
    output_dir = root / "data" / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir

def copy_to_inputs(
    video_file: str,
    srt_file: Optional[str],
) -> tuple[str, Optional[str]]:
    """
    Copy user-uploaded files into data/input with stable names expected by AppConfig.

    - Video is always required
    - Subtitles are OPTIONAL (shorts mode)
    - Preserves video extension (mkv/mp4/etc)
    """
    input_dir, _ = ensure_dirs()

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
    placeholder = str(project_root() / "data/assets/blank.jpg")
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

def split_narration_text(panel_text: str):
    """
    Split global narration panel text into:
    - Original dialogue (full extracted subtitles)
    - AI narration summary

    Works with the ONE-SUMMARY format produced by run_ai_narrated_recap.
    """

    if not panel_text:
        return "", ""

    if "ORIGINAL:" not in panel_text or "SUMMARY:" not in panel_text:
        # Safety fallback (should not happen)
        return "", ""

    try:
        original_part = panel_text.split("ORIGINAL:", 1)[1]
        original_text, summary_part = original_part.split("SUMMARY:", 1)

        original_text = original_text.strip().rstrip("-").strip()
        summary_text = summary_part.strip()

        return original_text, summary_text

    except Exception:
        # Absolute safety: never crash the UI
        return "", ""

# def split_narration_text(panel_text: str):
#     """
#     Splits narration panel text into original vs summary
#     for side-by-side display.
#     """
#     originals = []
#     summaries = []

#     blocks = panel_text.split("-" * 48)

#     for b in blocks:
#         if "ORIGINAL:" in b and "SUMMARY:" in b:
#             orig = b.split("ORIGINAL:")[1].split("SUMMARY:")[0].strip()
#             summ = b.split("SUMMARY:")[1].strip()
#             originals.append(orig)
#             summaries.append(summ)

#     return "\n\n".join(originals), "\n\n".join(summaries)

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