from pathlib import Path
from faster_whisper import WhisperModel


def transcribe_to_srt(
    video_path: Path,
    srt_out: Path,
    model_name: str = "small",
    language: str | None = None,
):
    """
    Transcribe video to SRT using Faster-Whisper.
    """

    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
    )

    segments, info = model.transcribe(
        str(video_path),
        language=language,
    )

    with open(srt_out, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = _fmt(seg.start)
            end = _fmt(seg.end)
            f.write(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n\n")


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")
