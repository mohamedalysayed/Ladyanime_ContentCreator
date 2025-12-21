from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """
    Central configuration for the MVP.

    This MVP is deterministic:
    - Parses SRT
    - Scores subtitle lines based on keyword hits
    - Clusters hits into time segments
    - Cuts video segments via FFmpeg
    - Concatenates them into a recap video
    """

    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    input_dir: Path = project_root / "data" / "input"
    output_dir: Path = project_root / "data" / "output"

    # I/O file names (MVP expects these names by default)
    video_filename: str = "video.mkv"
    subtitles_filename: str = "subtitles.srt"

    # Keyword matching
    # You will evolve this later (LLM blocks), but MVP uses a static list.
    keywords: tuple[str, ...] = (
        "fight",
        "kill",
        "escape",
        "reveal",
        "secret",
        "promise",
        "enemy",
        "attack",
        "betray",
        "save",
        "die",
    )

    # Segment logic
    # - padding_sec adds context around each detected cluster
    # - merge_gap_sec merges clusters close to each other
    # - min_segment_sec avoids too-short clips
    padding_sec: float = 2.0
    merge_gap_sec: float = 4.0
    min_segment_sec: float = 3.0

    # Safety / output
    max_segments: int = 40  # MVP cap to avoid producing a giant recap unintentionally
    output_basename: str = "recap"

    @property
    def video_path(self) -> Path:
        return self.input_dir / self.video_filename

    @property
    def subtitles_path(self) -> Path:
        return self.input_dir / self.subtitles_filename

