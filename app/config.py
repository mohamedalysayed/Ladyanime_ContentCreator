from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    """
    Central configuration for the MVP.

    Supports:
    - Recap mode (with subtitles)
    - Shorts mode (NO subtitles)
    """

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    project_root: Path = Path(__file__).resolve().parents[1]
    input_dir: Path = project_root / "data" / "input"
    output_dir: Path = project_root / "data" / "output"

    # --------------------------------------------------
    # I/O filenames
    # --------------------------------------------------
    video_filename: str = "video.mkv"
    subtitles_filename: Optional[str] = None  # shorts-safe

    # --------------------------------------------------
    # Keyword matching (recap only)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Segment logic
    # --------------------------------------------------
    padding_sec: float = 2.0
    merge_gap_sec: float = 4.0
    min_segment_sec: float = 3.0

    # --------------------------------------------------
    # Safety / output
    # --------------------------------------------------
    max_segments: int = 40
    output_basename: str = "recap"

    # --------------------------------------------------
    # Derived paths
    # --------------------------------------------------
    @property
    def video_path(self) -> Path:
        return self.input_dir / self.video_filename

    @property
    def subtitles_path(self) -> Optional[Path]:
        if not self.subtitles_filename:
            return None
        return self.input_dir / self.subtitles_filename
