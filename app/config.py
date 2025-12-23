from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    # --------------------------------------------------
    # Core
    # --------------------------------------------------
    video_filename: str
    subtitles_filename: Optional[str]
    max_segments: int
    output_basename: str = "recap"
    output_subdir: Optional[str] = None

    # --------------------------------------------------
    # Project paths
    # --------------------------------------------------
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def input_dir(self) -> Path:
        return self.project_root / "data" / "input"

    @property
    def output_dir(self) -> Path:
        base = self.project_root / "data" / "output"
        return base / self.output_subdir if self.output_subdir else base

    # --------------------------------------------------
    # Derived paths
    # --------------------------------------------------
    @property
    def video_path(self) -> Path:
        return self.input_dir / self.video_filename

    @property
    def subtitles_path(self) -> Optional[Path]:
        return self.input_dir / self.subtitles_filename if self.subtitles_filename else None
