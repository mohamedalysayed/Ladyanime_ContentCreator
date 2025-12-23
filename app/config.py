from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    """
    Central immutable configuration object for the LadyAnime video pipeline.

    One AppConfig instance represents ONE processing job
    (i.e. one source video → one recap/shorts output folder).
    """

    # --------------------------------------------------
    # Core inputs
    # --------------------------------------------------

    # Filename of the source video (stored inside data/input)
    video_filename: str

    # Optional subtitles filename (stored inside data/input)
    # - None → auto-transcription will be used
    subtitles_filename: Optional[str]

    # Maximum number of segments to generate
    max_segments: int

    # Base name for generated outputs (e.g. recap.mp4, shorts_01.mp4)
    output_basename: str = "recap"

    # Optional subfolder name under output_root
    # (usually derived from the episode filename)
    output_subdir: Optional[str] = None

    # Optional user-selected output directory
    # If None → defaults to <project_root>/data/output
    output_root: Optional[Path] = None

    # --------------------------------------------------
    # Project-level paths
    # --------------------------------------------------

    @property
    def project_root(self) -> Path:
        """
        Absolute path to the project root directory.

        Assumes:
        project_root/
            app/
                config.py  <-- this file
        """
        return Path(__file__).resolve().parents[1]

    @property
    def input_dir(self) -> Path:
        """
        Directory where input files are stored.
        This is managed internally and should not be user-configurable.
        """
        return self.project_root / "data" / "input"

    @property
    def output_dir(self) -> Path:
        """
        Final output directory for all generated artifacts.

        Resolution order:
        1) User-selected output_root (via GUI), if provided
        2) Default project_root/data/output

        If output_subdir is set, outputs go into:
            <base>/<output_subdir>/
        """
        base = self.output_root if self.output_root else (
            self.project_root / "data" / "output"
        )
        return base / self.output_subdir if self.output_subdir else base

    # --------------------------------------------------
    # Derived paths (computed, never stored)
    # --------------------------------------------------

    @property
    def video_path(self) -> Path:
        """
        Absolute path to the input video file.
        """
        return self.input_dir / self.video_filename

    @property
    def subtitles_path(self) -> Optional[Path]:
        """
        Absolute path to the subtitles file, if provided.
        """
        if not self.subtitles_filename:
            return None
        return self.input_dir / self.subtitles_filename
