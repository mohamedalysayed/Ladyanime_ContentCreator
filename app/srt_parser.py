from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pysrt


@dataclass(frozen=True)
class SubtitleLine:
    """
    A normalized subtitle line with start/end times in seconds.
    """
    start_s: float
    end_s: float
    text: str


def _to_seconds(t: pysrt.SubRipTime) -> float:
    return (t.hours * 3600) + (t.minutes * 60) + t.seconds + (t.milliseconds / 1000.0)


def load_srt(path: str) -> List[SubtitleLine]:
    """
    Load an SRT file and convert it into a list of SubtitleLine.

    This is a key MVP primitive: everything depends on having clean timestamps + text.
    """
    subs = pysrt.open(path, encoding="utf-8")
    lines: List[SubtitleLine] = []
    for s in subs:
        start_s = _to_seconds(s.start)
        end_s = _to_seconds(s.end)
        text = " ".join(s.text.split())  # normalize whitespace/newlines
        if not text.strip():
            continue
        if end_s <= start_s:
            continue
        lines.append(SubtitleLine(start_s=start_s, end_s=end_s, text=text))
    return lines

