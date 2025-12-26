from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

from app.srt_parser import load_srt


@dataclass
class RecapTimelineBlock:
    recap_start: float
    recap_end: float
    episode_start: float
    episode_end: float
    original_text: str
    summary_text: str

def build_recap_timeline(
    segments_json: Path,
    episode_srt: Path,
) -> List[RecapTimelineBlock]:
    segments = json.loads(segments_json.read_text(encoding="utf-8"))
    subtitles = load_srt(str(episode_srt))

    blocks: List[RecapTimelineBlock] = []
    recap_cursor = 0.0

    for seg in segments:
        seg_start = seg["start_s"]
        seg_end = seg["end_s"]

        texts = []
        for s in subtitles:
            if s.end_s <= seg_start:
                continue
            if s.start_s >= seg_end:
                break
            texts.append(s.text)

        text = " ".join(texts).strip()
        duration = max(0.1, seg_end - seg_start)

        block = RecapTimelineBlock(
            recap_start=recap_cursor,
            recap_end=recap_cursor + duration,
            episode_start=seg_start,
            episode_end=seg_end,
            original_text=text,
            summary_text=text,  # initially same
        )

        blocks.append(block)
        recap_cursor += duration

    return blocks
