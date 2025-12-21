from dataclasses import dataclass
from typing import List, Iterable, Tuple


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    score: int  # placeholder for later brain


def build_segments_from_subtitles(
    subtitle_lines: Iterable[Tuple[float, float, str]],
    max_block_sec: float = 30.0,
    silence_gap_sec: float = 5.0,
    min_segment_sec: float = 15.0,
    max_segments: int = 40,
) -> List[Segment]:

    segments: List[Segment] = []

    lines = list(subtitle_lines)
    if not lines:
        return []

    block_start = lines[0][0]
    last_end = lines[0][1]

    for i, (start, end, _) in enumerate(lines[1:], start=1):
        gap = start - last_end
        duration = end - block_start

        is_last = i == len(lines) - 1

        if duration >= max_block_sec or gap >= silence_gap_sec or is_last:
            if duration >= min_segment_sec:
                segments.append(
                    Segment(
                        start_s=block_start,
                        end_s=last_end,
                        score=1
                    )
                )
            block_start = start

        last_end = end

        if len(segments) >= max_segments:
            break

    return segments
