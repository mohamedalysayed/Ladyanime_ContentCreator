from pathlib import Path
from typing import List
import subprocess

from .recap_timeline import RecapTimelineBlock
from .voice import VoiceSynthesizer


def generate_block_audio(
    blocks: List[RecapTimelineBlock],
    out_dir: Path,
) -> List[Path]:
    """
    Generate one WAV audio file per recap timeline block.

    Each block:
    - Uses *summary_text* (not original text)
    - Is synthesized to exactly match the block duration
    - Will later be aligned precisely on the recap timeline
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    voice = VoiceSynthesizer()
    audio_files: List[Path] = []

    for i, b in enumerate(blocks, 1):
        out = out_dir / f"block_{i:03d}.wav"

        # Duration of this narration block
        duration = max(0.1, b.recap_end - b.recap_start)

        # 🔑 IMPORTANT:
        # Use summary_text for narration
        voice.synthesize(
            text=b.summary_text,
            duration=duration,
            out_wav=out,
        )

        audio_files.append(out)

    return audio_files


def align_audio_to_timeline(
    blocks: List[RecapTimelineBlock],
    audio_files: List[Path],
    out_wav: Path,
):
    """
    Align all narration WAV files onto a single audio track,
    placing each one at its correct recap_start time.

    This uses ffmpeg adelay + amix to build a synchronized narration track.
    """

    filters = []
    inputs = []

    for i, (b, wav) in enumerate(zip(blocks, audio_files)):
        delay_ms = int(b.recap_start * 1000)

        inputs.extend(["-i", str(wav)])
        filters.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")

    mix_inputs = "".join(f"[a{i}]" for i in range(len(audio_files)))
    #filter_complex = ";".join(filters) + f";{mix_inputs}amix=inputs={len(audio_files)}[out]"
    filter_complex = (
        ";".join(filters)
        + f";{mix_inputs}amix=inputs={len(audio_files)}:"
        "dropout_transition=0:"
        "normalize=0[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        str(out_wav),
    ]

    subprocess.run(cmd, check=True)
