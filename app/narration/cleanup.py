from pathlib import Path

def cleanup_narration_cache(out_dir: Path):
    """
    Remove heavy intermediate files while keeping
    final outputs and debug metadata.
    """

    # Remove per-block audio
    blocks_dir = out_dir / "blocks"
    if blocks_dir.exists():
        for f in blocks_dir.glob("*.wav"):
            f.unlink()
        blocks_dir.rmdir()

    # Remove silent recap
    silent = out_dir / "recap_silent.mp4"
    if silent.exists():
        silent.unlink()

    # Remove merged narration wav
    narration = out_dir / "narration.wav"
    if narration.exists():
        narration.unlink()