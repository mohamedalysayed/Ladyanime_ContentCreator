import subprocess

def remove_audio(video_in: Path, video_out: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_in),
        "-c:v", "copy",
        "-an",
        str(video_out),
    ]
    subprocess.run(cmd, check=True)
