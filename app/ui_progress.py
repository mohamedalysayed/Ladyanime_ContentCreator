
# ui_progress.py
import threading
import gradio as gr

STOP_EVENT = threading.Event()

class ProgressController:
    def __init__(self, progress: gr.Progress, prefix: str = ""):
        self.progress = progress
        self.prefix = prefix

    def set(self, frac: float, msg: str):
        if STOP_EVENT.is_set():
            raise gr.Error("🛑 Processing stopped by user.")
        self.progress(max(0.0, min(frac, 1.0)), desc=f"{self.prefix}{msg}")

    def map(self, base: float, span: float, frac: float, msg: str):
        self.set(base + span * frac, msg)


def progress_call(progress_fn, frac: float, msg: str):
    try:
        progress_fn(frac, desc=msg)
    except TypeError:
        progress_fn(frac, msg)


def guarded_progress_cb(progress, prefix: str = ""):
    def _cb(frac: float, msg: str):
        if STOP_EVENT.is_set():
            raise gr.Error("🛑 Processing stopped by user.")
        progress_call(progress, frac, f"{prefix}{msg}")
    return _cb
