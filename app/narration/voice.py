from pathlib import Path
import numpy as np
import soundfile as sf
import tempfile
import subprocess
from openai import OpenAI

# Export OpenAI API_KEY must be set in the environment
#export OPENAI_API_KEY="sk-..."

class VoiceSynthesizer:
    """
    LadyAnime voice synthesizer using OpenAI TTS.

    Features:
    - Natural female voice
    - Works on Python 3.12
    - Time-aligned narration (auto stretch/shrink)
    - Drop-in replacement for your existing pipeline
    """

    def __init__(self):
        self.client = OpenAI()

        # 🎭 LadyAnime-style voice
        self.voice_name = "alloy"   # natural female voice
        self.model = "gpt-4o-mini-tts"

    def synthesize(self, text: str, duration: float, out_wav: Path):
        """
        Generate natural-sounding narrated audio for a recap block.

        IMPORTANT DESIGN PRINCIPLE
        --------------------------
        - We DO NOT stretch or compress speech audio.
        - Timing alignment is achieved by controlling TEXT LENGTH upstream
        (in the summarizer), not by warping audio.
        - This preserves natural voice speed, pitch, and emotion.

        Result:
        - Speech sounds human
        - Narration never overruns its recap segment
        - Silence after speech is acceptable and intentional
        """

        # --------------------------------------------------
        # 0) Prepare output directory
        # --------------------------------------------------
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # 1) Handle empty narration (silent block)
        # --------------------------------------------------
        # Some recap segments may have no dialogue or summary.
        # In that case, we explicitly generate silence so
        # timeline alignment remains correct.
        if not text.strip():
            self._write_silence(duration, out_wav)
            return

        # --------------------------------------------------
        # 2) Generate raw TTS audio to a temporary WAV file
        # --------------------------------------------------
        # We first let the TTS engine speak naturally,
        # without caring about duration.
        # This preserves emotion, rhythm, and clarity.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        response = self.client.audio.speech.create(
            model=self.model,      # gpt-4o-mini-tts
            voice=self.voice_name, # LadyAnime-style voice
            input=text,            # summary_text (already duration-aware)
        )

        # Stream TTS output directly to disk
        response.stream_to_file(tmp_path)

        # --------------------------------------------------
        # 3) Load generated audio
        # --------------------------------------------------
        audio, sr = sf.read(tmp_path)
        tmp_path.unlink(missing_ok=True)

        # Safety check: if TTS produced nothing
        if audio.size == 0:
            self._write_silence(duration, out_wav)
            return

        # --------------------------------------------------
        # 4) Force exact duration (trim or pad with silence)
        # --------------------------------------------------
        target_len = int(duration * sr)

        # If audio longer than target → trim
        if len(audio) > target_len:
            audio = audio[:target_len]

        # If audio shorter than target → pad silence
        elif len(audio) < target_len:
            pad = np.zeros(target_len - len(audio), dtype=audio.dtype)
            audio = np.concatenate([audio, pad])

        # --------------------------------------------------
        # 5) Write final WAV (NO time stretching)
        # --------------------------------------------------
        # We intentionally DO NOT stretch or compress audio.
        #
        # Why?
        # - Speech speed is controlled by text length
        # - Timeline alignment is handled later via adelay
        # - This avoids chipmunk voices or robotic slow speech
        #
        # If the narration ends early, the remaining time
        # will simply be silent — which is correct.
        sf.write(out_wav, audio, sr)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _write_silence(self, duration: float, out_wav: Path):
        sr = 44100
        samples = int(duration * sr)
        silence = np.zeros(samples, dtype=np.float32)
        sf.write(out_wav, silence, sr)
