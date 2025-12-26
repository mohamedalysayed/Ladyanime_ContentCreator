from typing import List
from math import floor
from openai import OpenAI
import re

# Timeline data structure used throughout the AI narration pipeline
from .recap_timeline import RecapTimelineBlock


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# Average narration speaking speed (words per minute).
# 150–160 WPM is natural for clear narration.
# WORDS_PER_MIN = 155
WORDS_PER_MIN = 210  # slightly faster for anime style


def apply_ladyanime_tone(text: str) -> str:
    """
    Inject LadyAnime narration style.
    This text will be fed directly into the TTS engine.
    """
    if not text:
        return ""

    return (
        "Narración dulce, expresiva y emocional, "
        "voz femenina joven, estilo anime. "
        "Entonación cálida y envolvente:\n\n"
        + text
    )


def summarize_blocks(
    blocks: List[RecapTimelineBlock],
) -> List[RecapTimelineBlock]:
    """
    Generate summarized narration text for each recap timeline block.

    Goals:
    ------
    - Keep narration duration aligned with recap video timing
    - Avoid narration overflow beyond visual segments
    - Preserve original dialogue for UI/debugging

    Current strategy:
    -----------------
    - Estimate how many words can be spoken in the block duration
    - Trim original subtitle text to that limit
    - Apply LadyAnime narration tone

    This logic is intentionally SIMPLE.
    You can later replace it with:
      - LLM summarization
      - Emotion-aware rewriting
      - Character-aware narration
    """

    summarized: List[RecapTimelineBlock] = []

    for b in blocks:
        # --------------------------------------------------
        # 1) Determine allowed narration length
        # --------------------------------------------------
        duration = max(0.1, b.recap_end - b.recap_start)

        max_words = max(
            5,  # always allow a minimal phrase
            floor(duration * WORDS_PER_MIN / 60)
        )

        # --------------------------------------------------
        # 2) Prepare source text
        # --------------------------------------------------
        src = (b.original_text or "").strip()
        words = src.split()

        # --------------------------------------------------
        # 3) Trim text to fit duration
        # --------------------------------------------------
        trimmed = " ".join(words[:max_words])

        # --------------------------------------------------
        # 4) Apply LadyAnime narration tone
        # --------------------------------------------------
        summary = apply_ladyanime_tone(trimmed)

        # --------------------------------------------------
        # 5) Build new timeline block
        # --------------------------------------------------
        summarized.append(
            RecapTimelineBlock(
                recap_start=b.recap_start,
                recap_end=b.recap_end,
                episode_start=b.episode_start,
                episode_end=b.episode_end,
                original_text=b.original_text,
                summary_text=summary,
            )
        )

    return summarized
def summarize_full_episode(
    episode_srt,
    recap_duration_sec: float,
) -> tuple[str, str]:
    """
    Build ONE global narration in *Mexican Spanish only* from the FULL episode SRT,
    then keep it short enough to fit the recap duration.

    Returns:
        (original_text, summary_text)
    """

    # --------------------------------------------------
    # 1) Load & clean raw subtitle text
    # --------------------------------------------------
    text_lines = []
    with open(episode_srt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.isdigit():
                continue
            if "-->" in line:
                continue

            # remove common subtitle artifacts
            line = re.sub(r"<[^>]+>", "", line)                 # remove HTML tags
            line = re.sub(r"\{\\.*?\}", "", line)               # remove ASS-like tags
            line = re.sub(r"♪.*?♪", "", line)                   # remove song markers
            line = line.replace("♪", "").strip()

            if line:
                text_lines.append(line)

    original_text = " ".join(text_lines).strip()
    if not original_text:
        return "", ""

    # --------------------------------------------------
    # 2) Compute safe word budget (rough)
    # --------------------------------------------------
    # Keep some safety margin so audio doesn't overrun.
    # If you speak ~210 WPM, words/sec = 210/60 = 3.5
    # Use 85% margin to be safe.
    #max_words = max(60, int(recap_duration_sec * (WORDS_PER_MIN / 60) * 0.85))

    # Target words to FILL the recap (not just fit)
    target_words = int(recap_duration_sec * (WORDS_PER_MIN / 60) * 0.98)

    # Allow small overflow safety
    max_words = int(target_words * 1.05)
    min_words = int(target_words * 0.92)

    # --------------------------------------------------
    # 3) REAL summarization (forces Spanish MX only)
    # --------------------------------------------------
    client = OpenAI()

    prompt = f"""
    Eres "LadyAnime", una narradora profesional de anime.

    OBJETIVO:
    Crear una narración continua que ACOMPAÑE TODO el video recap
    desde el segundo 0 hasta el final.

    REGLAS ABSOLUTAS:
    - Español de México únicamente.
    - Prohibido inglés, japonés o palabras mezcladas.
    - Prohibido introducir el video (nada de “en este resumen”, “aquí vemos”, etc).
    - Comienza directamente con los eventos de la historia.
    - Narración fluida, continua, sin repeticiones.
    - Ritmo natural tipo YouTube Lady Anime.
    - Duración objetivo: {target_words} palabras (NO menos de {min_words}).

    CONTENIDO BASE (subtítulos originales):
    {original_text}
    """.strip()

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )

    # `output_text` exists in OpenAI python 2.x
    summary = (resp.output_text or "").strip()

    # # Hard safety: if model returns too long, trim again
    # summary_words = summary.split()
    # if len(summary_words) > max_words:
    #     summary = " ".join(summary_words[:max_words])

    summary_words = summary.split()

    if len(summary_words) < min_words:
        # Ask model to CONTINUE, not repeat
        extend_prompt = f"""
    Continúa la narración manteniendo el mismo tono y estilo.
    NO repitas frases anteriores.
    NO cierres la historia aún.
    Texto actual:
    {summary}
    """
        ext = client.responses.create(
            model="gpt-4o-mini",
            input=extend_prompt,
        )
        extra = (ext.output_text or "").strip()
        summary = f"{summary} {extra}"

    # Apply your tone header (optional; you can remove if it sounds unnatural)
    summary_text = apply_ladyanime_tone(summary)

    return original_text, summary_text
