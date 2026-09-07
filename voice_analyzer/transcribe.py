"""Local speech-to-text with word timestamps (faster-whisper). Optional."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Word:
    word: str
    start: float
    end: float
    prob: float


@dataclass
class Transcript:
    words: list[Word]
    segments: list[tuple[float, float, str]]
    language: str

    @property
    def text(self) -> str:
        return " ".join(s[2].strip() for s in self.segments)


def transcribe(y: np.ndarray, sr: int, model_size: str = "base.en", device: str = "cpu") -> Transcript:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type="int8" if device == "cpu" else "float16")
    segs, info = model.transcribe(
        y.astype(np.float32), beam_size=1, word_timestamps=True, vad_filter=True,
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    segments: list[tuple[float, float, str]] = []
    for s in segs:
        segments.append((s.start, s.end, s.text))
        for w in s.words or []:
            words.append(Word(w.word.strip(), w.start, w.end, w.probability))
    return Transcript(words=words, segments=segments, language=info.language)
