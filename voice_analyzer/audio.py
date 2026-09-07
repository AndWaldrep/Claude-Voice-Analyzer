"""Decode any audio file (m4a, mp3, wav, ogg, ...) to a mono float32 array via ffmpeg."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

TARGET_SR = 16000


def highpass(y: np.ndarray, sr: int, cutoff: float = 80.0, order: int = 4) -> np.ndarray:
    """Remove rumble / handling noise below `cutoff` Hz (standard speech pre-filter)."""
    if cutoff <= 0:
        return y
    sos = butter(order, cutoff, "hp", fs=sr, output="sos")
    return sosfiltfilt(sos, y.astype(np.float64)).astype(np.float32)


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`."
        ) from e


def load_audio(path: str | Path, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Return (samples, sr). Samples are mono float32 in [-1, 1]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    # Fast path: soundfile can read wav/flac/ogg directly.
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        data, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        if native_sr != sr:
            import librosa

            data = librosa.resample(data, orig_sr=native_sr, target_sr=sr)
        return data.astype(np.float32), sr

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "decoded.wav"
        cmd = [
            ffmpeg_path(), "-y", "-loglevel", "error", "-i", str(path),
            "-ac", "1", "-ar", str(sr), "-f", "wav", str(out),
        ]
        subprocess.run(cmd, check=True)
        data, _ = sf.read(str(out), dtype="float32")
    return data.astype(np.float32), sr
