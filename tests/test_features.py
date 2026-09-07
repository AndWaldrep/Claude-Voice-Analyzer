"""Smoke tests on synthetic audio (no model download needed)."""
import numpy as np

from voice_analyzer import audio, features


def synth(sr=16000):
    """3 s of a 120 Hz buzz, 1 s silence, 3 s of 150 Hz buzz, with light noise."""
    t = np.arange(3 * sr) / sr
    a = 0.3 * np.sign(np.sin(2 * np.pi * 120 * t)) * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
    b = 0.3 * np.sign(np.sin(2 * np.pi * 150 * t))
    y = np.concatenate([a, np.zeros(sr), b]).astype(np.float32)
    y += 0.001 * np.random.default_rng(0).standard_normal(len(y)).astype(np.float32)
    return y, sr


def test_pipeline_runs_and_detects_pause():
    y, sr = synth()
    y = audio.highpass(y, sr, 80)
    vol = features.volume_stats(y, sr)
    pau = features.pause_stats(vol)
    assert 0.5 < pau.speech_time_s / pau.total_time_s < 0.95
    assert pau.n_pauses == 1 and 0.8 < pau.pauses[0][1] < 1.3
    pit = features.pitch_stats(y, sr, speech_mask=vol.speech_mask)
    assert 110 < pit.median_hz < 160
    assert 2 < pit.range_semitones < 6           # 120 -> 150 Hz is 3.9 st
    cla = features.clarity_stats(y, sr, vol.speech_mask)
    assert 0 < cla.consonant_energy_ratio < 1
    tail = features.trailing_off(vol, pau)
    assert tail["n_phrases"] >= 1


def test_word_mask():
    class W:  # minimal stand-in for transcribe.Word
        def __init__(self, s, e):
            self.start, self.end = s, e
    m = features.speech_mask_from_words([W(0.0, 1.0), W(1.1, 2.0), W(3.0, 4.0)], 100, 0.05)
    assert m[:40].all() and not m[45:59].any() and m[60:80].all()
