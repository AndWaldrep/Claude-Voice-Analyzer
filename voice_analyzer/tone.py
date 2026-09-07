"""Emotional-undertone indicators from prosody, per speaker turn.

Everything here is relative to the *speaker's own* baseline (z-scores), which is what a
listener actually perceives: a normally quiet person getting louder reads as animated even
if they are still quieter than the other person in the room.

Indicators
  arousal   – energy + pitch height + speaking rate (high = energised / agitated, low = flat / withdrawn)
  expressiveness – pitch variability within the turn
  final contour  – pitch slope over the last 0.5 s of the turn (rising on a statement = tentative /
                   seeking approval; falling = assertive / closed)
  tension   – harmonics-to-noise ratio dip + high jitter relative to baseline (only trusted on clean audio)
  hesitancy – fillers, restarts and mid-turn pauses per minute of the turn
An optional categorical model (SpeechBrain wav2vec2 IEMOCAP: neutral / angry / happy / sad) can be
layered on; it is marked experimental because such models are trained on acted speech and are
poorly calibrated on real conversations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .diarize import Turn
from .features import PitchStats, VolumeStats, semitones
from .text_metrics import FILLERS, count_syllables

EMO_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"


@dataclass
class TurnTone:
    start: float
    end: float
    speaker: str
    text: str
    energy_z: float
    pitch_z: float
    rate_z: float
    arousal: float
    pitch_spread_st: float
    final_slope_st_per_s: float
    hesitancy_per_min: float
    tags: list[str] = field(default_factory=list)
    categorical: dict[str, float] | None = None


@dataclass
class SpeakerTone:
    speaker: str
    turns: list[TurnTone]
    baseline: dict
    summary_tags: dict[str, int]
    notable: list[TurnTone]


def _turn_feats(t: Turn, vol: VolumeStats, pit: PitchStats, frame_s: float = 0.05):
    i0, i1 = int(t.start / frame_s), int(np.ceil(t.end / frame_s))
    fdb = vol.frame_db[i0:i1][vol.speech_mask[i0:i1]]
    sel = (pit.times >= t.start) & (pit.times <= t.end)
    f0 = pit.f0[sel]; f0 = f0[~np.isnan(f0)]
    syl = sum(count_syllables(w.word) for w in t.words)
    dur = max(t.end - t.start, 0.3)
    # final contour: slope of semitone pitch over last 0.5 s of voiced frames
    sel_end = (pit.times >= t.end - 0.5) & (pit.times <= t.end)
    fe = pit.f0[sel_end]; te = pit.times[sel_end]
    ok = ~np.isnan(fe)
    slope = float(np.polyfit(te[ok], semitones(fe[ok], pit.median_hz), 1)[0]) if ok.sum() >= 5 else 0.0
    toks = [re.sub(r"[^a-z']", "", w.word.lower()) for w in t.words]
    fill = sum(1 for i, tk in enumerate(toks) if tk in FILLERS and tk in {"um", "uh", "er", "ah", "hmm"}
               or (i + 1 < len(toks) and f"{tk} {toks[i+1]}" in {"you know", "i mean"}))
    restarts = sum(1 for a, b in zip(toks, toks[1:]) if a and a == b)
    gaps = sum(1 for a, b in zip(t.words, t.words[1:]) if b.start - a.end > 0.6)
    return dict(
        energy=float(fdb.mean()) if len(fdb) else np.nan,
        pitch=float(np.median(semitones(f0, pit.median_hz))) if len(f0) else np.nan,
        spread=float(semitones(f0, pit.median_hz).std()) if len(f0) > 5 else np.nan,
        rate=syl / dur, slope=slope,
        hes=(fill + restarts + gaps) / (dur / 60),
    )


def _z(x, mu, sd):
    return float((x - mu) / sd) if sd > 1e-6 and not np.isnan(x) else 0.0


def analyze_speaker_tone(spk: str, turns: list[Turn], vol: VolumeStats, pit: PitchStats,
                         min_turn_s: float = 1.5, y=None, sr=None, categorical_model=None) -> SpeakerTone:
    ts = [t for t in turns if t.end - t.start >= min_turn_s and len(t.words) >= 3]
    feats = [_turn_feats(t, vol, pit) for t in ts]
    if not feats:
        return SpeakerTone(spk, [], {}, {}, [])
    base = {}
    for k in ("energy", "pitch", "rate", "spread"):
        v = np.array([f[k] for f in feats]); v = v[~np.isnan(v)]
        base[k] = (float(np.nanmedian(v)) if len(v) else 0.0, float(v.std()) if len(v) > 2 else 1.0)
    out: list[TurnTone] = []
    for t, f in zip(ts, feats):
        ez, pz, rz = _z(f["energy"], *base["energy"]), _z(f["pitch"], *base["pitch"]), _z(f["rate"], *base["rate"])
        ar = float(np.mean([ez, pz, rz]))
        tags = []
        if ar > 0.8: tags.append("energised")
        if ar < -0.8: tags.append("flat / low-energy")
        if ez > 1.2: tags.append("louder than usual")
        if ez < -1.2: tags.append("quieter than usual")
        if rz > 1.2: tags.append("rushing")
        if rz < -1.2: tags.append("slow / deliberate")
        if not np.isnan(f["spread"]) and f["spread"] > base["spread"][0] + base["spread"][1]: tags.append("expressive")
        if not np.isnan(f["spread"]) and f["spread"] < max(1.5, base["spread"][0] - base["spread"][1]): tags.append("monotone")
        is_q = t.text.strip().endswith("?")
        if f["slope"] > 4 and not is_q: tags.append("rising end (tentative)")
        if f["slope"] < -4: tags.append("falling end (assertive)")
        if f["hes"] > 8: tags.append("hesitant")
        cat = None
        if categorical_model is not None and y is not None:
            cat = categorical(categorical_model, y[int(t.start * sr): int(t.end * sr)])
        out.append(TurnTone(t.start, t.end, spk, t.text, ez, pz, rz, ar,
                            float(f["spread"]) if not np.isnan(f["spread"]) else 0.0, f["slope"], f["hes"], tags, cat))
    counts: dict[str, int] = {}
    for tt in out:
        for tg in tt.tags:
            counts[tg] = counts.get(tg, 0) + 1
    notable = sorted(out, key=lambda x: -abs(x.arousal))[:6]
    return SpeakerTone(spk, out, {k: v[0] for k, v in base.items()}, counts, sorted(notable, key=lambda x: x.start))


def load_categorical(device: str = "cpu"):
    from speechbrain.inference.interfaces import foreign_class

    return foreign_class(source=EMO_SOURCE, pymodule_file="custom_interface.py",
                         classname="CustomEncoderWav2vec2Classifier", run_opts={"device": device})


def categorical(model, seg: np.ndarray, max_s: float = 10.0, sr: int = 16000) -> dict[str, float]:
    import torch

    seg = seg[: int(max_s * sr)]
    with torch.no_grad():
        prob, _, _, lab = model.classify_batch(torch.tensor(seg, dtype=torch.float32)[None])
    names = {0: "neutral", 1: "angry", 2: "happy", 3: "sad"}
    p = prob.squeeze().cpu().numpy()
    return {names[i]: float(p[i]) for i in range(len(p))}
