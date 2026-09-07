"""Diarization clustering tests on synthetic embeddings (no model download needed)."""
import numpy as np

from voice_analyzer import diarize
from voice_analyzer.transcribe import Word


def fake_embeddings(n_per_speaker, n_speakers, noise=0.9, seed=0):
    """Unit embeddings around random centres; noise=0.9 gives within-speaker cosine ≈ 0.5-0.6 and
    between-speaker ≈ 0.0-0.3 (roughly what ECAPA gives on real, noisy recordings)."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_speakers, 192))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    embs, labels = [], []
    for s in range(n_speakers):
        for _ in range(n_per_speaker):
            nv = rng.standard_normal(192); nv /= np.linalg.norm(nv)
            e = centres[s] + noise * nv
            embs.append(e / np.linalg.norm(e)); labels.append(s)
    embs, labels = np.array(embs), np.array(labels)
    perm = rng.permutation(len(labels))
    return embs[perm], labels[perm]


def _purity(pred, truth):
    total = 0
    for p in np.unique(pred):
        _, counts = np.unique(truth[pred == p], return_counts=True)
        total += counts.max()
    return total / len(truth)


def test_eigengap_counts_speakers():
    for k in (2, 3, 4):
        embs, truth = fake_embeddings(25, k)
        est, _ = diarize.estimate_speakers(embs)
        assert est == k, f"expected {k}, got {est}"
        pred = diarize.cluster(embs)
        assert pred.max() + 1 == k and _purity(pred, truth) > 0.95


def test_single_speaker_detected():
    embs, _ = fake_embeddings(60, 1)
    est, _ = diarize.estimate_speakers(embs)
    assert est == 1
    assert diarize.cluster(embs).max() == 0


def test_forced_speaker_count():
    embs, truth = fake_embeddings(20, 3)
    pred = diarize.cluster(embs, n_speakers=3)
    assert pred.max() + 1 == 3 and _purity(pred, truth) > 0.95


def test_windows_and_regions():
    words = [Word("a", 0.0, 0.5, 0.9), Word("b", 0.6, 1.0, 0.9), Word("c", 5.0, 9.5, 0.9)]
    regs = diarize._speech_regions(words)
    assert regs == [(0.0, 1.0), (5.0, 9.5)]
    wins = diarize._windows(regs, 3.0, 1.5)
    assert wins[0] == (0.0, 1.0)                     # short region kept whole
    assert all(b - a <= 3.0 + 1e-6 for a, b in wins)
    assert wins[-1][1] == 9.5


def test_interaction_uses_acoustic_gap():
    t1 = diarize.Turn("Speaker 1", 0.0, 2.0, [Word("hi", 0.0, 2.0, 0.9)])
    t2 = diarize.Turn("Speaker 2", 2.0, 4.0, [Word("yo", 2.0, 4.0, 0.9)])   # recognizer says no gap
    dia = diarize.Diarization([t1, t2], ["Speaker 1", "Speaker 2"], 2, [(0, 2, "Speaker 1"), (2, 4, "Speaker 2")])
    mask = np.ones(80, dtype=bool); mask[34:44] = False                     # but 0.5 s of real silence
    st = diarize.interaction_stats(dia, 4.0, speech_mask=mask, frame_s=0.05)
    assert st["Speaker 2"]["interruptions_made"] == 0
    st = diarize.interaction_stats(dia, 4.0, speech_mask=np.ones(80, dtype=bool), frame_s=0.05)
    assert st["Speaker 2"]["interruptions_made"] == 1
    assert abs(st["Speaker 1"]["talk_share_pct"] - 50) < 1
