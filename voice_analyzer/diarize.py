"""Speaker diarization: ECAPA-TDNN speaker embeddings (SpeechBrain) + agglomerative clustering,
aligned to word-level transcript timings. Runs fully offline on CPU."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transcribe import Transcript, Word

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


@dataclass
class Diarization:
    turns: list[Turn]
    speakers: list[str]
    n_windows: int
    window_labels: list[tuple[float, float, str]]     # (start, end, speaker) of embedding windows

    def turns_for(self, spk: str) -> list[Turn]:
        return [t for t in self.turns if t.speaker == spk]

    def words_for(self, spk: str) -> list[Word]:
        return [w for t in self.turns if t.speaker == spk for w in t.words]


def _speech_regions(words: list[Word], join_gap: float = 0.4) -> list[tuple[float, float]]:
    regions: list[list[float]] = []
    for w in words:
        if regions and w.start - regions[-1][1] < join_gap:
            regions[-1][1] = max(regions[-1][1], w.end)
        else:
            regions.append([w.start, w.end])
    return [(a, b) for a, b in regions]


def _windows(regions, win: float, hop: float):
    out = []
    for a, b in regions:
        t = a
        if b - a < win * 0.5:
            out.append((a, b))
            continue
        while t + win <= b + 1e-6:
            out.append((t, t + win))
            t += hop
        if not out or out[-1][1] < b - hop * 0.5:
            out.append((max(a, b - win), b))
    return out


def embed_windows(y: np.ndarray, sr: int, windows, device: str = "cpu", batch: int = 32) -> np.ndarray:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(source=ECAPA_SOURCE, run_opts={"device": device})
    embs = np.zeros((len(windows), 192), dtype=np.float32)
    # batch windows of equal length (most are exactly `win` long); pad the odd ones to batch max
    order = sorted(range(len(windows)), key=lambda i: windows[i][1] - windows[i][0])
    with torch.no_grad():
        for i0 in range(0, len(order), batch):
            idx = order[i0: i0 + batch]
            segs = [y[int(windows[i][0] * sr): int(windows[i][1] * sr)] for i in idx]
            L = max(max(len(s) for s in segs), int(sr * 0.5))
            x = torch.stack([torch.tensor(np.pad(s, (0, L - len(s))), dtype=torch.float32) for s in segs])
            lens = torch.tensor([len(s) / L for s in segs])
            e = enc.encode_batch(x, lens).squeeze(1).cpu().numpy()
            embs[idx] = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    return embs


def _affinity(embs: np.ndarray) -> np.ndarray:
    """Cosine affinity of mean-centred embeddings. Centring removes the channel / room component
    shared by every window, which sharpens the block structure between speakers."""
    X = embs - embs.mean(axis=0, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    A = np.clip(X @ X.T, 0, None)
    np.fill_diagonal(A, 1.0)
    return A


def estimate_speakers(embs: np.ndarray, max_speakers: int = 8, single_gap: float = 0.12) -> tuple[int, np.ndarray]:
    """Eigengap heuristic on the normalised Laplacian of the refined affinity.
    Returns (k, eigenvalues). k = 1 when the second eigenvalue is already large (no cluster structure)."""
    from scipy.sparse.csgraph import laplacian

    n = len(embs)
    if n < 3:
        return 1, np.zeros(n)
    A = _affinity(embs)
    ev = np.sort(np.linalg.eigvalsh(laplacian(A, normed=True)))
    m = min(max_speakers, n - 1)
    if ev[1] > single_gap:
        return 1, ev
    gaps = np.diff(ev[: m + 1])          # gaps[i] = ev[i+1] - ev[i]; k = i+1
    k = int(np.argmax(gaps[1:]) + 2) if m >= 2 else 1   # ignore gap after ev[0] (always 0)
    return k, ev


def cluster(embs: np.ndarray, n_speakers: int | None = None, threshold: float = 0.12,
            max_speakers: int = 8, merge_sim: float = 0.75) -> np.ndarray:
    """Spectral clustering of speaker embeddings. If n_speakers is None the count is chosen by the
    eigengap heuristic; `threshold` is the second-eigenvalue level above which the recording is
    treated as single-speaker (higher = more willing to call it one voice)."""
    from scipy.cluster.vq import kmeans2

    n = len(embs)
    if n <= 2:
        return np.zeros(n, dtype=int)
    k = n_speakers or estimate_speakers(embs, max_speakers, threshold)[0]
    k = max(1, min(k, n - 1))
    if k == 1:
        return np.zeros(n, dtype=int)
    from scipy.sparse.csgraph import laplacian

    A = _affinity(embs)
    w, V = np.linalg.eigh(laplacian(A, normed=True))
    U = V[:, np.argsort(w)[:k]]
    U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-9
    best, best_d = None, np.inf
    for seed in range(10):
        cents, lab = kmeans2(U, k, minit="++", seed=seed)
        d = ((U - cents[lab]) ** 2).sum()
        if d < best_d:
            best, best_d = lab, d
    best = np.asarray(best)
    # safeguard: merge clusters whose *raw* centroids are still near-identical voices
    if n_speakers is None:
        changed = True
        while changed and best.max() > 0:
            changed = False
            ids = np.unique(best)
            C = np.stack([embs[best == i].mean(axis=0) for i in ids])
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
            S = C @ C.T
            np.fill_diagonal(S, -1)
            i, j = np.unravel_index(np.argmax(S), S.shape)
            if S[i, j] > merge_sim:
                best[best == ids[j]] = ids[i]
                changed = True
    order: dict[int, int] = {}
    return np.array([order.setdefault(int(l), len(order)) for l in best])


def _smooth(labels: np.ndarray, k: int = 3) -> np.ndarray:
    """Majority vote over a small temporal neighbourhood to remove single-window flips."""
    out = labels.copy()
    for i in range(len(labels)):
        lo, hi = max(0, i - k // 2), min(len(labels), i + k // 2 + 1)
        vals, counts = np.unique(labels[lo:hi], return_counts=True)
        if counts.max() > (hi - lo) / 2:
            out[i] = vals[np.argmax(counts)]
    return out


def _regions_from_mask(mask: np.ndarray, frame_s: float, join_gap: float = 0.4, min_len: float = 0.5):
    regs: list[list[float]] = []
    for i, v in enumerate(mask):
        if not v:
            continue
        t = i * frame_s
        if regs and t - regs[-1][1] < join_gap:
            regs[-1][1] = t + frame_s
        else:
            regs.append([t, t + frame_s])
    return [(a, b) for a, b in regs if b - a >= min_len]


def _merge_regions(*lists, join_gap: float = 0.3):
    allr = sorted(r for lst in lists for r in lst)
    out: list[list[float]] = []
    for a, b in allr:
        if out and a - out[-1][1] < join_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def diarize(y: np.ndarray, sr: int, tr: Transcript, n_speakers: int | None = None,
            win: float = 3.0, hop: float = 1.5, threshold: float = 0.25, device: str = "cpu",
            speech_mask: np.ndarray | None = None, frame_s: float = 0.05,
            embeddings: np.ndarray | None = None) -> Diarization:
    words = [w for w in tr.words if w.end > w.start]
    regions = _speech_regions(words)
    if speech_mask is not None:
        regions = _merge_regions(regions, _regions_from_mask(speech_mask, frame_s))
    wins = _windows(regions, win, hop)
    if not wins:
        return Diarization([], [], 0, [])
    embs = embeddings if embeddings is not None else embed_windows(y, sr, wins, device)
    k_est, ev = estimate_speakers(embs, single_gap=threshold)
    labels = _smooth(cluster(embs, n_speakers, threshold))
    spk_names = [f"Speaker {i + 1}" for i in range(labels.max() + 1)]
    win_labels = [(a, b, spk_names[l]) for (a, b), l in zip(wins, labels)]

    # assign each word to the window(s) covering its midpoint (weighted by overlap)
    starts = np.array([a for a, _ in wins]); ends = np.array([b for _, b in wins])
    turns: list[Turn] = []
    for w in words:
        mid = (w.start + w.end) / 2
        cover = np.where((starts <= mid) & (ends >= mid))[0]
        if len(cover) == 0:
            cover = [int(np.argmin(np.abs((starts + ends) / 2 - mid)))]
        votes = np.bincount(labels[cover], minlength=len(spk_names))
        spk = spk_names[int(np.argmax(votes))]
        if turns and turns[-1].speaker == spk and w.start - turns[-1].end < 2.0:
            turns[-1].words.append(w); turns[-1].end = w.end
        else:
            turns.append(Turn(spk, w.start, w.end, [w]))
    # absorb very short turns (1-2 words, < 0.6 s) sandwiched between the same speaker
    cleaned = turns
    merged: list[Turn] = []
    for i, t in enumerate(cleaned):
        if (0 < i < len(cleaned) - 1 and len(t.words) <= 2 and t.end - t.start < 0.6
                and cleaned[i - 1].speaker == cleaned[i + 1].speaker != t.speaker):
            t.speaker = cleaned[i - 1].speaker
        if merged and merged[-1].speaker == t.speaker and t.start - merged[-1].end < 2.0:
            merged[-1].words.extend(t.words); merged[-1].end = t.end
        else:
            merged.append(t)
    # every clustered voice is a speaker, even one with no transcribed words (acoustic-only)
    used = sorted({l for _, _, l in win_labels} | {t.speaker for t in merged}, key=lambda s: int(s.split()[-1]))
    d = Diarization(merged, used, len(wins), win_labels)
    d.embeddings = embs
    d.eigenvalues = ev[:8].tolist()
    d.k_estimated = k_est
    return d


def speaker_mask(dia: Diarization, spk: str, n_frames: int, frame_s: float) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for a, b, l in dia.window_labels:
        if l == spk:
            mask[int(a / frame_s): min(n_frames, int(np.ceil(b / frame_s)))] = True
    for t in dia.turns_for(spk):
        mask[int(t.start / frame_s): min(n_frames, int(np.ceil(t.end / frame_s)))] = True
    return mask


def interaction_stats(dia: Diarization, total_s: float, overlap_s: float = 0.15,
                      speech_mask: np.ndarray | None = None, frame_s: float = 0.05) -> dict:
    """Talk-time share, turn counts, quick take-overs (a speaker starting within `overlap_s` of, or before,
    the previous speaker's last word). Recognizer word timings are ~50-100 ms coarse, so this flags
    overlaps and near-overlaps, not polite 0.5 s hand-offs."""
    out = {}
    for spk in dia.speakers:
        ts = dia.turns_for(spk)
        # union of this speaker's embedding windows (they overlap by hop) and transcript turns
        spans = sorted([(a, b) for a, b, l in dia.window_labels if l == spk] + [(t.start, t.end) for t in ts])
        talk = 0.0; cur_a = cur_b = None
        for a, b in spans:
            if cur_b is not None and a <= cur_b:
                cur_b = max(cur_b, b)
            else:
                if cur_b is not None:
                    talk += cur_b - cur_a
                cur_a, cur_b = a, b
        if cur_b is not None:
            talk += cur_b - cur_a
        out[spk] = {
            "talk_time_s": talk,
            "talk_share_pct": 0.0,
            "n_turns": len(ts),
            "mean_turn_s": float(np.mean([t.end - t.start for t in ts])) if ts else 0.0,
            "longest_turn_s": max((t.end - t.start for t in ts), default=0.0),
            "words": sum(len(t.words) for t in ts),
            "interruptions_made": 0,
            "interrupted": 0,
        }
    tot_talk = sum(v["talk_time_s"] for v in out.values()) or 1
    for v in out.values():
        v["talk_share_pct"] = v["talk_time_s"] / tot_talk * 100
    def silence_around(t0: float, t1: float) -> float:
        """Longest run of acoustic silence (s) in [t0 - 1.5, t1 + 0.5], from the energy VAD.
        Recognizer word times are too loose to measure hand-off gaps directly."""
        if speech_mask is None:
            return 0.0
        i0, i1 = max(0, int((t0 - 1.5) / frame_s)), min(len(speech_mask), int((t1 + 0.5) / frame_s) + 1)
        best = run = 0
        for v in speech_mask[i0:i1]:
            run = 0 if v else run + 1
            best = max(best, run)
        return best * frame_s

    for a, b in zip(dia.turns, dia.turns[1:]):
        if a.speaker == b.speaker:
            continue
        gap = max(b.start - a.end, silence_around(a.end, b.start))
        if gap < overlap_s:
            out[b.speaker]["interruptions_made"] += 1
            out[a.speaker]["interrupted"] += 1
    return out
