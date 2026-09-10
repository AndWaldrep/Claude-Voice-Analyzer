"""Acoustic feature extraction: volume, pitch, pauses, spectral clarity."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import parselmouth
from parselmouth.praat import call

EPS = 1e-10


def db(x: np.ndarray | float) -> np.ndarray | float:
    return 20 * np.log10(np.maximum(x, EPS))


# --------------------------------------------------------------------------- volume
@dataclass
class VolumeStats:
    frame_times: np.ndarray
    frame_db: np.ndarray            # RMS dBFS per 50 ms frame (all frames)
    speech_mask: np.ndarray         # frames considered speech
    mean_db: float
    median_db: float
    p10_db: float
    p90_db: float
    dynamic_range_db: float         # p90 - p10 of 0.5 s peak levels during speech
    std_db: float
    peak_dbfs: float
    clipped_pct: float
    minute_means: list[float] = field(default_factory=list)
    drift_db_per_min: float = 0.0   # linear trend of volume across recording
    noise_floor_db: float = 0.0
    snr_db: float = 0.0             # median speech level minus noise floor


def speech_mask_from_words(words, n_frames: int, frame_s: float, join_gap: float = 0.25) -> np.ndarray:
    """Frame mask from word timestamps; gaps shorter than join_gap count as speech."""
    mask = np.zeros(n_frames, dtype=bool)
    prev_end = None
    for w in words:
        s, e = w.start, w.end
        if prev_end is not None and s - prev_end < join_gap:
            s = prev_end
        mask[int(s / frame_s): int(np.ceil(e / frame_s))] = True
        prev_end = e
    return mask


def volume_stats(y: np.ndarray, sr: int, frame_s: float = 0.05, mask: np.ndarray | None = None,
                 margin_db: float = 8.0) -> VolumeStats:
    hop = int(sr * frame_s)
    n = len(y) // hop
    frames = y[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    fdb = db(rms)
    times = np.arange(n) * frame_s
    live = fdb > -80          # ignore digital silence (exact zeros) when estimating the noise floor
    floor = float(np.percentile(fdb[live], 5)) if live.sum() > 10 else -80.0
    raw = fdb > floor + margin_db
    vad = np.convolve(raw.astype(float), np.ones(3) / 3, mode="same") >= 0.5
    if mask is None:
        mask = vad
    else:
        mask = mask[:n] if len(mask) >= n else np.pad(mask, (0, n - len(mask)))
        mask = mask & (fdb > floor + margin_db / 2)     # drop silent frames inside transcript spans
    if mask.sum() < 3:
        mask = vad if vad.sum() >= 3 else np.ones(n, dtype=bool)
    sp = fdb[mask]
    # dynamic range from 0.5 s peak levels (robust to quiet consonants / micro-pauses inside speech)
    pk = int(0.5 / frame_s)
    peaks = np.array([fdb[i: i + pk].max() for i in range(0, n - pk + 1, pk) if mask[i: i + pk].mean() >= 0.5])
    if len(peaks) < 4:
        peaks = sp
    minutes = int(np.ceil(times[-1] / 60)) if n else 0
    mm = []
    for m in range(minutes):
        sel = mask & (times >= m * 60) & (times < (m + 1) * 60)
        mm.append(float(fdb[sel].mean()) if sel.any() else float("nan"))
    valid = [(i, v) for i, v in enumerate(mm) if not np.isnan(v)]
    drift = 0.0
    if len(valid) >= 3:
        xs, ys = zip(*valid)
        drift = float(np.polyfit(xs, ys, 1)[0])
    return VolumeStats(
        frame_times=times, frame_db=fdb, speech_mask=mask,
        mean_db=float(sp.mean()), median_db=float(np.median(sp)),
        p10_db=float(np.percentile(sp, 10)), p90_db=float(np.percentile(sp, 90)),
        dynamic_range_db=float(np.percentile(peaks, 90) - np.percentile(peaks, 10)),
        std_db=float(sp.std()), peak_dbfs=float(db(np.abs(y).max())),
        clipped_pct=float((np.abs(y) >= 0.99).mean() * 100),
        minute_means=mm, drift_db_per_min=drift,
        noise_floor_db=floor, snr_db=float(np.median(sp) - floor),
    )


# --------------------------------------------------------------------------- pitch
@dataclass
class PitchStats:
    times: np.ndarray
    f0: np.ndarray                  # Hz, NaN where unvoiced
    median_hz: float
    mean_hz: float
    p5_hz: float
    p95_hz: float
    range_semitones: float          # p5 -> p95 span in semitones
    std_semitones: float
    voiced_pct: float
    monotone_pct: float             # share of 3 s windows with < 2 st std
    minute_std_st: list[float] = field(default_factory=list)
    jitter_pct: float = 0.0
    shimmer_pct: float = 0.0
    hnr_db: float = 0.0


def semitones(hz: np.ndarray, ref: float) -> np.ndarray:
    return 12 * np.log2(np.maximum(hz, EPS) / ref)


def pitch_stats(y: np.ndarray, sr: int, fmin: float = 75, fmax: float = 400,
                speech_mask: np.ndarray | None = None, frame_s: float = 0.05) -> PitchStats:
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=fmin, pitch_ceiling=fmax)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 == 0] = np.nan
    times = pitch.xs()
    if speech_mask is not None:
        idx = np.minimum((times / frame_s).astype(int), len(speech_mask) - 1)
        f0[~speech_mask[idx]] = np.nan
    v = f0[~np.isnan(f0)]
    med = float(np.median(v))
    st = semitones(v, med)
    # monotony: fraction of 3-second windows whose semitone std is < 2
    win = int(3 / 0.01)
    mono = []
    for i in range(0, len(f0) - win, win):
        seg = f0[i : i + win]
        seg = seg[~np.isnan(seg)]
        if len(seg) > win * 0.3:
            mono.append(semitones(seg, med).std() < 2.0)
    minute_std = []
    n_min = int(np.ceil(times[-1] / 60))
    for m in range(n_min):
        seg = f0[(times >= m * 60) & (times < (m + 1) * 60)]
        seg = seg[~np.isnan(seg)]
        minute_std.append(float(semitones(seg, med).std()) if len(seg) > 50 else float("nan"))
    # voice quality (jitter / shimmer / HNR). NOTE: only meaningful on clean, close-mic audio.
    if speech_mask is not None and speech_mask.any():
        hop = int(sr * frame_s)
        keep = np.repeat(speech_mask, hop)[: len(y)]
        snd = parselmouth.Sound(y[: len(keep)][keep].astype(np.float64), sampling_frequency=sr)
    pp = call(snd, "To PointProcess (periodic, cc)", fmin, fmax)
    jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    harm = call(snd, "To Harmonicity (cc)", 0.01, fmin, 0.1, 1.0)
    hnr = call(harm, "Get mean", 0, 0)
    return PitchStats(
        times=times, f0=f0, median_hz=med, mean_hz=float(v.mean()),
        p5_hz=float(np.percentile(v, 5)), p95_hz=float(np.percentile(v, 95)),
        range_semitones=float(12 * np.log2(np.percentile(v, 95) / np.percentile(v, 5))),
        std_semitones=float(st.std()), voiced_pct=float(len(v) / len(f0) * 100),
        monotone_pct=float(np.mean(mono) * 100) if mono else 0.0,
        minute_std_st=minute_std, jitter_pct=float(jitter),
        shimmer_pct=float(shimmer), hnr_db=float(hnr),
    )


# --------------------------------------------------------------------------- pauses
@dataclass
class PauseStats:
    pauses: list[tuple[float, float]]   # (start, duration) for pauses >= min_pause
    speech_time_s: float
    total_time_s: float
    pause_time_s: float
    n_pauses: int
    mean_pause_s: float
    long_pauses: list[tuple[float, float]]   # > 2 s
    pauses_per_min: float


def pause_stats(vol: VolumeStats, min_pause: float = 0.3, frame_s: float = 0.05,
                within: list[tuple[float, float]] | None = None) -> PauseStats:
    mask = vol.speech_mask
    total = float(len(mask) * frame_s)
    pauses: list[tuple[float, float]] = []
    run = 0
    for i, is_sp in enumerate(mask):
        if not is_sp:
            run += 1
        else:
            if run * frame_s >= min_pause:
                pauses.append(((i - run) * frame_s, run * frame_s))
            run = 0
    if run * frame_s >= min_pause:
        pauses.append(((len(mask) - run) * frame_s, run * frame_s))
    # ignore leading/trailing silence for pause time
    inner = [p for p in pauses if p[0] > 0.5 and p[0] + p[1] < total - 0.5]
    if within is not None:
        # only pauses that fall entirely inside one of this speaker's turns
        inner = [p for p in inner if any(a - 0.05 <= p[0] and p[0] + p[1] <= b + 0.05 for a, b in within)]
        total = sum(b - a for a, b in within) or total
    pt = sum(d for _, d in inner)
    speech = float(mask.sum() * frame_s)
    return PauseStats(
        pauses=inner, speech_time_s=speech, total_time_s=total, pause_time_s=pt,
        n_pauses=len(inner), mean_pause_s=float(np.mean([d for _, d in inner])) if inner else 0.0,
        long_pauses=[p for p in inner if p[1] > 2.0],
        pauses_per_min=len(inner) / (total / 60) if total else 0.0,
    )


# --------------------------------------------------------------------------- clarity
@dataclass
class ClarityStats:
    consonant_energy_ratio: float   # energy 2-8 kHz / total, speech frames only
    spectral_centroid_hz: float
    spectral_flux_cv: float         # articulation crispness proxy
    high_band_db_delta: float       # 4-8 kHz level minus 0.3-1 kHz level


def clarity_stats(y: np.ndarray, sr: int, mask: np.ndarray, frame_s: float = 0.05) -> ClarityStats:
    hop = int(sr * frame_s)
    n = min(len(mask), len(y) // hop)
    frames = y[: n * hop].reshape(n, hop)[mask[:n]]
    win = np.hanning(hop)
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(hop, 1 / sr)
    tot = spec.sum(axis=1) + EPS
    hi = spec[:, (freqs >= 2000) & (freqs <= 8000)].sum(axis=1)
    low = spec[:, (freqs >= 300) & (freqs <= 1000)].sum(axis=1)
    vhi = spec[:, (freqs >= 4000) & (freqs <= 8000)].sum(axis=1)
    centroid = (spec * freqs).sum(axis=1) / tot
    flux = np.sqrt(((np.diff(np.sqrt(spec), axis=0)) ** 2).sum(axis=1))
    return ClarityStats(
        consonant_energy_ratio=float((hi / tot).mean()),
        spectral_centroid_hz=float(centroid.mean()),
        spectral_flux_cv=float(flux.std() / (flux.mean() + EPS)),
        high_band_db_delta=float(10 * np.log10(vhi.mean() + EPS) - 10 * np.log10(low.mean() + EPS)),
    )


# --------------------------------------------------------------------------- audience response
def detect_audience(y: np.ndarray, sr: int, word_mask: np.ndarray, vol: VolumeStats,
                    frame_s: float = 0.05, min_dur: float = 0.8,
                    flat_thr: float = 0.07, centroid_thr: float = 2100.0) -> list[tuple[float, float, str]]:
    """Find audience reactions acoustically: energetic stretches that carry no transcribed words and
    are noticeably broader-band and brighter than this speaker's voice.

    Whisper's voice-activity filter strips laughter and applause before the model sees them, so text
    markers alone miss almost all of it. Applause is near-white noise; laughter is a rougher, brighter
    signal than speech. Both sit well above speech in spectral flatness and centroid.

    Returns (start, duration, kind) with kind in {"applause", "laughter"} -- the split between the two
    is by duration and flatness and is approximate; the detection itself is the reliable part.
    """
    import librosa

    hop = int(sr * frame_s)
    n = min(len(word_mask), len(vol.speech_mask), len(y) // hop)
    cand = vol.speech_mask[:n] & ~word_mask[:n]
    flat = librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=hop)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=hop)[0]
    out: list[tuple[float, float, str]] = []
    run = 0
    for i in range(n + 1):
        if i < n and cand[i]:
            run += 1
            continue
        if run * frame_s >= min_dur:
            a, b = i - run, i
            f, c = float(flat[a:b].mean()), float(cent[a:b].mean())
            if f > flat_thr and c > centroid_thr:
                dur = run * frame_s
                out.append((a * frame_s, dur, "applause" if (dur > 3.0 or f > 0.11) else "laughter"))
        run = 0
    return out


# --------------------------------------------------------------------------- phrase endings
def trailing_off(vol: VolumeStats, pauses: PauseStats, window_s: float = 0.6, frame_s: float = 0.05) -> dict:
    """Compare loudness in the last `window_s` of each phrase to the phrase's middle.

    Speakers who 'trail off' drop several dB at phrase ends."""
    ends = [start for start, _ in pauses.pauses]
    w = int(window_s / frame_s)
    drops = []
    for e in ends:
        i = int(round(e / frame_s))
        t0, t1, b0 = max(0, i - w), i, max(0, i - 4 * w)
        tail = vol.frame_db[t0:t1][vol.speech_mask[t0:t1]]
        body = vol.frame_db[b0:t0][vol.speech_mask[b0:t0]]
        if len(tail) >= 3 and len(body) >= 6:
            drops.append(float(body.mean() - tail.mean()))
    if not drops:
        return {"mean_drop_db": float("nan"), "pct_phrases_dropping_over_4db": float("nan"), "n_phrases": 0}
    drops = np.array(drops)
    return {
        "mean_drop_db": float(drops.mean()),
        "pct_phrases_dropping_over_4db": float((drops > 4).mean() * 100),
        "n_phrases": int(len(drops)),
    }
