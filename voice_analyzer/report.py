"""Turn raw metrics into a coaching report (markdown + JSON)."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def ts(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def _clean(o: Any) -> Any:
    if is_dataclass(o):
        o = asdict(o)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k not in {"frame_times", "frame_db", "speech_mask", "times", "f0"}}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def rate(value: float, good: tuple[float, float], ok: tuple[float, float]) -> str:
    if good[0] <= value <= good[1]:
        return "good"
    if ok[0] <= value <= ok[1]:
        return "fair"
    return "needs work"


def build_report(m: dict, transcript_text: str | None, source: str) -> tuple[str, dict]:
    vol, pit, pau, cla, tail = m["volume"], m["pitch"], m["pauses"], m["clarity"], m["trailing"]
    pace, flu, enu = m.get("pace"), m.get("fluency"), m.get("enunciation")
    total = pau.total_time_s
    findings: dict[str, dict] = {}
    L: list[str] = []
    L.append(f"# Voice & Oratory Analysis\n\n**Source:** `{source}`  \n**Duration:** {ts(total)} "
             f"({pau.speech_time_s/total*100:.0f}% speech, {pau.pause_time_s/total*100:.0f}% pauses)\n")

    # ---------------------------------------------------------------- recording quality
    q = "good" if vol.snr_db >= 20 else "fair" if vol.snr_db >= 12 else "poor"
    findings["recording"] = {"rating": q, "snr_db": vol.snr_db, "noise_floor_db": vol.noise_floor_db}
    L.append("## 0. Recording Quality\n")
    L.append(f"| Metric | Value | Note |\n|---|---|---|")
    L.append(f"| Speech-to-noise ratio | {vol.snr_db:.1f} dB | **{q}** – below ~12 dB the mic is far away or the room is loud; enunciation and volume metrics get less reliable |")
    L.append(f"| Noise floor | {vol.noise_floor_db:.1f} dBFS | |")
    L.append("")

    # ---------------------------------------------------------------- volume
    dr = vol.dynamic_range_db
    vr = rate(dr, (6, 14), (4, 18))
    findings["volume"] = {"rating": vr, "dynamic_range_db": dr, "drift_db_per_min": vol.drift_db_per_min}
    L.append("## 1. Volume\n")
    L.append(f"| Metric | Value | Note |\n|---|---|---|")
    L.append(f"| Average speech level | {vol.mean_db:.1f} dBFS | recording level, not absolute loudness |")
    L.append(f"| Loud vs. quiet passages (p90 − p10) | {dr:.1f} dB | **{vr}** – 6–14 dB is expressive without shouting |")
    L.append(f"| Volume drift over the talk | {vol.drift_db_per_min:+.2f} dB/min | negative = fading as the talk goes on |")
    L.append(f"| Peak / clipping | {vol.peak_dbfs:.1f} dBFS / {vol.clipped_pct:.2f}% samples | {'clipping present – mic too hot or speaker leaned in' if vol.clipped_pct > 0.05 else 'no clipping'} |")
    if tail["n_phrases"]:
        L.append(f"| Phrase endings | drop {tail['mean_drop_db']:.1f} dB in last 0.6 s | {tail['pct_phrases_dropping_over_4db']:.0f}% of {tail['n_phrases']} phrases trail off by more than 4 dB |")
    L.append("")
    mm = [v for v in vol.minute_means if v == v]
    if mm:
        L.append("Per-minute level (dBFS): " + ", ".join(f"{v:.0f}" for v in mm) + "\n")

    # ---------------------------------------------------------------- pitch
    pr = rate(pit.range_semitones, (8, 16), (5, 20))
    findings["pitch"] = {"rating": pr, "median_hz": pit.median_hz, "range_semitones": pit.range_semitones,
                         "monotone_pct": pit.monotone_pct}
    L.append("## 2. Pitch & Intonation\n")
    L.append("| Metric | Value | Note |\n|---|---|---|")
    L.append(f"| Median pitch | {pit.median_hz:.0f} Hz | typical adult male 85–155 Hz, female 165–255 Hz |")
    L.append(f"| Working range (p5–p95) | {pit.p5_hz:.0f}–{pit.p95_hz:.0f} Hz = {pit.range_semitones:.1f} semitones | **{pr}** – engaging speakers use 8–16 st |")
    L.append(f"| Pitch variability (std) | {pit.std_semitones:.1f} st | under 2 st sounds flat |")
    L.append(f"| Monotone stretches | {pit.monotone_pct:.0f}% of 3 s windows | share of the talk that is essentially one note |")
    L.append(f"| Voiced share of frames | {pit.voiced_pct:.0f}% | low values = lots of breathy / whispered / dropped syllables |")
    vq_note = ("jitter <1%, shimmer <4%, HNR >15 dB is a clean, steady voice; vocal fry / strain push these up"
               if vol.snr_db >= 15 else "**not reliable at this recording's noise level** – re-record close-mic to assess voice quality")
    L.append(f"| Voice quality | jitter {pit.jitter_pct:.2f}%, shimmer {pit.shimmer_pct:.1f}%, HNR {pit.hnr_db:.1f} dB | {vq_note} |")
    L.append("")
    ms = [v for v in pit.minute_std_st if v == v]
    if ms:
        L.append("Per-minute pitch variability (semitones): " + ", ".join(f"{v:.1f}" for v in ms) + "\n")

    # ---------------------------------------------------------------- pace
    L.append("## 3. Pace & Pauses\n")
    L.append("| Metric | Value | Note |\n|---|---|---|")
    if pace:
        wr = rate(pace.wpm_overall, (130, 165), (110, 185))
        findings["pace"] = {"rating": wr, "wpm_overall": pace.wpm_overall, "wpm_speaking": pace.wpm_speaking,
                            "syllables_per_sec": pace.syllables_per_sec}
        L.append(f"| Words per minute (overall) | {pace.wpm_overall:.0f} wpm | **{wr}** – conversational 130–165; presentations 120–150 |")
        L.append(f"| Words per minute while speaking | {pace.wpm_speaking:.0f} wpm | pace during actual speech runs |")
        L.append(f"| Articulation rate | {pace.syllables_per_sec:.1f} syll/s | 4–5 normal; >5.5 sounds rushed |")
        L.append(f"| Pace variability | CV {pace.wpm_cv:.2f} | 0.15–0.35 is dynamic; <0.1 is metronomic |")
        L.append(f"| Fastest 30 s | " + ", ".join(f"{ts(t)} ({w:.0f} wpm)" for t, w in pace.fastest_windows) + " | |")
        L.append(f"| Slowest 30 s | " + ", ".join(f"{ts(t)} ({w:.0f} wpm)" for t, w in pace.slowest_windows) + " | |")
    L.append(f"| Pauses (≥0.3 s) | {pau.n_pauses} total, {pau.pauses_per_min:.1f}/min, mean {pau.mean_pause_s:.2f} s | 8–15/min with 0.5–1 s means is a well-punctuated delivery |")
    L.append(f"| Long pauses (>2 s) | {len(pau.long_pauses)} | " + ", ".join(f"{ts(s)} ({d:.1f}s)" for s, d in pau.long_pauses[:8]) + " |")
    L.append("")
    if pace:
        L.append("Per-minute word count: " + ", ".join(f"{v:.0f}" for v in pace.minute_wpm) + "\n")

    # ---------------------------------------------------------------- enunciation
    L.append("## 4. Enunciation & Clarity\n")
    L.append("| Metric | Value | Note |\n|---|---|---|")
    L.append(f"| Consonant energy (2–8 kHz share) | {cla.consonant_energy_ratio*100:.1f}% | higher = crisper consonants (typical 5–15% for close mic speech) |")
    L.append(f"| High band vs. vowel band | {cla.high_band_db_delta:+.1f} dB | very negative = soft, muffled consonants |")
    L.append(f"| Spectral centroid | {cla.spectral_centroid_hz:.0f} Hz | brightness of the voice |")
    if enu:
        er = rate(enu.mean_word_confidence, (0.85, 1.0), (0.75, 1.0))
        findings["enunciation"] = {"rating": er, "mean_word_confidence": enu.mean_word_confidence,
                                   "low_confidence_pct": enu.low_confidence_pct, "rushed_multisyllable_pct": enu.short_word_pct}
        L.append(f"| Recognizer word confidence | {enu.mean_word_confidence:.2f} | **{er}** – how easily a listener (an ASR model) can make out each word |")
        L.append(f"| Hard-to-recognize words | {enu.low_confidence_pct:.1f}% | words the recognizer was <50% sure of |")
        L.append(f"| Rushed long words (3+ syllables) | {enu.short_word_pct:.0f}% | spoken under 120 ms per syllable; e.g. " + ", ".join(f"{ts(t)} “{w}”" for t, w in enu.rushed_examples[:6]) + " |")
        L.append("")
        if enu.low_confidence_examples:
            L.append("Least intelligible words (timestamp, word, confidence): " +
                     ", ".join(f"{ts(t)} “{w}” {p:.2f}" for t, w, p in enu.low_confidence_examples[:12]) + "\n")
    else:
        L.append("")

    # ---------------------------------------------------------------- fluency
    if flu:
        fr = rate(flu.fillers_per_min, (0, 2), (0, 4))
        findings["fluency"] = {"rating": fr, "fillers_per_min": flu.fillers_per_min,
                               "strong_fillers_per_min": flu.strong_fillers_per_min, "repetitions": len(flu.repetitions)}
        L.append("## 5. Fluency & Word Choice\n")
        L.append("| Metric | Value | Note |\n|---|---|---|")
        L.append(f"| Filler words | {flu.fillers_total} ({flu.fillers_per_min:.1f}/min) | **{fr}** – under 2/min is polished |")
        L.append(f"| um / uh only | {flu.strong_fillers_per_min:.1f}/min | |")
        L.append(f"| Immediate word repeats | {len(flu.repetitions)} | " + ", ".join(f"{ts(t)} “{w} {w}”" for t, w in flu.repetitions[:8]) + " |")
        L.append(f"| Sentence length | mean {flu.mean_sentence_words:.0f} words, longest {flu.longest_sentence_words} | >30 words is hard to follow aloud |")
        L.append("")
        if flu.filler_counts:
            L.append("Filler breakdown: " + ", ".join(f"{k} ×{v}" for k, v in flu.filler_counts.items()) + "\n")
        if flu.top_words:
            L.append("Most-used content words: " + ", ".join(f"{k} ×{v}" for k, v in flu.top_words) + "\n")

    # ---------------------------------------------------------------- coaching
    L.append("## 6. Coaching Priorities\n")
    tips: list[str] = []
    if vol.dynamic_range_db < 6:
        tips.append("**Volume is flat.** Add deliberate contrast: drop to near-quiet for the key sentence, then come back up. Mark 2–3 'punch' lines per minute in notes.")
    elif vol.dynamic_range_db > 14:
        tips.append("**Volume swings widely.** Keep the floor up: the quiet passages are falling out of range for listeners at the back.")
    if vol.drift_db_per_min < -0.25:
        tips.append(f"**Volume fades over the talk** ({vol.drift_db_per_min:+.2f} dB/min). Reset breath and posture at each section change; stand, feet shoulder-width, breathe low.")
    if tail["n_phrases"] and tail["pct_phrases_dropping_over_4db"] > 35:
        tips.append(f"**Trailing off at phrase ends** ({tail['pct_phrases_dropping_over_4db']:.0f}% of phrases drop >4 dB). Practice 'landing' the last word of each sentence at full support – the last word is usually the important one.")
    if pit.range_semitones < 8 or pit.monotone_pct > 40:
        tips.append(f"**Narrow pitch range** ({pit.range_semitones:.1f} st, {pit.monotone_pct:.0f}% monotone windows). Read aloud exaggerating: go up on questions and new ideas, down to finish. Record and aim for >10 st.")
    if vol.snr_db >= 15 and (pit.hnr_db < 12 or pit.jitter_pct > 1.5):
        tips.append("**Rough / strained voice quality** (low harmonics-to-noise or high jitter). Suggests vocal fry or pushing from the throat; work on breath support and speaking slightly above the lowest comfortable pitch.")
    if pace:
        if pace.wpm_overall > 170:
            tips.append(f"**Too fast** ({pace.wpm_overall:.0f} wpm). Target 140–150. Insert a full one-second pause after every key statement.")
        elif pace.wpm_overall < 115:
            tips.append(f"**Slow overall pace** ({pace.wpm_overall:.0f} wpm overall vs. {pace.wpm_speaking:.0f} while speaking). The gap means long thinking pauses, not slow speech – tighten the structure so the next point is ready.")
        if pace.wpm_cv < 0.12:
            tips.append("**Metronomic pace.** Vary it: speed through setup, slow down for the point.")
    if pau.pauses_per_min > 20:
        tips.append(f"**Choppy delivery** ({pau.pauses_per_min:.0f} pauses/min). Many short hesitations break the flow – rehearse phrases as complete units so pauses land between ideas, not inside them.")
    if len(pau.long_pauses) > total / 60:
        tips.append(f"**{len(pau.long_pauses)} pauses over 2 s.** Silence is fine when deliberate; these look like searching for words. Prepare transitions.")
    if enu and enu.mean_word_confidence < 0.85:
        tips.append(f"**Enunciation** – recognizer confidence {enu.mean_word_confidence:.2f}, {enu.low_confidence_pct:.0f}% of words unclear. Over-articulate final consonants (t, d, k, p) in practice; tongue-twisters daily for 3 minutes.")
    if enu and enu.short_word_pct > 25:
        tips.append(f"**Rushing long words** ({enu.short_word_pct:.0f}% of 3+ syllable words). Slow the long words down – give every syllable its vowel.")
    if cla.high_band_db_delta < -30:
        tips.append("**Muffled consonants** (very little 4–8 kHz energy). Part of this may be the mic, but also open the mouth more and crisp the consonants.")
    if flu and flu.fillers_per_min > 3:
        top = ", ".join(f"'{k}'" for k in list(flu.filler_counts)[:3])
        tips.append(f"**Fillers** ({flu.fillers_per_min:.1f}/min; mostly {top}). Replace each with a silent pause – silence reads as confidence, fillers as uncertainty.")
    if flu and len(flu.repetitions) > total / 60:
        tips.append(f"**Word repeats / restarts** ({len(flu.repetitions)}). Finish the thought before starting the sentence; slow the first three words of each sentence.")
    if vol.snr_db < 12:
        tips.append("**Record closer to the mouth next time** (phone 15–30 cm away, or a lapel mic). The current recording is noisy enough that fine enunciation metrics are approximate.")
    if not tips:
        tips.append("Solid fundamentals across the board – work on stylistic polish: stories, rhetorical questions, and deliberate silence.")
    for i, t in enumerate(tips, 1):
        L.append(f"{i}. {t}")
    L.append("")

    if transcript_text:
        L.append("## Appendix: Transcript\n")
        L.append(transcript_text.strip() + "\n")

    data = {"source": source, "duration_s": total, "findings": findings, "metrics": _clean(m)}
    return "\n".join(L), data


def dump_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
