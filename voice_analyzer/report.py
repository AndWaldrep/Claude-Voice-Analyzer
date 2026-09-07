"""Turn raw metrics into a coaching report (markdown + JSON)."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

SKIP = {"frame_times", "frame_db", "speech_mask", "times", "f0", "window_labels"}


def ts(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def _clean(o: Any) -> Any:
    if is_dataclass(o):
        o = asdict(o)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k not in SKIP}
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


def _table(rows: list[tuple[str, str, str]]) -> list[str]:
    return ["| Metric | Value | Note |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows] + [""]


# ------------------------------------------------------------------ per-speaker sections
def speaker_sections(m: dict, h: str, snr_db: float, total_s: float) -> tuple[list[str], dict, list[str]]:
    """Return (markdown lines, findings, coaching tips) for one speaker's metrics dict."""
    vol, pit, pau, cla, tail = m["volume"], m["pitch"], m["pauses"], m["clarity"], m["trailing"]
    pace, flu, enu = m.get("pace"), m.get("fluency"), m.get("enunciation")
    F: dict[str, dict] = {}
    L: list[str] = []

    dr = vol.dynamic_range_db
    vr = rate(dr, (5, 12), (3, 16))
    F["volume"] = {"rating": vr, "dynamic_range_db": dr, "drift_db_per_min": vol.drift_db_per_min}
    L.append(f"{h} Volume\n")
    rows = [("Average speech level", f"{vol.mean_db:.1f} dBFS", "recording level, not absolute loudness"),
            ("Loud vs. quiet passages (p90 − p10)", f"{dr:.1f} dB", f"**{vr}** – 5–12 dB of peak contrast is expressive without shouting"),
            ("Volume drift over the talk", f"{vol.drift_db_per_min:+.2f} dB/min", "negative = fading as the talk goes on"),
            ("Peak / clipping", f"{vol.peak_dbfs:.1f} dBFS / {vol.clipped_pct:.2f}% samples",
             "clipping present – mic too hot or speaker leaned in" if vol.clipped_pct > 0.05 else "no clipping")]
    if tail["n_phrases"]:
        rows.append(("Phrase endings", f"drop {tail['mean_drop_db']:.1f} dB in last 0.6 s",
                     f"{tail['pct_phrases_dropping_over_4db']:.0f}% of {tail['n_phrases']} phrases trail off by more than 4 dB"))
    L += _table(rows)
    mm = [v for v in vol.minute_means if v == v]
    if len(mm) > 1:
        L.append("Per-minute level (dBFS): " + ", ".join(f"{v:.0f}" for v in mm) + "\n")

    pr = rate(pit.range_semitones, (8, 16), (5, 20))
    F["pitch"] = {"rating": pr, "median_hz": pit.median_hz, "range_semitones": pit.range_semitones, "monotone_pct": pit.monotone_pct}
    vq_note = ("jitter <1%, shimmer <4%, HNR >15 dB is a clean, steady voice; vocal fry / strain push these up"
               if snr_db >= 15 else "**not reliable at this recording's noise level** – re-record close-mic to assess voice quality")
    L.append(f"{h} Pitch & Intonation\n")
    L += _table([
        ("Median pitch", f"{pit.median_hz:.0f} Hz", "typical adult male 85–155 Hz, female 165–255 Hz"),
        ("Working range (p5–p95)", f"{pit.p5_hz:.0f}–{pit.p95_hz:.0f} Hz = {pit.range_semitones:.1f} semitones", f"**{pr}** – engaging speakers use 8–16 st"),
        ("Pitch variability (std)", f"{pit.std_semitones:.1f} st", "under 2 st sounds flat"),
        ("Monotone stretches", f"{pit.monotone_pct:.0f}% of 3 s windows", "share of the talk that is essentially one note"),
        ("Voiced share of frames", f"{pit.voiced_pct:.0f}%", "low values = breathy / whispered / dropped syllables"),
        ("Voice quality", f"jitter {pit.jitter_pct:.2f}%, shimmer {pit.shimmer_pct:.1f}%, HNR {pit.hnr_db:.1f} dB", vq_note)])
    ms = [v for v in pit.minute_std_st if v == v]
    if len(ms) > 1:
        L.append("Per-minute pitch variability (semitones): " + ", ".join(f"{v:.1f}" for v in ms) + "\n")

    L.append(f"{h} Pace & Pauses\n")
    rows = []
    if pace:
        wr = rate(pace.wpm_overall, (130, 165), (110, 185))
        F["pace"] = {"rating": wr, "wpm_overall": pace.wpm_overall, "wpm_speaking": pace.wpm_speaking, "syllables_per_sec": pace.syllables_per_sec}
        rows += [("Words per minute (overall)", f"{pace.wpm_overall:.0f} wpm", f"**{wr}** – conversational 130–165; presentations 120–150"),
                 ("Words per minute while speaking", f"{pace.wpm_speaking:.0f} wpm", "pace during actual speech runs"),
                 ("Articulation rate", f"{pace.syllables_per_sec:.1f} syll/s", "4–5 normal; >5.5 sounds rushed"),
                 ("Pace variability", f"CV {pace.wpm_cv:.2f}", "0.15–0.35 is dynamic; <0.1 is metronomic")]
        if pace.fastest_windows:
            rows.append(("Fastest 30 s", ", ".join(f"{ts(t)} ({w:.0f} wpm)" for t, w in pace.fastest_windows), ""))
            rows.append(("Slowest 30 s", ", ".join(f"{ts(t)} ({w:.0f} wpm)" for t, w in pace.slowest_windows), ""))
    rows.append(("Pauses (≥0.3 s)", f"{pau.n_pauses} total, {pau.pauses_per_min:.1f}/min, mean {pau.mean_pause_s:.2f} s", "8–15/min with 0.5–1 s means is a well-punctuated delivery"))
    rows.append(("Long pauses (>2 s)", f"{len(pau.long_pauses)}", ", ".join(f"{ts(s)} ({d:.1f}s)" for s, d in pau.long_pauses[:8])))
    L += _table(rows)
    if pace and len(pace.minute_wpm) > 1:
        L.append("Per-minute word count: " + ", ".join(f"{v:.0f}" for v in pace.minute_wpm) + "\n")

    L.append(f"{h} Enunciation & Clarity\n")
    rows = [("Consonant energy (2–8 kHz share)", f"{cla.consonant_energy_ratio*100:.1f}%", "higher = crisper consonants (typical 5–15% for close-mic speech)"),
            ("High band vs. vowel band", f"{cla.high_band_db_delta:+.1f} dB", "very negative = soft, muffled consonants"),
            ("Spectral centroid", f"{cla.spectral_centroid_hz:.0f} Hz", "brightness of the voice")]
    if enu:
        er = rate(enu.mean_word_confidence, (0.85, 1.0), (0.75, 1.0))
        F["enunciation"] = {"rating": er, "mean_word_confidence": enu.mean_word_confidence,
                            "low_confidence_pct": enu.low_confidence_pct, "rushed_long_word_pct": enu.short_word_pct}
        rows += [("Recognizer word confidence", f"{enu.mean_word_confidence:.2f}", f"**{er}** – how easily a listener (an ASR model) can make out each word"),
                 ("Hard-to-recognize words", f"{enu.low_confidence_pct:.1f}%", "words the recognizer was <50% sure of"),
                 ("Rushed long words (3+ syllables)", f"{enu.short_word_pct:.0f}%",
                  "spoken under 120 ms per syllable; e.g. " + ", ".join(f"{ts(t)} “{w}”" for t, w in enu.rushed_examples[:6]))]
    L += _table(rows)
    if enu and enu.low_confidence_examples:
        L.append("Least intelligible words (timestamp, word, confidence): " +
                 ", ".join(f"{ts(t)} “{w}” {p:.2f}" for t, w, p in enu.low_confidence_examples[:12]) + "\n")

    if flu:
        fr = rate(flu.fillers_per_min, (0, 2), (0, 4))
        F["fluency"] = {"rating": fr, "fillers_per_min": flu.fillers_per_min,
                        "strong_fillers_per_min": flu.strong_fillers_per_min, "repetitions": len(flu.repetitions)}
        L.append(f"{h} Fluency & Word Choice\n")
        L += _table([("Filler words", f"{flu.fillers_total} ({flu.fillers_per_min:.1f}/min)", f"**{fr}** – under 2/min is polished"),
                     ("um / uh only", f"{flu.strong_fillers_per_min:.1f}/min", ""),
                     ("Immediate word repeats", f"{len(flu.repetitions)}", ", ".join(f"{ts(t)} “{w} {w}”" for t, w in flu.repetitions[:8])),
                     ("Sentence length", f"mean {flu.mean_sentence_words:.0f} words, longest {flu.longest_sentence_words}", ">30 words is hard to follow aloud")])
        if flu.filler_counts:
            L.append("Filler breakdown: " + ", ".join(f"{k} ×{v}" for k, v in flu.filler_counts.items()) + "\n")
        if flu.top_words:
            L.append("Most-used content words: " + ", ".join(f"{k} ×{v}" for k, v in flu.top_words) + "\n")

    # ------------------------------------------------------------ coaching tips
    tips: list[str] = []
    if vol.dynamic_range_db < 5:
        tips.append("**Volume is flat.** Add deliberate contrast: drop to near-quiet for the key sentence, then come back up. Mark 2–3 'punch' lines per minute in notes.")
    elif vol.dynamic_range_db > 12:
        tips.append("**Volume swings widely.** Keep the floor up: the quiet passages are falling out of range for listeners at the back.")
    if vol.drift_db_per_min < -0.25 and total_s >= 180:
        tips.append(f"**Volume fades over the talk** ({vol.drift_db_per_min:+.2f} dB/min). Reset breath and posture at each section change.")
    if tail["n_phrases"] and tail["pct_phrases_dropping_over_4db"] > 35:
        tips.append(f"**Trailing off at phrase ends** ({tail['pct_phrases_dropping_over_4db']:.0f}% of phrases drop >4 dB). Practice 'landing' the last word of each sentence at full support.")
    if pit.range_semitones < 8 or pit.monotone_pct > 40:
        tips.append(f"**Narrow pitch range** ({pit.range_semitones:.1f} st, {pit.monotone_pct:.0f}% monotone windows). Read aloud exaggerating: up on questions and new ideas, down to finish. Aim for >10 st.")
    if snr_db >= 15 and (pit.hnr_db < 12 or pit.jitter_pct > 1.5):
        tips.append("**Rough / strained voice quality** (low harmonics-to-noise or high jitter). Suggests vocal fry or pushing from the throat; work on breath support.")
    if pace:
        if pace.wpm_overall > 170:
            tips.append(f"**Too fast** ({pace.wpm_overall:.0f} wpm). Target 140–150. Insert a full one-second pause after every key statement.")
        elif pace.wpm_overall < 115 and pace.wpm_speaking - pace.wpm_overall > 25:
            tips.append(f"**Slow overall pace** ({pace.wpm_overall:.0f} wpm overall vs. {pace.wpm_speaking:.0f} while speaking). The gap is thinking pauses, not slow speech – tighten the structure so the next point is ready.")
        elif pace.wpm_overall < 115:
            tips.append(f"**Slow pace** ({pace.wpm_overall:.0f} wpm). Pick up energy on setup sentences; save the slow delivery for the point.")
        if pace.wpm_cv < 0.12 and total_s >= 120:
            tips.append("**Metronomic pace.** Vary it: speed through setup, slow down for the point.")
    if pau.pauses_per_min > 20:
        tips.append(f"**Choppy delivery** ({pau.pauses_per_min:.0f} pauses/min). Rehearse phrases as complete units so pauses land between ideas, not inside them.")
    if len(pau.long_pauses) > total_s / 60:
        tips.append(f"**{len(pau.long_pauses)} pauses over 2 s.** Silence is fine when deliberate; these look like searching for words. Prepare transitions.")
    if enu and enu.mean_word_confidence < 0.85:
        tips.append(f"**Enunciation** – recognizer confidence {enu.mean_word_confidence:.2f}, {enu.low_confidence_pct:.0f}% of words unclear. Over-articulate final consonants (t, d, k, p); tongue-twisters daily for 3 minutes.")
    if enu and enu.short_word_pct > 25:
        tips.append(f"**Rushing long words** ({enu.short_word_pct:.0f}% of 3+ syllable words). Give every syllable its vowel.")
    if cla.high_band_db_delta < -30:
        tips.append("**Muffled consonants** (very little 4–8 kHz energy). Partly the mic, but also open the mouth more and crisp the consonants.")
    if flu and flu.fillers_per_min > 3:
        top = ", ".join(f"'{k}'" for k in list(flu.filler_counts)[:3])
        tips.append(f"**Fillers** ({flu.fillers_per_min:.1f}/min; mostly {top}). Replace each with a silent pause – silence reads as confidence, fillers as uncertainty.")
    if flu and len(flu.repetitions) > total_s / 60:
        tips.append(f"**Word repeats / restarts** ({len(flu.repetitions)}). Finish the thought before starting the sentence; slow the first three words of each sentence.")
    return L, F, tips


def tone_section(st, h: str) -> list[str]:
    L = [f"{h} Tone & Emotional Undertone\n"]
    if not st.turns:
        L.append("_Not enough speech (turns ≥ 1.5 s) to assess tone._\n")
        return L
    b = st.baseline
    L.append(f"Baseline for this speaker: {b.get('energy', 0):.0f} dBFS, pitch offset {b.get('pitch', 0):+.1f} st from median, "
             f"{b.get('rate', 0):.1f} syll/s, pitch spread {b.get('spread', 0):.1f} st. Indicators below are relative to this baseline.\n")
    if st.summary_tags:
        n = len(st.turns)
        L.append("Pattern across turns: " + ", ".join(f"{k} ({v}/{n})" for k, v in sorted(st.summary_tags.items(), key=lambda x: -x[1])) + "\n")
    L.append("Notable moments (largest departures from baseline):\n")
    L.append("| Time | Arousal | Tags | What was said |\n|---|---|---|---|")
    for t in st.notable:
        cat = ""
        if t.categorical:
            top = max(t.categorical, key=t.categorical.get)
            cat = f" _(model: {top})_"
        txt = t.text if len(t.text) < 90 else t.text[:87] + "…"
        L.append(f"| {ts(t.start)} | {t.arousal:+.1f} | {', '.join(t.tags) or '—'}{cat} | {txt} |")
    L.append("")
    return L


# ------------------------------------------------------------------ whole report
def build_report(source: str, total_s: float, snr_db: float, noise_floor_db: float,
                 speakers: list[dict], interaction: dict | None, dialogue: list | None,
                 transcript_text: str | None) -> tuple[str, dict]:
    """speakers: list of {name, metrics, tone (optional)} in speaking order."""
    L: list[str] = []
    multi = len(speakers) > 1
    speech_s = sum(s["metrics"]["pauses"].speech_time_s for s in speakers)
    L.append(f"# Voice & Oratory Analysis\n\n**Source:** `{source}`  \n**Duration:** {ts(total_s)} "
             f"({min(100, speech_s/total_s*100):.0f}% speech) · **Speakers detected:** {len(speakers)}\n")
    q = "good" if snr_db >= 20 else "fair" if snr_db >= 12 else "poor"
    findings: dict = {"recording": {"rating": q, "snr_db": snr_db, "noise_floor_db": noise_floor_db}, "speakers": {}}
    L.append("## Recording Quality\n")
    L += _table([("Speech-to-noise ratio", f"{snr_db:.1f} dB", f"**{q}** – below ~12 dB the mic is far away or the room is loud; enunciation and volume metrics get less reliable"),
                 ("Noise floor", f"{noise_floor_db:.1f} dBFS", "")])

    if multi and interaction:
        L.append("## Conversation Overview\n")
        L.append("| Speaker | Talk share | Turns | Mean turn | Longest turn | Words | Cut in on others | Was cut in on |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s in speakers:
            i = interaction[s["name"]]
            L.append(f"| {s['name']} | {i['talk_share_pct']:.0f}% | {i['n_turns']} | {i['mean_turn_s']:.1f} s | {i['longest_turn_s']:.0f} s | {i['words']} | {i['interruptions_made']} | {i['interrupted']} |")
        L.append("")
        findings["interaction"] = interaction

    all_tips: dict[str, list[str]] = {}
    for s in speakers:
        name = s["name"]
        h = "###" if multi else "##"
        if multi:
            L.append(f"## {name}\n")
        secs, F, tips = speaker_sections(s["metrics"], h, snr_db, s["metrics"]["pauses"].total_time_s)
        L += secs
        if s.get("tone") is not None:
            L += tone_section(s["tone"], h)
        findings["speakers"][name] = F
        all_tips[name] = tips

    L.append("## Coaching Priorities\n")
    for name, tips in all_tips.items():
        if multi:
            L.append(f"**{name}**\n")
        if not tips:
            tips = ["Solid fundamentals across the board – work on stylistic polish: stories, rhetorical questions, and deliberate silence."]
        if snr_db < 12:
            tips.append("**Record closer to the mouth next time** (phone 15–30 cm away, or a lapel mic). The current recording is noisy enough that fine enunciation metrics are approximate.")
        for i, t in enumerate(tips, 1):
            L.append(f"{i}. {t}")
        L.append("")

    if dialogue:
        L.append("## Transcript (by speaker)\n")
        for t in dialogue:
            L.append(f"**[{ts(t.start)}] {t.speaker}:** {t.text}\n")
    elif transcript_text:
        L.append("## Transcript\n")
        L.append(transcript_text.strip() + "\n")

    data = {"source": source, "duration_s": total_s, "findings": findings,
            "speakers": {s["name"]: {"metrics": _clean(s["metrics"]), "tone": _clean(s.get("tone"))} for s in speakers},
            "dialogue": [{"start": t.start, "end": t.end, "speaker": t.speaker, "text": t.text} for t in (dialogue or [])]}
    return "\n".join(L), data


def dump_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
