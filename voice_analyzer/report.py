"""Turn raw metrics into a coaching report (markdown + JSON)."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from . import benchmarks

SKIP = {"frame_times", "frame_db", "speech_mask", "times", "f0", "window_labels", "opening", "closing"}


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
            ("Loud vs. quiet passages (p90 − p10)", f"{dr:.1f} dB", f"**{vr}** – corpus runs 4.5–10.9 dB (median 7.9)"),
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

    pr = rate(pit.range_semitones, (9, 17), (6, 20))     # corpus p10 9.4, p90 14.8, max 17.2
    F["pitch"] = {"rating": pr, "median_hz": pit.median_hz, "range_semitones": pit.range_semitones, "monotone_pct": pit.monotone_pct}
    vq_note = ("jitter <1%, shimmer <4%, HNR >15 dB is a clean, steady voice; vocal fry / strain push these up"
               if snr_db >= 15 else "**not reliable at this recording's noise level** – re-record close-mic to assess voice quality")
    L.append(f"{h} Pitch & Intonation\n")
    L += _table([
        ("Median pitch", f"{pit.median_hz:.0f} Hz", "typical adult male 85–155 Hz, female 165–255 Hz"),
        ("Working range (p5–p95)", f"{pit.p5_hz:.0f}–{pit.p95_hz:.0f} Hz = {pit.range_semitones:.1f} semitones", f"**{pr}** – corpus runs 9.4–14.8 st (median 11.8)"),
        ("Pitch variability (std)", f"{pit.std_semitones:.1f} st", "under 2 st sounds flat"),
        ("Monotone stretches", f"{pit.monotone_pct:.0f}% of 3 s windows", "corpus median 13%, p90 23%; one note for long stretches"),
        ("Voiced share of frames", f"{pit.voiced_pct:.0f}%", "low values = breathy / whispered / dropped syllables"),
        ("Voice quality", f"jitter {pit.jitter_pct:.2f}%, shimmer {pit.shimmer_pct:.1f}%, HNR {pit.hnr_db:.1f} dB", vq_note)])
    ms = [v for v in pit.minute_std_st if v == v]
    if len(ms) > 1:
        L.append("Per-minute pitch variability (semitones): " + ", ".join(f"{v:.1f}" for v in ms) + "\n")

    L.append(f"{h} Pace & Pauses\n")
    rows = []
    if pace:
        # Corpus of 13 recognized speakers: p10 140, median 153, p90 178, max 218 (Tony Robbins).
        # The conventional "130-165" advice would flag six of them as too fast.
        wr = rate(pace.wpm_overall, (130, 190), (110, 215))
        F["pace"] = {"rating": wr, "wpm_overall": pace.wpm_overall, "wpm_speaking": pace.wpm_speaking, "syllables_per_sec": pace.syllables_per_sec}
        rows += [("Words per minute (overall)", f"{pace.wpm_overall:.0f} wpm", f"**{wr}** – corpus of great speakers runs 140–178 (median 153); conventional advice understates this"),
                 ("Words per minute while speaking", f"{pace.wpm_speaking:.0f} wpm", "pace during actual speech runs"),
                 ("Articulation rate", f"{pace.syllables_per_sec:.1f} syll/s", "corpus median 5.1, p90 5.5; above ~6 starts to sound rushed"),
                 ("Pace variability", f"CV {pace.wpm_cv:.2f}", "0.15–0.35 is dynamic; <0.1 is metronomic")]
        if pace.fastest_windows:
            rows.append(("Fastest 30 s", ", ".join(f"{ts(t)} ({w:.0f} wpm)" for t, w in pace.fastest_windows), ""))
            rows.append(("Slowest 30 s", ", ".join(f"{ts(t)} ({w:.0f} wpm)" for t, w in pace.slowest_windows), ""))
    rows.append(("Pauses (≥0.3 s)", f"{pau.n_pauses} total, {pau.pauses_per_min:.1f}/min, mean {pau.mean_pause_s:.2f} s", "corpus median 15.8/min with 0.86 s mean — strong speakers pause far more often than expected"))
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
    if pau.pauses_per_min > 25:      # corpus p90 is 17.9/min; frequent pausing is normal, not choppy
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


BENCH_ROWS = [
    ("wpm", "pace", lambda m: getattr(m.get("pace"), "wpm_overall", None)),
    ("syll_s", "pace", lambda m: getattr(m.get("pace"), "syllables_per_sec", None)),
    ("pitch_st", "pitch", lambda m: m["pitch"].range_semitones),
    ("monotone", "pitch", lambda m: m["pitch"].monotone_pct),
    ("dyn_db", "volume", lambda m: m["volume"].dynamic_range_db),
    ("pause_min", "pause", lambda m: m["pauses"].pauses_per_min),
    ("rhet_pct", "pause", lambda m: getattr(getattr(m.get("rhetoric"), "pause_use", None), "rhetorical_pct", None)),
    ("fillers", "fluency", lambda m: getattr(m.get("fluency"), "fillers_per_min", None)),
    ("fig_1k", "rhetoric", lambda m: (m["rhetoric"].figures.per_1000_words["all_figures"] if m.get("rhetoric") else None)),
    ("concrete", "language", lambda m: (m["rhetoric"].language.concreteness_ratio if m.get("rhetoric") else None)),
    ("sent_cv", "language", lambda m: (m["rhetoric"].language.sentence_len_cv if m.get("rhetoric") else None)),
    ("short_sent", "language", lambda m: (m["rhetoric"].language.short_sentence_pct if m.get("rhetoric") else None)),
    ("narr", "language", lambda m: (m["rhetoric"].language.narrative_cues_per_100 if m.get("rhetoric") else None)),
]


def benchmark_section(m: dict, h: str) -> tuple[list[str], dict]:
    """Compare this speaker against the measured corpus of recognized speakers."""
    L = [f"{h} Against the benchmark corpus\n"]
    L.append(f"Percentile against {len(benchmarks.CORPUS)} recorded speeches by recognized speakers "
             f"({', '.join(c[0] for c in benchmarks.CORPUS[:4])} and others), measured with this same pipeline. "
             "The corpus disagrees with itself on almost everything, so treat a low percentile as a question, "
             "not a verdict.\n")
    L.append("| Metric | This speaker | Corpus p10–p90 (median) | Percentile |")
    L.append("|---|---|---|---|")
    out: dict[str, float] = {}
    for key, _group, get in BENCH_ROWS:
        try:
            v = get(m)
        except Exception:
            v = None
        if v is None or key not in benchmarks.PERCENTILES:
            continue
        p, verdict = benchmarks.compare(key, float(v))
        lo, _q, med, _q3, hi = benchmarks.PERCENTILES[key]
        out[key] = p
        bar = "▓" * max(1, round(p / 12)) if p is not None else ""
        L.append(f"| {benchmarks.LABELS.get(key, key)} | {float(v):.1f} | {lo:.1f}–{hi:.1f} ({med:.1f}) | "
                 f"p{p:.0f} {bar} {verdict} |")
    L.append("")
    return L, out


def rhetoric_section(r, h: str, total_s: float) -> tuple[list[str], list[str]]:
    """Composition analysis: the figures, the language texture, the shape. Returns (lines, tips)."""
    fig, lang, st = r.figures, r.language, r.structure
    L = [f"{h} Rhetoric & Composition\n"]
    L.append("_What the Greeks called elocutio and dispositio: the words themselves and their arrangement. "
             "The delivery sections above are Cicero's pronuntiatio._\n")
    rows = [
        ("Rhetorical figures", f"{fig.total} total, {fig.per_1000_words['all_figures']:.0f} per 1000 words",
         "corpus runs 12–29 per 1000 words (median 23); a status update runs near 0"),
        ("Anaphora (repeated openings)", f"{len(fig.anaphora)}",
         ", ".join(f"“{f.text}” ×{f.n}" for f in fig.anaphora[:4]) or "none — the single most powerful device in oratory"),
        ("Tricolon (rule of three)", f"{len(fig.tricolon)}",
         "; ".join(f.text[:52] for f in fig.tricolon[:3]) or "none"),
        ("Antithesis / chiasmus", f"{len(fig.antithesis)}",
         "; ".join(f.text[:52] for f in fig.antithesis[:3]) or "none"),
        ("Rhetorical questions", f"{len(fig.questions)} ({fig.per_1000_words['rhetorical_question']:.1f}/1000 words)",
         "; ".join(q[:48] for q in fig.questions[:2]) or "none"),
    ]
    if fig.epistrophe:
        rows.append(("Epistrophe (repeated endings)", f"{len(fig.epistrophe)}",
                     ", ".join(f"“{f.text}” ×{f.n}" for f in fig.epistrophe[:3])))
    L += _table(rows)

    L.append(f"{h}# Language texture\n")
    L += _table([
        ("Concrete & sensory vs. abstract words", f"{lang.concrete_sensory_per_100:.1f} vs {lang.abstract_noun_per_100:.1f} per 100",
         f"ratio {lang.concreteness_ratio:.1f} — corpus median 0.62; the storytellers reach 1.2–3.4"),
        ("Long words (3+ syllables)", f"{lang.long_word_pct:.0f}%", "corpus runs 6–10%; above ~14% the register turns academic"),
        ("Sentence length", f"median {lang.median_sentence_words:.0f}, mean {lang.mean_sentence_words:.0f} words",
         f"contrast CV {lang.sentence_len_cv:.2f} — 0.6+ means real long/short variety"),
        ("Short punch sentences (≤6 words)", f"{lang.short_sentence_pct:.0f}%", "corpus runs 15–33% (median 24%)"),
        ("Overlong sentences (>30 words)", f"{lang.long_sentence_pct:.0f}%", "hard to follow aloud"),
        ("Direct address", f"you {lang.pronouns['you']:.1f}, we {lang.pronouns['we']:.1f}, I {lang.pronouns['i']:.1f} per 100 words",
         "'you' pulls the listener in; 'we' includes them; 'I' builds ethos through story"),
        ("Narrative cues", f"{lang.narrative_cues_per_100:.1f} per 100 words", "corpus median 0.51; the storytellers (Adiche, Pierson, Brown) run 1.4–1.6"),
        ("Metaphor / analogy cues", f"{lang.metaphor_cues_per_100:.1f} per 100 words", "Aristotle: mastery of metaphor is 'the mark of genius'"),
    ])
    if lang.key_phrase:
        L.append(f"Most repeated phrase (the throughline): “{lang.key_phrase[0]}” ×{lang.key_phrase[1]}\n")
    if lang.signposts:
        L.append("Signposting used: " + ", ".join(f"“{x}”" for x in lang.signposts) + "\n")

    shape = []
    shape.append("opens with a story" if st.opens_with_story else "opens with exposition")
    if st.opens_with_question: shape.append("opens with a question")
    if st.opens_with_stat_or_fact: shape.append("opens with a number")
    shape.append("closes with a call to action" if st.closes_with_call else "closes without a call to action")
    if st.closes_with_callback: shape.append("closes by echoing the opening (ring composition)")
    L.append(f"**Shape:** {', '.join(shape)}.\n")

    if r.pause_use:
        pu = r.pause_use
        L.append(f"{h}# How the pauses are used\n")
        L += _table([
            ("Rhetorical pauses (after a finished sentence)", f"{pu.rhetorical} ({pu.rhetorical_pct:.0f}% of pauses), mean {pu.mean_rhetorical_s:.2f} s",
             "the confident kind: the point lands, then silence (corpus median 68% of all pauses)"),
            ("Phrasing pauses (at a comma)", f"{pu.grammatical}", "breath and clause structure"),
            ("Hesitation pauses (mid-clause)", f"{pu.hesitation}, mean {pu.mean_hesitation_s:.2f} s",
             "the anxious kind: searching for the next word with the sentence still open"),
        ] + ([("Breaks (>5 s)", f"{pu.breaks}, {pu.break_time_s:.0f} s total",
               "too long to be delivery — a stop, an edit point or an interruption; excluded from the split above")]
             if pu.breaks else []))
        if pu.longest_rhetorical:
            L.append("Longest deliberate silences: " +
                     ", ".join(f"{ts(t)} ({d:.1f}s) after “…{w[-42:]}”" for t, d, w in pu.longest_rhetorical[:3]) + "\n")

    if r.audience and (r.audience.laughter or r.audience.applause):
        a = r.audience
        L.append(f"{h}# Audience response\n")
        L += _table([("Laughter", f"{a.laughter}", "moments the room audibly reacted"),
                     ("Applause", f"{a.applause}", ""),
                     ("Reactions per minute", f"{a.events_per_min:.2f}", "the only direct evidence in the audio that the room was with them")])

    # ---- composition coaching
    tips = []
    if fig.per_1000_words["all_figures"] < 10:      # corpus p10 = 11.6
        tips.append("**Almost no rhetorical figures.** The delivery may be fine while the writing is flat. "
                    "Add one anaphora (start three consecutive sentences the same way) and one tricolon to the key section — "
                    "these two devices carry most of the memorable lines in the canon.")
    if not fig.anaphora:
        tips.append("**No anaphora anywhere.** 'I have a dream', 'We shall fight', 'Let both sides' — repeating an opening "
                    "across three clauses is the cheapest way to make a passage feel inevitable. Pick the central claim and build one.")
    if len(fig.tricolon) < max(1, total_s / 300):
        tips.append("**Few threes.** Three is the smallest number that makes a pattern, which is why 'life, liberty and the "
                    "pursuit of happiness' outlives every four-item list. Convert the main list to exactly three.")
    if lang.concreteness_ratio < 0.35:      # corpus p10 = 0.43; the earlier 0.7 cut flagged the corpus median
        tips.append(f"**Abstraction outweighs image** ({lang.abstract_noun_per_100:.1f} abstract nouns per 100 words vs "
                    f"{lang.concrete_sensory_per_100:.1f} concrete). Replace the biggest abstraction with one specific person, "
                    "place or object doing something. Nobody pictures 'organizational complacency'; they picture a runner who came in last.")
    if lang.narrative_cues_per_100 < 0.15:      # corpus p10 = 0.09; median only 0.51
        tips.append("**Little narrative.** The talk explains rather than recounts. One 60-second story with a named person, "
                    "a place and a turn will be remembered when the argument is forgotten.")
    if lang.sentence_len_cv < 0.5:
        tips.append(f"**Sentences are all the same length** (CV {lang.sentence_len_cv:.2f}). Rhythm comes from contrast: "
                    "follow a long, winding sentence with a four-word one. Like this.")
    if lang.short_sentence_pct < 12:      # corpus p10 = 14.8
        tips.append("**No short sentences to land on.** End each section with something under six words.")
    if lang.long_word_pct > 14:      # corpus p90 = 10.5
        tips.append(f"**Latinate register** ({lang.long_word_pct:.0f}% of words are 3+ syllables). Swap the longest word in "
                    "each paragraph for a shorter one; the ear prefers Anglo-Saxon.")
    if not fig.questions:
        tips.append("**No questions asked.** A rhetorical question makes the listener answer in their head, which is "
                    "participation rather than reception. Aristotle's enthymeme works the same way: leave the last step to them.")
    if lang.pronouns["you"] < 0.4:      # corpus p10 = 0.47, but JFK and Adichie sit there by design
        tips.append(f"**Rarely says 'you'** ({lang.pronouns['you']:.1f} per 100 words). Direct address is the difference "
                    "between a lecture and a conversation.")
    if not st.opens_with_story and not st.opens_with_question:
        tips.append("**The opening explains instead of hooking.** The first 30 seconds decide whether the room leans in. "
                    "Open with a scene, a question, or a number that shouldn't be true.")
    if not st.closes_with_call:
        tips.append("**No clear ending.** Classical peroratio does two things: it recapitulates and it asks. "
                    "Decide the one sentence you want repeated afterwards, and end on it.")
    if r.pause_use and r.pause_use.rhetorical_pct < 45 and r.pause_use.hesitation > 10:
        tips.append(f"**Pauses fall in the wrong places.** Only {r.pause_use.rhetorical_pct:.0f}% land after a completed "
                    "sentence, against a corpus median of 68% and a floor of 58% across every great speaker measured. "
                    "This is the single most discriminating metric in the whole analysis: the same quantity of silence "
                    "reads as command when it follows a full stop and as anxiety when it interrupts a clause. "
                    "Move the pause to *after* the point instead of before the next word.")
    return L, tips


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
        bl, bpct = benchmark_section(s["metrics"], h)
        L += bl
        findings.setdefault("benchmark_percentiles", {})[name] = bpct
        rhet = s["metrics"].get("rhetoric")
        if rhet is not None:
            rl, rtips = rhetoric_section(rhet, h, s["metrics"]["pauses"].total_time_s)
            L += rl
            tips = tips + rtips
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
