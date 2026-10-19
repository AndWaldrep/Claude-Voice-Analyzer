from voice_analyzer.text_metrics import count_syllables, fluency_stats, pace_stats, enunciation_stats
from voice_analyzer.transcribe import Transcript, Word


def test_syllables():
    assert count_syllables("organization") == 5
    assert count_syllables("the") == 1
    assert count_syllables("complacency") == 4


def test_fillers_and_pace():
    words, t = [], 0.0
    for tok in "um so I think you know the plan is is good um".split():
        words.append(Word(tok, t, t + 0.3, 0.9))
        t += 0.4
    tr = Transcript(words=words, segments=[(0, t, " ".join(w.word for w in words))], language="en")
    flu = fluency_stats(tr, total_s=60)
    assert flu.filler_counts["um"] == 2
    assert flu.filler_counts.get("you know") == 1
    assert len(flu.repetitions) == 1
    pace = pace_stats(tr, total_s=60, speech_s=t)
    assert pace.n_words == 12 and pace.wpm_overall == 12
    enu = enunciation_stats(tr)
    assert 0 <= enu.low_confidence_pct <= 100


REFERENCE = [
    ("organization", 5), ("complacency", 4), ("information", 4), ("management", 3),
    ("statement", 2), ("movement", 2), ("the", 1), ("strength", 1), ("accountability", 6),
    ("responsibility", 6), ("opportunity", 5), ("people", 2), ("business", 2), ("idea", 3),
    ("area", 3), ("hour", 1), ("fire", 1), ("crispian", 3), ("ocean", 2), ("christian", 2),
    ("remembered", 3), ("freedom", 2), ("liberty", 3), ("nation", 2), ("dedicated", 4),
    ("consecrate", 3), ("devotion", 3), ("government", 3), ("perish", 2), ("earth", 1),
    ("walked", 1), ("wanted", 2), ("needed", 2), ("plea", 1), ("sea", 1), ("flea", 1),
    ("audacity", 4), ("conspiracy", 4), ("senate", 2), ("patience", 2), ("happiness", 3),
]


def test_syllable_reference_set():
    """The bare vowel-group rule got 72% of these right; the corrected one gets all of them."""
    wrong = [(w, count_syllables(w), want) for w, want in REFERENCE if count_syllables(w) != want]
    assert not wrong, f"mis-counted: {wrong}"


def test_syllable_counts_have_no_net_bias():
    got = sum(count_syllables(w) for w, _ in REFERENCE)
    want = sum(n for _, n in REFERENCE)
    assert abs(got - want) / want < 0.02, f"{got} vs {want}"
