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
