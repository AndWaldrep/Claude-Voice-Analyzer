"""Rhetoric detection tested against canonical examples from the oratory canon."""
import numpy as np

from voice_analyzer import rhetoric as R
from voice_analyzer.transcribe import Word


def test_anaphora_on_churchill():
    t = ("We shall fight on the beaches. We shall fight on the landing grounds. "
         "We shall fight in the fields and in the streets.")
    f = R.figure_stats(t)
    assert any(x.text == "we shall fight" and x.n == 3 for x in f.anaphora)


def test_chiasmus_on_kennedy_but_not_on_parallel_repetition():
    chi = "Ask not what your country can do for you, ask what you can do for your country."
    assert R.find_chiasmus(chi)
    # plain anaphora contains an ABBA by accident; it must not be reported as chiasmus
    ana = "We shall fight on the beaches, we shall fight on the grounds, we shall fight in the fields."
    assert not R.find_chiasmus(ana)


def test_antithesis_and_tricolon():
    t = ("They will not be judged by the color of their skin but by the content of their character. "
         "We need energy, faith, devotion. ")
    f = R.figure_stats(t)
    assert f.antithesis and f.tricolon


def test_plain_prose_has_no_figures():
    t = ("The quarterly revenue increased because the team executed the plan. "
         "I went to the store and bought milk. The meeting is at three.")
    f = R.figure_stats(t)
    assert not f.anaphora and not f.antithesis and f.per_1000_words["all_figures"] < 25


def test_language_stats_distinguish_concrete_from_abstract():
    concrete = "The boy opened the door and saw his mother crying in the kitchen. Rain hit the window."
    abstract = ("Organizational complacency reflects institutional deterioration and the "
                "normalization of underperformance across management structures.")
    assert R.language_stats(concrete).concreteness_ratio > R.language_stats(abstract).concreteness_ratio
    assert R.language_stats(abstract).long_word_pct > R.language_stats(concrete).long_word_pct


def test_sentence_contrast_and_short_punch():
    varied = "I thought about this for a very long time, turning it over and over in my mind. Then I knew. It was wrong."
    flat = "I went to the store today. I bought some milk there. I drove home again. I put it away."
    assert R.language_stats(varied).sentence_len_cv > R.language_stats(flat).sentence_len_cv
    assert R.language_stats(varied).short_sentence_pct > 0


def test_pause_classification_splits_rhetorical_from_hesitation():
    # "the point lands." <1.0s> "then" ... vs "searching" <1.0s> "for"
    words = [Word("the", 0.0, 0.3, .9), Word("point", 0.3, 0.6, .9), Word("lands.", 0.6, 0.9, .9),
             Word("Then", 1.9, 2.2, .9), Word("searching", 2.2, 2.6, .9),
             Word("for", 3.6, 3.9, .9), Word("it.", 3.9, 4.2, .9),
             Word("Next", 12.0, 12.4, .9)]
    pu = R.classify_pauses(words)
    assert pu.rhetorical == 1 and pu.hesitation == 1
    assert pu.breaks == 1                        # the 7.8 s gap is a stop, not delivery
    assert 40 < pu.rhetorical_pct < 60


def test_structure_detects_story_opening_and_call_ending():
    t = ("One day I remember standing in my mother's kitchen. She said something I never forgot. "
         "It changed how I saw the work. So let us go and do the same thing in our own kitchens.")
    st = R.structure_stats(t)
    assert st.opens_with_story and st.closes_with_call


def test_audience_stats_prefer_acoustic_source():
    a = R.audience_stats(None, 600.0, acoustic=[(10.0, 1.2, "laughter"), (50.0, 4.0, "applause")])
    assert a.laughter == 1 and a.applause == 1 and abs(a.events_per_min - 0.2) < 1e-6
