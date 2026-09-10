"""Empirical benchmarks measured from a corpus of recognized speakers.

Every number here was produced by running this same pipeline over the recordings listed in
CORPUS (see docs/BENCHMARKS.md for licences and provenance). They are descriptive, not
prescriptive: the corpus spans a 1961 inaugural address and a Tony Robbins keynote, and those two
agree on almost nothing. Where the spread is wide, that itself is the finding -- there is no single
correct pace or pitch range, and a report should say so rather than push everyone to a median.
"""
from __future__ import annotations

CORPUS = [
    ('Amy Cuddy', 'Your body language shapes who you are', 21.0),
    ('Angela Duckworth', 'Grit', 6.2),
    ('Brené Brown', 'The power of vulnerability', 20.3),
    ('Chimamanda Ngozi Adichie', 'The danger of a single story', 18.8),
    ('Derek Sivers', 'How to start a movement', 3.2),
    ('John F. Kennedy', 'Inaugural Address 1961', 14.6),
    ('Ken Robinson', "How to escape education's death valley", 19.2),
    ('Rita Pierson', 'Every kid needs a champion', 7.8),
    ('Sarah Kay', 'If I should have a daughter', 18.5),
    ('Seth Godin', 'Standing out', 17.1),
    ('Simon Sinek', 'How great leaders inspire action', 18.1),
    ('Susan Cain', 'The power of introverts', 19.1),
    ('Tony Robbins', 'Why we do what we do', 21.8),
]

# metric -> (p10, p25, p50, p75, p90) across the corpus
PERCENTILES: dict[str, tuple[float, float, float, float, float]] = {
    "wpm": (140.01, 141.88, 153.29, 175.64, 178.41),
    "wpm_speaking": (195.53, 206.58, 230.38, 240.25, 262.57),
    "syll_s": (4.29, 4.90, 5.09, 5.37, 5.46),
    "pitch_st": (9.42, 10.67, 11.77, 13.57, 14.83),
    "pitch_sd": (3.19, 3.60, 3.71, 4.25, 4.47),
    "monotone": (1.54, 9.27, 12.62, 17.50, 23.32),
    "dyn_db": (4.49, 4.58, 7.87, 10.06, 10.85),
    "pause_min": (14.05, 14.74, 15.77, 17.42, 17.86),
    "pause_mean": (0.72, 0.82, 0.86, 1.06, 1.10),
    "rhet_pct": (58.17, 63.25, 68.08, 74.23, 77.15),
    "rhet_s": (0.75, 0.80, 0.92, 1.13, 1.24),
    "fillers": (0.66, 0.76, 1.41, 2.71, 3.91),
    "fig_1k": (11.63, 14.80, 22.67, 27.60, 28.99),
    "q_1k": (1.54, 3.49, 5.76, 8.97, 9.05),
    "concrete": (0.43, 0.51, 0.62, 1.18, 2.52),
    "longword": (6.26, 7.65, 8.09, 9.37, 10.49),
    "sent_med": (9.20, 10.00, 11.00, 14.00, 17.60),
    "sent_cv": (0.63, 0.65, 0.73, 0.78, 0.83),
    "short_sent": (14.78, 17.32, 24.05, 30.13, 32.53),
    "you": (0.47, 1.55, 1.99, 2.56, 2.92),
    "we": (0.53, 0.63, 1.21, 1.61, 1.75),
    "i": (0.39, 1.44, 2.03, 3.46, 4.11),
    "narr": (0.09, 0.18, 0.51, 0.95, 1.44),
}

LABELS: dict[str, str] = {
    "wpm": 'Words per minute (overall)',
    "wpm_speaking": 'Words per minute while speaking',
    "syll_s": 'Articulation rate (syllables/s)',
    "pitch_st": 'Pitch range (semitones)',
    "pitch_sd": 'Pitch variability (semitones)',
    "monotone": 'Monotone stretches (%)',
    "dyn_db": 'Volume contrast (dB)',
    "pause_min": 'Pauses per minute',
    "pause_mean": 'Mean pause (s)',
    "rhet_pct": 'Pauses that land after a full stop (%)',
    "rhet_s": 'Mean deliberate silence (s)',
    "fillers": 'Fillers per minute',
    "fig_1k": 'Rhetorical figures per 1000 words',
    "q_1k": 'Rhetorical questions per 1000 words',
    "concrete": 'Concrete-to-abstract word ratio',
    "longword": 'Words of 3+ syllables (%)',
    "sent_med": 'Median sentence length (words)',
    "sent_cv": 'Sentence-length contrast (CV)',
    "short_sent": 'Short punch sentences (%)',
    "you": "'you' per 100 words",
    "we": "'we' per 100 words",
    "i": "'I' per 100 words",
    "narr": 'Narrative cues per 100 words',
}

# Metrics where a lower value is the better one.
LOWER_IS_BETTER = ['fillers', 'longword', 'monotone']



def percentile_of(metric: str, value: float) -> float | None:
    """Approximate percentile of `value` within the corpus, by linear interpolation between the
    stored decile points. Returns None for an unknown metric."""
    pts = PERCENTILES.get(metric)
    if pts is None or value is None:
        return None
    qs = (10.0, 25.0, 50.0, 75.0, 90.0)
    if value <= pts[0]:
        return max(0.0, 10.0 * value / pts[0]) if pts[0] > 0 else 0.0
    if value >= pts[-1]:
        return 95.0
    for (a, b), (qa, qb) in zip(zip(pts, pts[1:]), zip(qs, qs[1:])):
        if a <= value <= b:
            return qa + (qb - qa) * ((value - a) / (b - a) if b > a else 0.0)
    return None


def compare(metric: str, value: float) -> tuple[float | None, str]:
    """Return (percentile, one-line verdict) for a measured value against the corpus.

    The verdict is derived from the percentile itself so the words and the number never disagree.
    For metrics in LOWER_IS_BETTER a high percentile is the bad direction, and the wording says so."""
    p = percentile_of(metric, value)
    if p is None:
        return None, ""
    inverted = metric in LOWER_IS_BETTER
    if p < 10:
        return p, "more than the whole corpus" if inverted else "below the whole corpus"
    if p < 30:
        return p, "lower than most of the corpus"
    if p <= 70:
        return p, "mid-range for the corpus"
    if p <= 90:
        return p, "high end of the corpus"
    return p, "higher than the whole corpus" if inverted else "above the whole corpus"
