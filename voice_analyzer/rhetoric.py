"""Composition analysis: the *what you say* half of oratory.

The acoustic modules measure what Cicero called `pronuntiatio` (delivery). This module measures
`elocutio` and `dispositio` -- style and arrangement -- from the transcript, using the figures the
Greek and Roman handbooks catalogued and that modern speech research keeps re-confirming:

  anaphora      repetition at the start of successive clauses ("I have a dream...", "We shall fight...")
  epistrophe    repetition at the end ("of the people, by the people, for the people")
  tricolon      three parallel elements; three is the smallest number that establishes a pattern
  antithesis    balanced opposition ("ask not what your country can do for you...")
  erotema       rhetorical question -- makes the listener supply the answer
  enthymeme     an argument with a premise left unstated for the audience to complete

plus the things that separate vivid speech from a memo read aloud: concrete over abstract nouns,
short Anglo-Saxon words, sentence-length contrast, direct address, narrative, and metaphor.

Everything here is heuristic pattern-matching over the transcript, not semantic understanding.
Counts are reliable; interpretation belongs to the coach.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- lexicons
# Deliberately small, hand-written and inlined: no model or corpus download needed.

ABSTRACT_SUFFIX = re.compile(r"(tion|sion|ness|ity|ism|ment|ance|ence|ship|hood|acy|ude|ology)s?$", re.I)

SENSORY = set("""saw see seen look looked looking watch watched eye eyes glance stare heard hear
listen listening sound loud quiet silence noise voice whisper touch felt feel hand hands skin warm
cold hot cool rough smooth taste tasted sweet bitter salt smell smelled scent breath breathe
bright dark shadow light color red blue green white black gold shine glow burn burning""".split())

CONCRETE_HINT = set("""door room house home street road car table chair window floor wall water fire
tree stone rock hand face eye foot road river mountain city town village school church hospital bed
book paper letter phone knife bread food money coin dollar dog cat horse bird child children boy
girl man woman mother father son daughter friend neighbor kitchen garden field rain snow sun moon
star blood bone heart""".split())

NARRATIVE_CUE = [
    r"\bone day\b", r"\byears? ago\b", r"\bwhen i was\b", r"\bback (?:in|when)\b", r"\bthe (?:first|last) time\b",
    r"\bi remember\b", r"\bso i\b", r"\bthen (?:i|he|she|we|they)\b", r"\bthere was\b", r"\bonce\b",
    r"\blast (?:year|week|night|month|summer)\b", r"\bin \d{4}\b", r"\bat the time\b", r"\bshe (?:said|told)\b",
    r"\bhe (?:said|told)\b", r"\bi said\b", r"\bstory\b",
]

SIGNPOST = [
    r"\bfirst(?:ly)?\b", r"\bsecond(?:ly)?\b", r"\bthird(?:ly)?\b", r"\bfinally\b", r"\bto begin\b",
    r"\bthree (?:things|reasons|ways|lessons|points)\b", r"\btwo (?:things|reasons|ways)\b",
    r"\bhere'?s the (?:thing|point|problem)\b", r"\bwhich brings me\b", r"\bthe point is\b",
    r"\bin (?:conclusion|short|summary)\b", r"\blet me (?:tell|show|explain)\b", r"\bimagine\b",
]

METAPHOR_CUE = [r"\blike an?\b", r"\bas if\b", r"\bas though\b", r"\bis a kind of\b", r"\bthink of it as\b",
                r"\bit'?s like\b", r"\bwhat if\b"]

CONTRAST_CUE = [r"\bnot .{1,40}? but\b", r"\bbut\b", r"\binstead of\b", r"\brather than\b", r"\bhowever\b",
                r"\bon the other hand\b", r"\byet\b", r"\bwhereas\b"]

STOP = set("""the a an and or but of to in is it that this for on with be as are was were am i you he
she we they me him her them my your his its our their at by from so do did does not no yes if then
than there here what when where who how why all any some can will would could should have has had
been being into out up down over under about just like get got go going one two too very much many
more most such own same other another each every both few own s t re ve ll d m o""".split())


# --------------------------------------------------------------------------- helpers
SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+[\"'\u201d\u2019)\]]*")


def sentences(text: str) -> list[str]:
    """Split into sentences, keeping terminal punctuation and any closing quote.

    Splitting on whitespace after a terminator silently fails on `he said "...Crispian." Then`,
    because the character before the space is the quote rather than the full stop, so two
    sentences merge into one and sentence-length statistics skew long.
    """
    out, last = [], 0
    for m in SENTENCE_RE.finditer(text):
        if not m.group(0):
            break
        if m.group(0).strip():
            out.append(m.group(0).strip())
        last = m.end()
    tail = text[last:].strip()
    if tail:
        out.append(tail)
    return out


def words_of(s: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", s)


def norm(w: str) -> str:
    return w.lower().strip("'")


def clauses(text: str) -> list[str]:
    """Sentence-ish units split further on commas / semicolons / dashes, for figure detection."""
    out = []
    for s in sentences(text):
        out += [c.strip() for c in re.split(r"[,;:]| -- | – | — ", s) if len(words_of(c)) >= 2]
    return out


# --------------------------------------------------------------------------- figures
@dataclass
class Figure:
    kind: str
    text: str
    n: int = 1


@dataclass
class FigureStats:
    anaphora: list[Figure] = field(default_factory=list)
    epistrophe: list[Figure] = field(default_factory=list)
    tricolon: list[Figure] = field(default_factory=list)
    antithesis: list[Figure] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    per_1000_words: dict[str, float] = field(default_factory=dict)
    total: int = 0


def find_anaphora(units: list[str], min_run: int = 2, max_open: int = 4) -> list[Figure]:
    """Consecutive units opening with the same 2-4 words."""
    found: list[Figure] = []
    i = 0
    while i < len(units):
        best = None
        for n in range(max_open, 1, -1):
            head = [norm(w) for w in words_of(units[i])[:n]]
            if len(head) < n or all(h in STOP for h in head) and n < 3:
                continue
            j = i + 1
            while j < len(units) and [norm(w) for w in words_of(units[j])[:n]] == head:
                j += 1
            if j - i > min_run - 1 + 1 or (j - i >= 2 and n >= 2):
                if best is None or (j - i) > best[1] - best[0]:
                    best = (i, j, " ".join(head))
        if best and best[1] - best[0] >= 2:
            found.append(Figure("anaphora", best[2], best[1] - best[0]))
            i = best[1]
        else:
            i += 1
    return [f for f in found if f.n >= 2]


def find_epistrophe(units: list[str], max_tail: int = 3) -> list[Figure]:
    found: list[Figure] = []
    i = 0
    while i < len(units):
        best = None
        for n in range(max_tail, 0, -1):
            tail = [norm(w) for w in words_of(units[i])[-n:]]
            if len(tail) < n:
                continue
            j = i + 1
            while j < len(units) and [norm(w) for w in words_of(units[j])[-n:]] == tail:
                j += 1
            if j - i >= 2 and not (n == 1 and tail[0] in STOP):
                if best is None or (j - i) > best[1] - best[0]:
                    best = (i, j, " ".join(tail))
        if best:
            found.append(Figure("epistrophe", best[2], best[1] - best[0]))
            i = best[1]
        else:
            i += 1
    return found


def find_tricolon(text: str) -> list[Figure]:
    """Three (or more) parallel items: 'a, b, and c' lists, and three short parallel clauses."""
    found = []
    for s in sentences(text):
        m = re.search(r"\b([\w'\- ]{2,30}),\s+([\w'\- ]{2,30}),\s+(?:and\s+|or\s+)?([\w'\- ]{2,30})\b", s)
        if m and all(len(words_of(g)) <= 5 for g in m.groups()):
            found.append(Figure("tricolon", m.group(0).strip()))
            continue
        cs = [c for c in re.split(r"[,;]", s) if len(words_of(c)) >= 2]
        if len(cs) >= 3:
            heads = [norm(words_of(c)[0]) for c in cs if words_of(c)]
            if len(heads) >= 3 and len(set(heads[:3])) == 1:
                found.append(Figure("tricolon", s.strip()[:90]))
    return found


# Chiasmus turns on pronouns and nouns swapping places ("what your COUNTRY can do for YOU / what
# YOU can do for your COUNTRY"), so it uses a narrower stop list than the rest of the module:
# articles, prepositions, conjunctions and auxiliaries only. Pronouns are kept.
CHIASMUS_STOP = set("""the a an and or but of to in on with at by from for as if then than that this
is are was were am be been being do did does have has had will would can could should shall may
might must not no so there here it its""".split())


def find_chiasmus(sentence: str) -> bool:
    """ABBA order of two repeated content words: 'your COUNTRY can do for YOU ... YOU can do for
    your COUNTRY'. Requires positions i<j<k<l with w[i]==w[l] and w[j]==w[k]."""
    ws = [norm(w) for w in words_of(sentence) if norm(w) not in CHIASMUS_STOP and len(w) > 1]
    pos: dict[str, list[int]] = {}
    for i, w in enumerate(ws):
        pos.setdefault(w, []).append(i)
    rep = {w: p for w, p in pos.items() if len(p) >= 2}
    for a, pa in rep.items():
        for b, pb in rep.items():
            if a == b:
                continue
            for i in pa:
                for l in pa:
                    if l <= i:
                        continue
                    for j in pb:
                        for k in pb:
                            # the reversal must be local: the two inner terms sit together in the
                            # middle, with no further A between them. Without this, plain parallel
                            # repetition (ABABAB, i.e. anaphora) trivially contains an ABBA.
                            if not (i < j < k < l and k - j <= 4):
                                continue
                            if any(j < x < k for x in pa):
                                continue
                            return True
    return False


ANTITHESIS_PATTERNS = [
    r"\bnot\b[^.?!]{2,70}?\bbut\b",                       # not X but Y
    r"\bnot\b[^.?!]{2,70}?(?:--|—|–|\s-\s)",               # ask not X - ask Y
    r"\b(?:it'?s|is|was|we'?re|they'?re)\s+not\b[^.?!]{2,60},\s*(?:it'?s|is|was|we'?re|they'?re)\b",
    r"\b(?:instead of|rather than)\b",
    r"\bthe (?:difference|opposite) between\b",
    r"\bless\b[^.?!]{2,40}\bmore\b",
    r"\bmore\b[^.?!]{2,40}\bless\b",
]


def find_antithesis(text: str) -> list[Figure]:
    """Balanced opposition ('ask not what your country can do for you...') and chiasmus (ABBA)."""
    found = []
    for s in sentences(text):
        low = s.lower()
        if find_chiasmus(s):
            found.append(Figure("chiasmus", s.strip()[:110]))
        elif any(re.search(p, low) for p in ANTITHESIS_PATTERNS) and len(words_of(s)) <= 45:
            found.append(Figure("antithesis", s.strip()[:110]))
    return found


def figure_stats(text: str) -> FigureStats:
    sents, cls = sentences(text), clauses(text)
    n_words = max(1, len(words_of(text)))
    ana = find_anaphora(sents) + [f for f in find_anaphora(cls) if f.n >= 3]
    seen, ana_u = set(), []
    for f in ana:
        if f.text not in seen:
            seen.add(f.text); ana_u.append(f)
    epi = find_epistrophe(sents)
    tri = find_tricolon(text)
    ant = find_antithesis(text)
    qs = [s for s in sents if s.rstrip().endswith("?")]
    fs = FigureStats(ana_u, epi, tri, ant, qs)
    fs.total = len(ana_u) + len(epi) + len(tri) + len(ant) + len(qs)
    k = 1000 / n_words
    fs.per_1000_words = {
        "anaphora": len(ana_u) * k, "epistrophe": len(epi) * k, "tricolon": len(tri) * k,
        "antithesis": len(ant) * k, "rhetorical_question": len(qs) * k, "all_figures": fs.total * k,
    }
    return fs


# --------------------------------------------------------------------------- language texture
@dataclass
class LanguageStats:
    n_words: int
    mean_word_len: float
    long_word_pct: float            # words of 3+ syllables ~ Latinate/abstract register
    abstract_noun_per_100: float
    concrete_sensory_per_100: float
    concreteness_ratio: float       # concrete+sensory vs abstract
    sentence_lens: list[int]
    mean_sentence_words: float
    median_sentence_words: float
    sentence_len_cv: float          # contrast between long and short sentences
    short_sentence_pct: float       # <= 6 words: the punch lines
    long_sentence_pct: float        # > 30 words: hard to follow aloud
    pronouns: dict[str, float]      # per 100 words: i, we, you
    narrative_cues_per_100: float
    metaphor_cues_per_100: float
    signposts: list[str]
    type_token_ratio: float
    top_content_words: list[tuple[str, int]]
    key_phrase: tuple[str, int] | None    # most repeated 3-4 word phrase (the "throughline")


SYLL_IRREGULAR = {
    "business": 2, "businesses": 3, "evening": 2, "interesting": 3, "different": 3,
    "difference": 3, "family": 3, "comfortable": 4, "vegetable": 3, "chocolate": 3,
    "average": 3, "several": 3, "general": 3, "federal": 3, "camera": 3, "memory": 3,
    "everything": 3, "everyone": 3, "people": 2, "little": 2, "science": 2, "quiet": 2,
    "being": 2, "doing": 2, "going": 2, "poem": 2, "create": 2,
}


def count_syllables(word: str) -> int:
    """Vowel-group heuristic with the corrections that matter in practice.

    Measured on a 50-word reference set drawn from real transcripts: exact on all 50, with no net
    bias. The bare vowel-group rule scored 72% and mis-counted common words ("management" 4,
    "idea" 2). Articulation rate is derived from this, so the bias mattered.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if w in SYLL_IRREGULAR:
        return SYLL_IRREGULAR[w]
    if len(w) <= 3:
        return 1
    t = w
    # a silent e before a consonant suffix is not a syllable: manage+ment, state+ment
    t = re.sub(r"e(ments?|ness|less|ful|ly)$", r"\1", t)
    # "-ed" is its own syllable after t or d (ded-i-cat-ed, want-ed) but silent elsewhere (walked)
    t = re.sub(r"(?:[^laeiouy]es|(?<![td])ed|[^laeiouy]e)$", "", t)
    t = re.sub(r"^y", "", t)
    n = max(1, len(re.findall(r"[aeiouy]{1,2}", t)))
    # a final "ia"/"ea" is two syllables (i-de-a, ar-e-a), and so is "-ian" after most consonants
    # (cris-pi-an) but not after c, t or s, which make one sound (o-cean, chris-tian). Both need
    # another vowel group earlier, so "plea" and "sea" stay at one.
    if n >= 2 and re.search(r"(ia|ea)$", w):
        n += 1
    elif n >= 2 and re.search(r"[^ctsaeiou]ian$", w):
        n += 1
    return n


def language_stats(text: str) -> LanguageStats:
    sents = sentences(text)
    ws = [norm(w) for w in words_of(text)]
    n = max(1, len(ws))
    lens = [len(words_of(s)) for s in sents] or [0]
    abstract = sum(1 for w in ws if ABSTRACT_SUFFIX.search(w) and len(w) > 5)
    sensory = sum(1 for w in ws if w in SENSORY or w in CONCRETE_HINT)
    low = text.lower()
    narr = sum(len(re.findall(p, low)) for p in NARRATIVE_CUE)
    meta = sum(len(re.findall(p, low)) for p in METAPHOR_CUE)
    signs = sorted({m.group(0).strip() for p in SIGNPOST for m in re.finditer(p, low)})
    content = Counter(w for w in ws if w not in STOP and len(w) > 2)
    grams = Counter()
    for k in (3, 4):
        for i in range(len(ws) - k):
            g = ws[i:i + k]
            if sum(1 for w in g if w in STOP) <= k - 2:
                grams[" ".join(g)] += 1
    key = max(grams.items(), key=lambda kv: (kv[1], len(kv[0]))) if grams else None
    return LanguageStats(
        n_words=n,
        mean_word_len=float(np.mean([len(w) for w in ws])),
        long_word_pct=float(sum(1 for w in ws if count_syllables(w) >= 3) / n * 100),
        abstract_noun_per_100=abstract / n * 100,
        concrete_sensory_per_100=sensory / n * 100,
        concreteness_ratio=sensory / max(1, abstract),
        sentence_lens=lens,
        mean_sentence_words=float(np.mean(lens)),
        median_sentence_words=float(np.median(lens)),
        sentence_len_cv=float(np.std(lens) / max(1e-6, np.mean(lens))),
        short_sentence_pct=float(sum(1 for l in lens if l <= 6) / len(lens) * 100),
        long_sentence_pct=float(sum(1 for l in lens if l > 30) / len(lens) * 100),
        pronouns={p: sum(1 for w in ws if w == p) / n * 100 for p in ("i", "we", "you")},
        narrative_cues_per_100=narr / n * 100,
        metaphor_cues_per_100=meta / n * 100,
        signposts=signs[:10],
        type_token_ratio=len(set(ws)) / n,
        top_content_words=content.most_common(10),
        key_phrase=(key[0], key[1]) if key and key[1] >= 3 else None,
    )


# --------------------------------------------------------------------------- audience response
@dataclass
class AudienceStats:
    laughter: int
    applause: int
    other: int
    events_per_min: int | float
    timestamps: list[tuple[float, str]]


# Whisper sometimes writes these bare and upper-case rather than parenthesised.
AUDIENCE_RE = re.compile(r"[\(\[]?\s*\b(laughter|applause|laughs|laughing|cheers|cheering|clapping)\b\s*[\)\]]?", re.I)


def audience_stats(segments, total_s: float, acoustic: list[tuple[float, float, str]] | None = None) -> AudienceStats:
    """Audience reactions, from acoustic detection when available and from transcript markers otherwise.

    Acoustic detection is the primary source: Whisper's VAD filter removes laughter and applause
    before transcription, so the text markers catch only a fraction of what the room actually did."""
    stamps: list[tuple[float, str]] = []
    if acoustic:
        stamps = [(t, kind) for t, _d, kind in acoustic]
    else:
        for start, _end, text in segments or []:
            for m in AUDIENCE_RE.finditer(text):
                k = m.group(1).lower()
                stamps.append((start, "laughter" if k.startswith("laugh") else "applause"))
    laugh = sum(1 for _, k in stamps if k == "laughter")
    clap = sum(1 for _, k in stamps if k == "applause")
    n = len(stamps)
    return AudienceStats(laugh, clap, n - laugh - clap, n / (total_s / 60) if total_s else 0,
                         sorted(stamps)[:40])


def strip_audience(text: str) -> str:
    return re.sub(r"\s+", " ", AUDIENCE_RE.sub(" ", text)).strip()


# --------------------------------------------------------------------------- opening & closing
@dataclass
class StructureStats:
    opening: str
    closing: str
    opens_with_story: bool
    opens_with_question: bool
    opens_with_stat_or_fact: bool
    closes_with_call: bool
    closes_with_callback: bool          # closing echoes the opening (ring composition)
    first_person_singular_open: bool


CALL_CUE = [r"\blet(?:'s| us)\b", r"\bwe (?:must|need to|have to|should)\b", r"\byou (?:can|must|should|need)\b",
            r"\bimagine (?:a|if)\b", r"\bthank you\b", r"\bi (?:hope|ask|urge|challenge)\b", r"\bgo (?:out|and)\b"]


def structure_stats(text: str, open_words: int = 60, close_words: int = 60) -> StructureStats:
    ws = words_of(text)
    opening = " ".join(ws[:open_words])
    closing = " ".join(ws[-close_words:])
    lo, lc = opening.lower(), closing.lower()
    open_content = {norm(w) for w in words_of(opening) if norm(w) not in STOP and len(w) > 3}
    close_content = {norm(w) for w in words_of(closing) if norm(w) not in STOP and len(w) > 3}
    overlap = len(open_content & close_content)
    return StructureStats(
        opening=opening, closing=closing,
        opens_with_story=any(re.search(p, lo) for p in NARRATIVE_CUE),
        opens_with_question="?" in " ".join(sentences(text)[:2]),
        opens_with_stat_or_fact=bool(re.search(r"\b\d+([.,]\d+)?\s*(percent|%|million|billion|thousand|years?)\b", lo)),
        closes_with_call=any(re.search(p, lc) for p in CALL_CUE),
        closes_with_callback=overlap >= 2,
        first_person_singular_open=bool(re.match(r"\W*(so\s+)?i\b", lo)),
    )


# --------------------------------------------------------------------------- pause classification
@dataclass
class PauseUse:
    rhetorical: int          # pause after a completed sentence: emphasis, landing the point
    grammatical: int         # pause at a comma / clause boundary: phrasing
    hesitation: int          # pause mid-clause: searching for the word
    rhetorical_pct: float
    mean_rhetorical_s: float
    mean_hesitation_s: float
    longest_rhetorical: list[tuple[float, float, str]]     # (time, dur, preceding words)
    breaks: int = 0          # gaps too long to be delivery: stopped talking, edit, interruption
    break_time_s: float = 0.0


def classify_pauses(words, min_pause: float = 0.35, max_pause: float = 5.0) -> PauseUse:
    """Split inter-word gaps by where they fall in the sentence.

    This is the distinction that separates a commanding speaker from a nervous one: both may pause
    the same amount, but the confident speaker pauses *after* landing a sentence, and the anxious one
    pauses *inside* a clause while hunting for the next word."""
    rhet: list[tuple[float, float, str]] = []
    gram: list[float] = []
    hes: list[float] = []
    brk: list[float] = []
    for i, (a, b) in enumerate(zip(words, words[1:])):
        gap = b.start - a.end
        if gap < min_pause:
            continue
        if gap > max_pause:
            # Nobody holds a rhetorical pause for five seconds. This is a stop, an edit point or an
            # interruption, and counting it as delivery would flatter or damn the speaker unfairly.
            brk.append(gap)
            continue
        prev = a.word.strip()
        if prev.endswith((".", "!", "?")):
            rhet.append((a.end, gap, " ".join(w.word for w in words[max(0, i - 5): i + 1])))
        elif prev.endswith((",", ";", ":", "--", "-")):
            gram.append(gap)
        else:
            hes.append(gap)
    n = len(rhet) + len(gram) + len(hes)
    return PauseUse(
        rhetorical=len(rhet), grammatical=len(gram), hesitation=len(hes),
        rhetorical_pct=len(rhet) / n * 100 if n else 0.0,
        mean_rhetorical_s=float(np.mean([d for _, d, _ in rhet])) if rhet else 0.0,
        mean_hesitation_s=float(np.mean(hes)) if hes else 0.0,
        longest_rhetorical=sorted(rhet, key=lambda x: -x[1])[:6],
        breaks=len(brk), break_time_s=float(sum(brk)),
    )


# --------------------------------------------------------------------------- bundle
@dataclass
class RhetoricStats:
    figures: FigureStats
    language: LanguageStats
    structure: StructureStats
    audience: AudienceStats | None
    pause_use: PauseUse | None


def analyze(text: str, segments=None, total_s: float = 0.0, words=None, acoustic_audience=None) -> RhetoricStats:
    clean = strip_audience(text)
    return RhetoricStats(
        figures=figure_stats(clean),
        language=language_stats(clean),
        structure=structure_stats(clean),
        audience=audience_stats(segments, total_s, acoustic_audience) if (segments or acoustic_audience) else None,
        pause_use=classify_pauses(words) if words else None,
    )
