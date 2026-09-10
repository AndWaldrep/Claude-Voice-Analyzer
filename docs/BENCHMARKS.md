# Benchmark corpus

The percentiles in `voice_analyzer/benchmarks.py` are measured, not asserted. This is where they
come from and what they do and do not support.

## The corpus

Thirteen recordings, all obtained from the Internet Archive and all either public domain or released
by TED under Creative Commons BY-NC-ND 3.0. The audio was used for measurement only; none of it is
redistributed in this repository.

| Speaker | Talk | Minutes | Source |
|---|---|---|---|
| Amy Cuddy | Your body language shapes who you are | 21 | TED, CC BY-NC-ND 3.0 |
| Angela Duckworth | Grit | 6 | TED, CC BY-NC-ND 3.0 |
| Brené Brown | The power of vulnerability | 20 | TED, CC BY-NC-ND 3.0 |
| Chimamanda Ngozi Adichie | The danger of a single story | 19 | TED, CC BY-NC-ND 3.0 |
| Derek Sivers | How to start a movement | 3 | TED, CC BY-NC-ND 3.0 |
| John F. Kennedy | Inaugural Address, 1961 | 15 | The White House, public domain |
| Ken Robinson | How to escape education's death valley | 19 | TED, CC BY-NC-ND 3.0 |
| Rita Pierson | Every kid needs a champion | 8 | TED, CC BY-NC-ND 3.0 |
| Sarah Kay | If I should have a daughter | 18 | TED, CC BY-NC-ND 3.0 |
| Seth Godin | Standing out | 17 | TED, CC BY-NC-ND 3.0 |
| Simon Sinek | How great leaders inspire action | 18 | TED, CC BY-NC-ND 3.0 |
| Susan Cain | The power of introverts | 19 | TED, CC BY-NC-ND 3.0 |
| Tony Robbins | Why we do what we do | 22 | TED, CC BY-NC-ND 3.0 |

A fourteenth recording, an FDR excerpt, was fetched but excluded: at 13 seconds it is far too short
for stable statistics.

Every file was processed identically: decoded to 16 kHz mono, high-passed at 80 Hz, transcribed with
`base.en`, and analyzed with `--speakers 1`.

## What the corpus changed

Building the corpus was worth it mainly because it falsified several thresholds that had been set
from conventional speaking advice. The corrections were not small.

| Metric | Old threshold | What the corpus shows | Correction |
|---|---|---|---|
| Words per minute | "130–165 good" | p10 140, median 153, p90 178, Tony Robbins 218 | good 130–190; the old rule flagged six of thirteen great speakers as too fast |
| Pauses per minute | ">20 is choppy" | median 15.8, p90 17.9 | choppy only above 25; frequent pausing is what good speakers do |
| Concrete-to-abstract ratio | "below 0.7 reads as a memo" | median 0.62 | flag below 0.35; the old cut condemned the corpus median |
| Narrative cues per 100 words | "below 0.8 is thin" | median 0.51, p10 0.09 | flag below 0.15 |
| Words of 3+ syllables | "above 22% is academic" | median 8%, p90 10.5% | flag above 14% |
| Articulation rate | ">5.5 syll/s is rushed" | median 5.09, p90 5.46 | rushed above 6.0 |

The general shape of the error was the same each time: conventional advice tells people to slow
down, simplify and pause less than the best speakers actually do.

## What the corpus agrees on, and what it does not

The interesting result is how *little* the corpus agrees on. Pace runs from 93 to 218 words per
minute. Pitch range runs from 9.1 to 17.2 semitones. Volume contrast runs from 3.6 to 12.1 dB.
Concreteness varies eightfold. There is no single profile of a great speaker, which is why the
report presents percentiles as questions rather than targets.

Three things do converge:

1. **Pause placement.** Every speaker in the corpus puts most pauses after a completed sentence:
   the median is 68% and the lowest TED speaker is 57%. This is the tightest agreement in the whole
   dataset and the strongest single discriminator the tool has.
2. **Sentence-length contrast.** Coefficient of variation clusters at 0.63–0.83 for all thirteen.
   Nobody writes uniform sentences.
3. **Density of figures.** Twelve of thirteen sit between 11 and 29 rhetorical figures per thousand
   words. This is a deliberate register, not an accident of speaking aloud.

The one systematic outlier is Kennedy, who is slower (93 wpm), more monotone (36%) and less
inclined to place pauses at full stops (36%) than any TED speaker. That is a real difference between
1961 platform oratory and modern conversational stage delivery, not a defect, and it is a useful
reminder that the corpus encodes a period style as well as a standard of quality.

## Limitations

- Thirteen speeches is a small sample and the confidence intervals on any percentile are wide.
- Twelve of thirteen are TED talks, so the corpus is biased toward one format: eighteen minutes,
  single speaker, live audience, heavily rehearsed.
- Transcription used `base.en`, which makes word timings approximate and occasionally garbles proper
  nouns, so text-derived metrics carry a few percent of noise.
- Audience-reaction detection is deliberately conservative and undercounts; treat the counts as a
  floor.
- All of these are recordings of *successful* talks. The corpus therefore describes what good
  speakers do, not what distinguishes good from bad, since it contains no bad examples.

## Reproducing

The fetch and analysis scripts are not committed, because they download several hundred megabytes of
third-party media. To rebuild: fetch each identifier from the Internet Archive, extract mono 16 kHz
audio, run `python analyze.py <file> --speakers 1 --model base.en`, and take percentiles across the
resulting JSON reports.
