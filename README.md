# Claude Voice Analyzer

Offline speech analysis for coaches: **who spoke, what they said, and how they said it**.
Point it at any recording (m4a, mp3, wav, ogg, …) of one person or a whole meeting and it produces a
coaching report per speaker covering volume, pitch, pace, pauses, enunciation, fluency, emotional
undertone, and rhetorical composition – with a speaker-labelled transcript, and every number
benchmarked against a measured corpus of recognized speakers.
Everything runs locally on CPU; no audio leaves the machine.

The thresholds are not invented. They come from running this same pipeline over thirteen speeches by
Brené Brown, Simon Sinek, Chimamanda Ngozi Adichie, Ken Robinson, Kennedy and others; see
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). The reasoning behind what gets measured, from Aristotle
and Cicero through to modern practice, is in [docs/ORATORY.md](docs/ORATORY.md).

## What it measures

| Area | Metrics |
|---|---|
| Recording quality | speech-to-noise ratio, noise floor (tells you which numbers to trust) |
| Speakers | automatic speaker count and labels, talk-time share, turns, mean/longest turn, cut-ins |
| Volume | average level, peak contrast (loud vs. quiet), drift across the talk, clipping, trailing-off at phrase ends, per-minute levels |
| Pitch | median F0, working range in semitones, variability, monotone share, voiced share, jitter / shimmer / HNR (Praat) |
| Pace | words per minute (overall and while speaking), articulation rate, pace variability, fastest/slowest 30 s, pause count/length, long pauses with timestamps |
| Enunciation | consonant (2–8 kHz) energy, high-band balance, recognizer word confidence, least-intelligible words with timestamps, rushed long words |
| Fluency | filler words (um / uh / you know / like / so …), immediate repeats, sentence length, most-used words |
| Tone | per-turn arousal (energy + pitch + rate vs. the speaker's own baseline), expressiveness, rising/falling final contour (tentative vs. assertive), hesitancy; notable moments with what was said; optional categorical model (neutral / angry / happy / sad, experimental) |
| Pause *use* | every pause classified as rhetorical (after a full stop), grammatical (at a comma) or hesitation (mid-clause), plus long breaks. The rhetorical share is the single most discriminating metric in the tool: every great speaker measured sits above 57%, median 68% |
| Rhetoric | anaphora, epistrophe, tricolon, antithesis, chiasmus, rhetorical questions, with the actual quoted text; figure density per 1000 words |
| Language | concrete-and-sensory vs. abstract vocabulary, Anglo-Saxon vs. Latinate register, sentence-length contrast, short punch sentences, direct address (you / we / I), narrative cues, metaphor cues, the repeated throughline phrase, signposting |
| Structure | whether the talk opens with a story, a question or a number; whether it closes with a call to action; whether the ending echoes the opening (ring composition) |
| Audience | laughter and applause detected acoustically (Whisper's VAD strips them from the transcript), with timestamps – the only direct evidence that the room reacted |
| Benchmark | every headline metric as a percentile against the corpus of recognized speakers |
| Coaching | prioritised, rule-based suggestions per speaker, covering both delivery and composition |

Output: `reports/<name>.md` (report + speaker-labelled transcript) and `reports/<name>.json` (every metric).

## How it works

1. **Decode** with ffmpeg → mono 16 kHz, 80 Hz high-pass to remove rumble / handling noise.
2. **Transcribe** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`small.en` by default) with word timestamps.
3. **Diarize**: 3 s windows over detected speech → [ECAPA-TDNN speaker embeddings](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
   (SpeechBrain) → spectral clustering; the number of speakers is chosen by the eigengap of the
   affinity Laplacian (or pass `--speakers N`). Words are assigned to speakers by window overlap and
   merged into turns. Speaker talk time comes from the acoustic windows, so a voice that the
   recognizer could not transcribe still appears.
4. **Per-speaker acoustics** with Praat (via parselmouth) and numpy/scipy on that speaker's frames only.
5. **Tone** from prosody relative to each speaker's own baseline (z-scores), so a normally quiet
   person getting louder reads as animated even if still quieter than the room.
6. **Rhetoric** from the transcript: figures, language texture and structure, plus pause use
   classified by where each pause falls in the sentence.
7. **Audience reactions** detected acoustically — energetic stretches carrying no transcribed words
   that are broader-band and brighter than the speaker's voice.
8. **Report** with thresholds measured from the benchmark corpus, not from conventional advice.

Validated on a 3-reader LibriSpeech mix (all 11 turns and 3 speakers recovered exactly), a
14-minute single-speaker phone recording (correctly reported as one speaker), and the benchmark
corpus, where audience detection independently recovered Ken Robinson's actual laugh lines.

## The practice app

`docs/` is a self-contained web app that records you through the device microphone and scores the
take live. It is plain HTML and ES modules with no build step and no dependencies, so GitHub Pages
can serve it as-is, and an iPhone can install it to the home screen where it behaves like an app.

Pick a form of speaking, read its model text aloud, and the page measures eight scored qualities
against what that form actually calls for: pace, articulation rate, pitch range, volume contrast,
pauses per minute, mean pause, monotone stretches, and where in the sentence the pauses fall. A
eulogy and a battlefield exhortation want opposite things, so each form carries its own target
bands and switching form rescores the same take.

**Where the pauses fall** is the measurement worth explaining, because it needs no transcription
and it is the most discriminating thing in the whole project. The passage is known, so its
syllables are known, and its sentence and comma positions are known. The syllable nuclei a reader
produces are detected acoustically. Laying one against the other says how far through the text
someone had got when they stopped: a reader who has produced 120 of 240 nuclei is halfway through,
and the sentence ending at syllable 118 is the one they just finished. Every pause is then placed
as after a full stop, at a comma, or mid-clause. Across the thirteen recorded speeches in the
benchmark corpus, at least 57% of pauses landed after a full stop and the median was 68%. The same
silence reads as command after a full stop and as hunting for a word inside a clause.

Beyond the score there is a deeper reading: harmonics-to-noise ratio (Boersma's formulation, from
the autocorrelation peak), pitch and loudness steadiness, consonant energy in the 2 to 8 kHz band,
rhythm variability, loudness drift across the reading, whether phrases trail off at their ends,
whether final pitch rises or falls, and a sentence-by-sentence table of rate and level. Every
headline figure is also placed as a percentile against the benchmark corpus.

**Turning it on:** repository Settings, then Pages, then Source "Deploy from a branch", branch
`main` (or this feature branch), folder `/docs`. It publishes to
`https://<user>.github.io/Claude-Voice-Analyzer/`. The microphone needs HTTPS, which Pages provides.
On an iPhone, open that URL in Safari, then Share and "Add to Home Screen".

**Why a web app rather than a native one.** The microphone is the whole point, and a browser reaches
it on both desktop and iOS. A native iOS build would need a Mac, Xcode, a developer account and a
device to test on, none of which exist in the environment this was written in, so it could have been
written but never run. The web version is tested here against a simulated microphone on every change.

**What the recorder does that matters.** The browser's automatic gain control, noise suppression and
echo cancellation are all switched off. Left at their defaults they would corrupt three of the seven
measurements: gain control flattens loud against quiet, which is exactly the volume contrast being
measured, and noise suppression strips the low end, which removes the fundamental of a low voice and
makes pitch read an octave high. Audio is tapped as raw samples off the audio graph rather than
encoded, so the analysis sees what the microphone produced and playback is built from the same array.

Practice history is kept in browser storage on the device. Nothing is uploaded anywhere.

```
docs/
  index.html    page, styles, and the phone record dock
  app.js        UI, scoring, charts, history
  dsp.js        the measurement code, shared with the test suite
  modes.js      the six forms, their model texts, target bands and script positions
  benchmarks.js measured percentiles from the corpus of recorded speeches
  capture.js    microphone capture to raw samples
  worklet.js    the audio worklet that taps the graph
  sw.js         offline cache, so practice works without a signal
web-tests/
  dsp.test.mjs   109 checks against signals with known answers
  live.test.mjs   35 checks driving the real microphone path in Chromium
```

Run them with `node web-tests/dsp.test.mjs` and `node web-tests/live.test.mjs`. The live test feeds
Chromium a synthetic voice of known pitch as its microphone, so the whole chain is exercised:
permission, worklet capture, sample buffer, analysis and the rendered score.

Pause placement is tested by building audio that reads the real script and pauses only at its full
stops, which must score 100%, and again pausing only inside sentences, which must score near zero.

(`docs/package.json` exists only so Node treats the `.js` files there as ES modules when the tests
import them. It is not a package and has no dependencies.)

## Install

```bash
pip install -r requirements.txt
```

`ffmpeg` on PATH is used for decoding; otherwise the bundled `imageio-ffmpeg` binary is used.
Models are downloaded from Hugging Face on first use and cached (`~/.cache`):

| Model | Size | License | Used for |
|---|---|---|---|
| `Systran/faster-whisper-small.en` (or the `--model` you pick) | 250 MB | MIT | transcript + word timings |
| `speechbrain/spkrec-ecapa-voxceleb` | 80 MB | Apache 2.0 | speaker embeddings |
| `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` + `facebook/wav2vec2-base` | 380 MB | Apache 2.0 | only with `--emotion-model` |

None of these are gated; no Hugging Face account or token is needed.

## Use

```bash
python analyze.py meeting.m4a                    # auto speaker count, small.en transcript
python analyze.py talk.m4a --speakers 1          # single speaker, skip diarization
python analyze.py panel.mp3 --speakers 4         # force a known speaker count
python analyze.py call.wav --model base.en       # faster, slightly rougher transcript
python analyze.py call.wav --emotion-model       # add the categorical emotion model per turn
python analyze.py call.wav --no-transcript       # acoustic metrics only, no model downloads
```

Options: `-o DIR` output directory · `--diarize-threshold 0.12` (raise to be more willing to call
a recording single-speaker) · `--highpass 80` · `--pitch-floor 75` / `--pitch-ceiling 400`
(use ~100 / 500 for higher voices).

Timing on a 4-core CPU for a 14-minute file: ~2.5 min transcription (`small.en`; `base.en` is
~30 s), ~1 min diarization, a few seconds for everything else.

## Layout

```
voice_analyzer/
  audio.py         decode via ffmpeg -> mono 16 kHz float, high-pass filter
  transcribe.py    faster-whisper word-level transcript
  diarize.py       ECAPA embeddings, eigengap spectral clustering, word->speaker alignment, interaction stats
  features.py      volume, pitch (Praat), pauses, spectral clarity, phrase endings
  text_metrics.py  pace, fillers, repetitions, enunciation proxies
  tone.py          prosodic emotional-undertone indicators (+ optional categorical model)
  rhetoric.py      figures, language texture, structure, pause classification, audience response
  benchmarks.py    measured percentiles from the corpus of recognized speakers
  report.py        markdown + JSON report and coaching rules
  cli.py           command line
tests/             unit tests on synthetic audio / embeddings (python -m pytest), no downloads
```

## Notes on interpretation

* Thresholds ("130–165 wpm", "8–16 semitones") are general public-speaking guidelines, not clinical values.
* Jitter / shimmer / HNR are only meaningful on clean close-mic audio; the report flags them when the
  speech-to-noise ratio is below 15 dB.
* Recognizer word timings are ~50–100 ms coarse and turn boundaries can be off by a second, so per-word
  enunciation metrics and cut-in counts are indicative. Cut-ins are measured from acoustic silence, not word times.
* Tone indicators are prosodic, relative to the speaker's own recording, and describe *delivery*
  (energised, flat, tentative, assertive, hesitant) – not inner emotional state. The optional categorical
  model was trained on acted speech and is poorly calibrated on real conversation; treat it as a hint.
* Diarization is embedding-based without overlap handling: two people talking at once get one label.
* Rhetorical figures are found by pattern matching, not by understanding. Counts are dependable;
  a metaphor that uses no cue word will be missed, and the metaphor metric in particular is weak.
* The benchmark corpus is thirteen speeches, twelve of them TED talks, and contains only successful
  ones. It describes what good speakers do; it cannot tell you what separates good from bad.
