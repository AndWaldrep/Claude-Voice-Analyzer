# Claude Voice Analyzer

Offline voice / oratory analysis for speech recordings. Point it at any audio
file (m4a, mp3, wav, ogg, ...) and it produces a coaching report covering
**volume, pitch & intonation, pace & pauses, enunciation, and fluency** – no
cloud transcription service required.

## What it measures

| Area | Metrics |
|---|---|
| Recording quality | speech-to-noise ratio, noise floor (tells you which numbers to trust) |
| Volume | average level, loud-vs-quiet range, drift across the talk, clipping, trailing-off at phrase ends, per-minute levels |
| Pitch | median F0, working range in semitones, variability, monotone share, voiced share, jitter / shimmer / HNR (Praat) |
| Pace | words per minute (overall and while speaking), articulation rate (syllables/s), pace variability, fastest/slowest 30 s windows, pause count/length, long pauses with timestamps |
| Enunciation | consonant (2–8 kHz) energy, high-band balance, recognizer word confidence, least-intelligible words with timestamps, rushed long words |
| Fluency | filler words (um/uh/you know/like/so…), immediate repeats, sentence length, most-used words |
| Coaching | prioritized, rule-based suggestions derived from the numbers |

Output is a markdown report (`reports/<name>.md`) with the full transcript, plus a
JSON file with every metric for further processing.

## Install

```bash
pip install -r requirements.txt
```

`ffmpeg` is used for decoding; if it is not on your PATH the `imageio-ffmpeg`
package supplies a bundled static binary. Speech-to-text uses
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), which downloads the
model weights (`base.en`, ~75 MB) from Hugging Face on first run and caches them locally.
Everything runs on CPU; a 14-minute recording takes ~40 s.

## Use

```bash
python analyze.py recording.m4a                 # full report -> reports/recording.md + .json
python analyze.py recording.m4a --model small.en   # more accurate transcript (slower)
python analyze.py recording.m4a --no-transcript    # acoustic metrics only, no model download
python -m voice_analyzer recording.m4a -o out/     # same thing, module form
```

Options: `--highpass 80` (rumble filter cutoff, Hz), `--pitch-floor 75`,
`--pitch-ceiling 400` (use e.g. 100/500 for higher voices).

## Layout

```
voice_analyzer/
  audio.py         decode via ffmpeg -> mono 16 kHz float, high-pass filter
  features.py      volume, pitch (parselmouth/Praat), pauses, spectral clarity, phrase endings
  transcribe.py    faster-whisper word-level transcript
  text_metrics.py  pace, fillers, repetitions, enunciation proxies
  report.py        markdown + JSON report and coaching rules
  cli.py           command line
tests/             synthetic-signal unit tests (python -m pytest)
```

## Notes on interpretation

* Thresholds in the report ("130–165 wpm", "8–16 semitones") are general
  public-speaking guidelines, not clinical values.
* Jitter / shimmer / HNR are only meaningful on clean close-mic audio; the report
  flags them when the speech-to-noise ratio is below 15 dB.
* Word timings from the recognizer are approximate (~50 ms), so per-word
  enunciation metrics are indicative rather than precise.
* Nothing leaves your machine: decoding, pitch tracking and transcription are all local.
