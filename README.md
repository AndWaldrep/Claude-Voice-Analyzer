# Claude Voice Analyzer

Offline speech analysis for coaches: **who spoke, what they said, and how they said it**.
Point it at any recording (m4a, mp3, wav, ogg, …) of one person or a whole meeting and it
produces a coaching report per speaker covering volume, pitch & intonation, pace & pauses,
enunciation, fluency, and emotional undertone – with a speaker-labelled transcript.
Everything runs locally on CPU; no audio leaves the machine.

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
| Coaching | prioritised, rule-based suggestions per speaker derived from the numbers |

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
6. **Report** with thresholds from public-speaking practice and coaching rules.

Validated on a 3-reader LibriSpeech mix (all 11 turns and 3 speakers recovered exactly) and a
14-minute single-speaker phone recording (correctly reported as one speaker).

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
