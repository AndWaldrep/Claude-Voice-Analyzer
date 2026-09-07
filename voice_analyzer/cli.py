"""CLI: python -m voice_analyzer <audio> [options]"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import audio, features, report

FRAME_S = 0.05


def speaker_metrics(y, sr, mask, words, total_s, pitch_floor, pitch_ceiling, spans=None):
    from . import text_metrics
    from .transcribe import Transcript

    vol = features.volume_stats(y, sr, mask=mask)
    pau = features.pause_stats(vol, within=spans)
    pit = features.pitch_stats(y, sr, pitch_floor, pitch_ceiling, vol.speech_mask)
    cla = features.clarity_stats(y, sr, vol.speech_mask)
    tail = features.trailing_off(vol, pau)
    m = {"volume": vol, "pitch": pit, "pauses": pau, "clarity": cla, "trailing": tail}
    if words is not None:
        tr = Transcript(words=words, segments=[(0, total_s, " ".join(w.word for w in words))], language="en")
        m["pace"] = text_metrics.pace_stats(tr, pau.total_time_s, pau.speech_time_s)
        m["fluency"] = text_metrics.fluency_stats(tr, pau.total_time_s)
        m["enunciation"] = text_metrics.enunciation_stats(tr)
    return m


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Offline voice / oratory analyzer with speaker diarization and tone indicators")
    p.add_argument("audio", help="any audio file ffmpeg can read (m4a, mp3, wav, ...)")
    p.add_argument("-o", "--out", default="reports", help="output directory")
    p.add_argument("--no-transcript", action="store_true", help="skip speech-to-text (acoustic metrics only, single speaker)")
    p.add_argument("--model", default="small.en", help="faster-whisper model (tiny.en, base.en, small.en, medium.en, large-v3)")
    p.add_argument("--speakers", default="auto", help="number of speakers, or 'auto' (default), or 1 to skip diarization")
    p.add_argument("--diarize-threshold", type=float, default=0.12, help="single-speaker cut for auto speaker count (higher = more willing to call it one voice)")
    p.add_argument("--emotion-model", action="store_true", help="also run the experimental categorical emotion model per turn (slower)")
    p.add_argument("--highpass", type=float, default=80.0, help="high-pass cutoff Hz (0 to disable)")
    p.add_argument("--pitch-floor", type=float, default=75.0)
    p.add_argument("--pitch-ceiling", type=float, default=400.0)
    p.add_argument("--no-cache", action="store_true", help="ignore cached transcript / embeddings in <out>/.cache")
    a = p.parse_args(argv)

    t0 = time.time()
    log = lambda msg: print(f"[{time.time()-t0:6.1f}s] {msg}", file=sys.stderr)
    y_raw, sr = audio.load_audio(a.audio)
    y = audio.highpass(y_raw, sr, a.highpass)
    total_s = len(y) / sr
    n_frames = len(y) // int(sr * FRAME_S)
    log(f"decoded {total_s:.1f} s @ {sr} Hz")

    out = Path(a.out)
    cache = out / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    key = f"{Path(a.audio).stem}-{hashlib.sha1(y_raw[::997].tobytes()).hexdigest()[:10]}"
    tr = None
    if not a.no_transcript:
        from . import transcribe

        tcache = cache / f"{key}-{a.model}.json"
        if tcache.exists() and not a.no_cache:
            d = json.loads(tcache.read_text())
            tr = transcribe.Transcript([transcribe.Word(**w) for w in d["words"]], [tuple(s) for s in d["segments"]], d["language"])
            log(f"loaded cached transcript ({len(tr.words)} words, {a.model})")
        else:
            tr = transcribe.transcribe(y, sr, a.model)
            tcache.write_text(json.dumps({"words": [w.__dict__ for w in tr.words], "segments": tr.segments, "language": tr.language}))
            log(f"transcribed {len(tr.words)} words with {a.model}")

    n_spk = None if a.speakers == "auto" else int(a.speakers)
    dia = None
    if tr is not None and tr.words and n_spk != 1:
        from . import diarize

        vad = features.volume_stats(y, sr).speech_mask
        ecache = cache / f"{key}-{a.model}-emb.npy"
        embs = np.load(ecache) if ecache.exists() and not a.no_cache else None
        dia = diarize.diarize(y, sr, tr, n_spk, threshold=a.diarize_threshold, speech_mask=vad, embeddings=embs)
        if embs is None and getattr(dia, "embeddings", None) is not None:
            np.save(ecache, dia.embeddings)
        log(f"diarized: {len(dia.speakers)} speaker(s) (eigengap estimate {dia.k_estimated}, eigenvalues "
            f"{', '.join(f'{v:.3f}' for v in dia.eigenvalues[:5])}), {len(dia.turns)} turns from {dia.n_windows} windows")

    speakers = []
    interaction = None
    if dia is not None and len(dia.speakers) >= 1:
        from . import tone

        emo = tone.load_categorical() if a.emotion_model else None
        interaction = diarize.interaction_stats(dia, total_s, speech_mask=vad, frame_s=FRAME_S)
        for spk in dia.speakers:
            mask = diarize.speaker_mask(dia, spk, n_frames, FRAME_S)
            spans = diarize.speaker_spans(dia, spk)
            m = speaker_metrics(y, sr, mask, dia.words_for(spk), total_s, a.pitch_floor, a.pitch_ceiling, spans)
            st = tone.analyze_speaker_tone(spk, dia.turns_for(spk), m["volume"], m["pitch"], y=y, sr=sr, categorical_model=emo)
            speakers.append({"name": spk, "metrics": m, "tone": st})
        log("per-speaker metrics + tone done")
    else:
        mask = features.speech_mask_from_words(tr.words, n_frames, FRAME_S) if tr and tr.words else None
        m = speaker_metrics(y, sr, mask, tr.words if tr else None, total_s, a.pitch_floor, a.pitch_ceiling)
        speakers.append({"name": "Speaker 1", "metrics": m, "tone": None})
        log("acoustic features done")

    # overall SNR from a whole-file pass
    whole = features.volume_stats(y, sr, mask=features.speech_mask_from_words(tr.words, n_frames, FRAME_S) if tr and tr.words else None)
    md, data = report.build_report(Path(a.audio).name, total_s, whole.snr_db, whole.noise_floor_db, speakers, interaction,
                                   dia.turns if dia else None, tr.text if tr else None)
    stem = Path(a.audio).stem
    (out / f"{stem}.md").write_text(md)
    report.dump_json(data, str(out / f"{stem}.json"))
    log(f"wrote {out / (stem + '.md')} and .json")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
