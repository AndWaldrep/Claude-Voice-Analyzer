"""CLI: python -m voice_analyzer <audio> [--no-transcript] [--model base.en] [-o outdir]"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import audio, features, report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Offline voice / oratory analyzer")
    p.add_argument("audio", help="any audio file ffmpeg can read (m4a, mp3, wav, ...)")
    p.add_argument("-o", "--out", default="reports", help="output directory")
    p.add_argument("--no-transcript", action="store_true", help="skip speech-to-text (acoustic metrics only)")
    p.add_argument("--model", default="base.en", help="faster-whisper model size (tiny.en, base.en, small.en, medium.en)")
    p.add_argument("--highpass", type=float, default=80.0, help="high-pass cutoff Hz (0 to disable)")
    p.add_argument("--pitch-floor", type=float, default=75.0)
    p.add_argument("--pitch-ceiling", type=float, default=400.0)
    a = p.parse_args(argv)

    t0 = time.time()
    y_raw, sr = audio.load_audio(a.audio)
    y = audio.highpass(y_raw, sr, a.highpass)
    print(f"[{time.time()-t0:5.1f}s] decoded {len(y)/sr:.1f} s @ {sr} Hz", file=sys.stderr)

    tr = None
    mask = None
    if not a.no_transcript:
        from . import transcribe

        tr = transcribe.transcribe(y, sr, a.model)
        print(f"[{time.time()-t0:5.1f}s] transcribed {len(tr.words)} words", file=sys.stderr)
        if tr.words:
            mask = features.speech_mask_from_words(tr.words, len(y) // int(sr * 0.05), 0.05)

    vol = features.volume_stats(y, sr, mask=mask)
    pau = features.pause_stats(vol)
    pit = features.pitch_stats(y, sr, a.pitch_floor, a.pitch_ceiling, vol.speech_mask)
    cla = features.clarity_stats(y, sr, vol.speech_mask)
    tail = features.trailing_off(vol, pau)
    print(f"[{time.time()-t0:5.1f}s] acoustic features done", file=sys.stderr)

    m = {"volume": vol, "pitch": pit, "pauses": pau, "clarity": cla, "trailing": tail}
    text = None
    if tr is not None:
        from . import text_metrics

        m["pace"] = text_metrics.pace_stats(tr, pau.total_time_s, pau.speech_time_s)
        m["fluency"] = text_metrics.fluency_stats(tr, pau.total_time_s)
        m["enunciation"] = text_metrics.enunciation_stats(tr)
        text = tr.text

    md, data = report.build_report(m, text, Path(a.audio).name)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.audio).stem
    (out / f"{stem}.md").write_text(md)
    report.dump_json(data, str(out / f"{stem}.json"))
    print(f"[{time.time()-t0:5.1f}s] wrote {out / (stem + '.md')} and .json", file=sys.stderr)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
