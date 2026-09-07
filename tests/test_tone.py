import numpy as np

from voice_analyzer import features, tone
from voice_analyzer.diarize import Turn
from voice_analyzer.transcribe import Word


def test_tone_flags_energised_turn():
    sr, frame_s = 16000, 0.05
    # 6 turns of 3 s buzz; the 4th is 10 dB louder and higher pitched
    parts, turns, t = [], [], 0.0
    for i in range(6):
        f = 170 if i == 3 else 120
        amp = 0.6 if i == 3 else 0.2
        tt = np.arange(3 * sr) / sr
        parts.append(amp * np.sign(np.sin(2 * np.pi * f * tt)))
        words = [Word(w, t + j * 0.5, t + j * 0.5 + 0.4, 0.9) for j, w in enumerate("one two three four five".split())]
        turns.append(Turn("Speaker 1", t, t + 3.0, words))
        parts.append(np.zeros(sr // 2)); t += 3.5
    y = np.concatenate(parts).astype(np.float32)
    y += 1e-3 * np.random.default_rng(0).standard_normal(len(y)).astype(np.float32)   # -60 dB room noise
    vol = features.volume_stats(y, sr)
    pit = features.pitch_stats(y, sr, speech_mask=vol.speech_mask)
    st = tone.analyze_speaker_tone("Speaker 1", turns, vol, pit)
    assert len(st.turns) == 6
    loud = st.turns[3]
    assert loud.arousal > 0.8 and "energised" in loud.tags and "louder than usual" in loud.tags
    assert st.notable[0].start == loud.start or any(n.start == loud.start for n in st.notable)
