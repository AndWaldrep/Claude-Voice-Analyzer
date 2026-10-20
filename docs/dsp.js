/* Speech measurement, shared by the browser app and the Node test suite.
   No DOM, no Web Audio: plain arrays in, numbers out, so the shipped code is the tested code.

   Validated against Praat and against a Python reimplementation of the same algorithm on real
   recordings: speech detection, pause counts and volume contrast match exactly, pitch within 3%.
   See web-tests/ for the signals with known answers that pin each stage. */

export const HOP = 0.02;          // 20 ms analysis frame
export const PITCH_HOP = 0.025;
export const MIN_PAUSE = 0.30;
export const MAX_PAUSE = 5.0;     // longer than this is a stop, not delivery

/* ------------------------------------------------------------------ helpers */
export function pct(sorted, p) {
  if (!sorted.length) return NaN;
  const i = (sorted.length - 1) * p / 100;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}
const yieldToUI = () => new Promise(r => setTimeout(r, 0));

/* One biquad section, RBJ cookbook coefficients. Matches BiquadFilterNode, so the same
   filtering can run off the audio thread and inside tests. */
export function biquad(x, sr, type, f0, Q = 0.707) {
  const w = 2 * Math.PI * f0 / sr, cw = Math.cos(w), sw = Math.sin(w), al = sw / (2 * Q);
  let b0, b1, b2;
  const a0 = 1 + al, a1 = -2 * cw, a2 = 1 - al;
  if (type === "hp") { b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2; }
  else { b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2; }
  const B0 = b0 / a0, B1 = b1 / a0, B2 = b2 / a0, A1 = a1 / a0, A2 = a2 / a0;
  const y = new Float32Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = B0 * x[i] + B1 * x1 + B2 * x2 - A1 * y1 - A2 * y2;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
  }
  return y;
}

/** 80 Hz high-pass: removes rumble, handling noise and desk thump before anything is measured. */
export const highpass = (x, sr, f0 = 80) => biquad(x, sr, "hp", f0, 0.707);

/* ------------------------------------------------------------------ levels */
export function frameLevels(x, sr) {
  const hop = Math.round(sr * HOP), n = Math.floor(x.length / hop);
  const db = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    let s = 0;
    const a = f * hop, b = a + hop;
    for (let i = a; i < b; i++) s += x[i] * x[i];
    db[f] = 20 * Math.log10(Math.sqrt(s / hop) + 1e-10);
  }
  return db;
}

/** Which frames carry speech, plus the noise floor it was judged against.
 *
 *  Only true digital silence is excluded from the floor estimate. Dropping everything under
 *  -80 dBFS (an earlier version did) threw away the background of a quiet recording and left only
 *  speech in the estimate, so the floor landed above the speech and the clip read as silent. */
export function speechMask(db) {
  const live = Array.from(db).filter(v => v > -150).sort((a, b) => a - b);
  const floor = live.length > 10 ? pct(live, 5) : -90;
  const peak = live.length > 10 ? pct(live, 95) : 0;
  // With real silence present, sit 8 dB over the floor. With none -- a continuously voiced take,
  // or one trimmed hard to the speech -- accept anything within 25 dB of the loud level instead
  // of rejecting the whole recording.
  const thresh = (peak - floor >= 12) ? floor + 8 : peak - 25;
  const raw = Array.from(db, v => v > thresh);
  const m = new Array(raw.length);
  for (let i = 0; i < raw.length; i++) {          // 100 ms majority vote, kills single-frame flicker
    let c = 0, t = 0;
    for (let j = Math.max(0, i - 2); j <= Math.min(raw.length - 1, i + 2); j++) { t++; if (raw[j]) c++; }
    m[i] = c / t >= 0.5;
  }
  return { mask: m, floor, thresh };
}

/** Loud-versus-quiet contrast: p90 minus p10 of half-second peak levels during speech.
 *  Peaks rather than frames, so quiet consonants and micro-gaps inside a phrase do not
 *  masquerade as dynamic range. */
export function dynamicRange(db, mask) {
  const per = Math.round(0.5 / HOP);
  const peaks = [];
  for (let i = 0; i + per <= db.length; i += per) {
    let voiced = 0, mx = -Infinity;
    for (let j = i; j < i + per; j++) { if (mask[j]) voiced++; if (db[j] > mx) mx = db[j]; }
    if (voiced / per >= 0.5) peaks.push(mx);
  }
  const src = (peaks.length >= 4 ? peaks : Array.from(db).filter((_, i) => mask[i])).slice().sort((a, b) => a - b);
  return pct(src, 90) - pct(src, 10);
}

/* ------------------------------------------------------------------ pauses */
export function pauses(mask) {
  const out = [];
  let run = 0;
  for (let i = 0; i <= mask.length; i++) {
    if (i < mask.length && !mask[i]) { run++; continue; }
    if (run * HOP >= MIN_PAUSE) out.push({ start: (i - run) * HOP, dur: run * HOP });
    run = 0;
  }
  const total = mask.length * HOP;
  return out.filter(p => p.start > 0.4 && p.start + p.dur < total - 0.4 && p.dur <= MAX_PAUSE);
}

/* ------------------------------------------------------------------ syllables */
/** Count syllable nuclei: maxima of the loudness envelope that stand clear of their valleys.
 *  The window is 5 frames (100 ms). At 3 frames, ripple inside one held vowel counted as two
 *  syllables and slow speech over-counted by as much as 76%. */
export function syllablePeaks(db, mask, floor) {
  const s = new Float32Array(db.length);
  for (let i = 0; i < db.length; i++) {
    let t = 0, c = 0;
    for (let j = Math.max(0, i - 2); j <= Math.min(db.length - 1, i + 2); j++) { t += db[j]; c++; }
    s[i] = t / c;
  }
  const minGap = Math.round(0.1 / HOP), vw = Math.round(0.3 / HOP);
  let last = -1e9, n = 0;
  const at = [];
  for (let i = 1; i < s.length - 1; i++) {
    if (!mask[i] || s[i] < floor + 10) continue;
    if (!(s[i] >= s[i - 1] && s[i] > s[i + 1])) continue;
    let lv = s[i], rv = s[i];
    for (let j = i; j > Math.max(0, i - vw); j--) lv = Math.min(lv, s[j]);
    for (let j = i; j < Math.min(s.length, i + vw); j++) rv = Math.min(rv, s[j]);
    if (s[i] - Math.max(lv, rv) < 3) continue;
    if (i - last < minGap) continue;
    last = i; n++; at.push(i * HOP);
  }
  return { count: n, at };
}

/* ------------------------------------------------------------------ pitch */
export function decimate(x, sr, target) {
  const k = Math.max(1, Math.round(sr / target));
  const n = Math.floor(x.length / k);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < k; j++) s += x[i * k + j];
    y[i] = s / k;
  }
  return { y, sr: sr / k };
}

/** Fundamental frequency per frame by normalised autocorrelation on an 8 kHz copy. */
export async function pitchTrack(x, sr, mask, onProgress = () => {}) {
  const { y, sr: fs } = decimate(x, sr, 8000);
  const N = Math.round(fs * 0.064);                       // 64 ms window
  const minLag = Math.floor(fs / 400), maxLag = Math.floor(fs / 70);
  const step = Math.round(fs * PITCH_HOP);
  const f0 = [], times = [], clarity = [];
  const w = new Float32Array(N), corr = new Float32Array(maxLag + 2);
  for (let f = 0, c = 0; f * step + N < y.length; f++, c++) {
    const t = f * step / fs;
    if (c % 150 === 0) { onProgress(f * step / y.length); await yieldToUI(); }
    if (!mask[Math.min(mask.length - 1, Math.floor(t / HOP))]) continue;
    const off = f * step;
    let mean = 0;
    for (let i = 0; i < N; i++) mean += y[off + i];
    mean /= N;
    let e = 0;
    for (let i = 0; i < N; i++) { w[i] = y[off + i] - mean; e += w[i] * w[i]; }
    if (e < 1e-7) continue;
    corr.fill(0);
    let best = 0, bestLag = -1;
    for (let lag = minLag; lag <= maxLag; lag++) {
      let num = 0, d1 = 0, d2 = 0;
      for (let i = 0; i < N - lag; i++) { num += w[i] * w[i + lag]; d1 += w[i] * w[i]; d2 += w[i + lag] * w[i + lag]; }
      const r = num / (Math.sqrt(d1 * d2) + 1e-12);
      corr[lag] = r;
      if (r > best) { best = r; bestLag = lag; }
    }
    if (bestLag < 0 || best < 0.45) continue;
    // Autocorrelation peaks at the period AND at every multiple of it, and the normalisation can
    // let a multiple win by a hair. Untreated, a 220 Hz voice reads as 73. So when a sub-multiple
    // of the winning lag scores nearly as well, prefer the shorter one.
    for (const divisor of [2, 3]) {
      const cand = Math.round(bestLag / divisor);
      if (cand >= minLag && corr[cand] > best * 0.85) { bestLag = cand; break; }
    }
    let lagF = bestLag;
    if (bestLag > minLag && bestLag < maxLag) {        // parabolic refinement
      const a = corr[bestLag - 1], b = corr[bestLag], cc = corr[bestLag + 1], den = a - 2 * b + cc;
      if (den !== 0) lagF = bestLag + 0.5 * (a - cc) / den;
    }
    const hz = fs / lagF;
    if (hz >= 70 && hz <= 400) { f0.push(hz); times.push(t); clarity.push(Math.min(0.9995, best)); }
  }
  return { f0, times, clarity };
}

const semis = (hz, ref) => 12 * Math.log2(hz / ref);

export function pitchStats(f0, times) {
  if (f0.length < 20) return { range: NaN, median: NaN, mono: NaN, sd: NaN };
  const sorted = f0.slice().sort((a, b) => a - b);
  const median = pct(sorted, 50);
  const p5 = pct(sorted, 5), p95 = pct(sorted, 95);
  const all = f0.map(v => semis(v, median));
  const mu = all.reduce((a, b) => a + b, 0) / all.length;
  const sd = Math.sqrt(all.reduce((a, b) => a + (b - mu) ** 2, 0) / all.length);
  let tot = 0, flat = 0;
  const span = 3;
  for (let t = times[0]; t < times[times.length - 1] - span; t += span) {
    const seg = [];
    for (let i = 0; i < times.length; i++) if (times[i] >= t && times[i] < t + span) seg.push(f0[i]);
    if (seg.length < 25) continue;
    const st = seg.map(v => semis(v, median));
    const m2 = st.reduce((a, b) => a + b, 0) / st.length;
    const s2 = Math.sqrt(st.reduce((a, b) => a + (b - m2) ** 2, 0) / st.length);
    tot++; if (s2 < 2) flat++;
  }
  return { range: 12 * Math.log2(p95 / p5), median, sd, mono: tot ? flat / tot * 100 : NaN };
}

/* ------------------------------------------------------------------ recording quality */
/** RMS of x inside [lo, hi] Hz over speech frames only. */
export function bandRms(x, sr, lo, hi, mask) {
  const b = biquad(biquad(x, sr, "hp", lo), sr, "lp", hi);
  const hop = Math.round(sr * HOP);
  let s = 0, n = 0;
  for (let f = 0; f < mask.length && (f + 1) * hop <= b.length; f++) {
    if (!mask[f]) continue;
    for (let i = f * hop; i < (f + 1) * hop; i++) { s += b[i] * b[i]; n++; }
  }
  return n ? Math.sqrt(s / n) : 0;
}

/** How much weaker the low end is than the formant region, in dB. Far below zero means the
 *  recording has had its bass removed: an old broadcast, a phone speaker, or noise suppression
 *  left switched on. A low voice recorded that way has no fundamental left for any time-domain
 *  tracker to find and pitch then reads an octave high. Calibrated on real recordings: the 1961
 *  Kennedy inaugural measures -16.6 dB, full-bandwidth speech runs -7 to +5 dB. */
export function lowEndBalance(x, sr, mask) {
  const low = bandRms(x, sr, 60, 150, mask);
  const ref = bandRms(x, sr, 300, 600, mask);
  return 20 * Math.log10((low + 1e-12) / (ref + 1e-12));
}


/* ------------------------------------------------------------------ voice quality */

/** Harmonics-to-noise ratio in dB, from the autocorrelation peak at the pitch period.
 *  This is Boersma's formulation: a peak of r means a fraction r of the frame's energy is
 *  periodic, so the harmonic-to-noise ratio is r/(1-r). Above about 15 dB is a clean, steady
 *  voice; below 10 dB suggests breathiness, hoarseness or simply a noisy room. */
export function hnr(clarity) {
  if (!clarity || clarity.length < 20) return NaN;
  const vals = clarity.map(r => 10 * Math.log10(r / Math.max(1e-6, 1 - r)));
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

/** Cycle-to-cycle instability as a percentage: the mean absolute difference between consecutive
 *  readings over their mean. Only genuinely consecutive frames count, so a gap between phrases is
 *  not read as a sudden jump. Applied to pitch this is a pitch-steadiness measure, and to loudness
 *  an amplitude-steadiness one. Both are frame-level, so they are cousins of clinical jitter and
 *  shimmer rather than the same measurement, and the report names them accordingly. */
export function perturbation(values, times, step) {
  if (!values || values.length < 20) return NaN;
  let sum = 0, n = 0;
  for (let i = 0; i + 1 < values.length; i++) {
    if (times && Math.abs(times[i + 1] - times[i] - step) > step * 0.6) continue;
    const mean = (values[i] + values[i + 1]) / 2;
    if (mean <= 0) continue;
    sum += Math.abs(values[i + 1] - values[i]) / mean;
    n++;
  }
  return n ? sum / n * 100 : NaN;
}

/* ------------------------------------------------------------------ phrases */

/** What is said between one pause and the next.
 *
 *  Phrases are carved out by the pauses, not by every gap in the speech mask: the closure in the
 *  middle of a word like "better" is 80 ms of near-silence, and treating each of those as a phrase
 *  boundary would turn one sentence into a dozen fragments. */
export function phrases(mask, pauseList, minDur = 0.3) {
  const first = mask.indexOf(true), lastI = mask.lastIndexOf(true);
  if (first < 0) return [];
  const end = (lastI + 1) * HOP;
  const out = [];
  let t = first * HOP;
  for (const p of (pauseList || []).slice().sort((a, b) => a.start - b.start)) {
    if (p.start > t) out.push({ start: t, end: p.start, dur: p.start - t });
    t = Math.max(t, p.start + p.dur);
  }
  if (end > t) out.push({ start: t, end, dur: end - t });
  return out.filter(p => p.dur >= minDur);
}

/** What happens at the end of each phrase.
 *
 *  Two habits show up here and both are coachable. Trailing off is the last few words falling away
 *  in loudness, which buries the word that usually matters most. A rising pitch on a statement
 *  reads as asking permission; a falling one reads as having decided. */
export function phraseEndings(db, mask, ph, f0, times, median, tailS = 0.6, bodyS = 1.8) {
  const tail = Math.round(tailS / HOP), body = Math.round(bodyS / HOP);
  const drops = [], slopes = [];
  for (const p of ph) {
    const e = Math.round(p.end / HOP);
    const t0 = Math.max(0, e - tail), b0 = Math.max(0, e - tail - body);
    const tv = [], bv = [];
    for (let i = t0; i < e; i++) if (mask[i]) tv.push(db[i]);
    for (let i = b0; i < t0; i++) if (mask[i]) bv.push(db[i]);
    if (tv.length >= 3 && bv.length >= 6) {
      drops.push(bv.reduce((a, b) => a + b, 0) / bv.length - tv.reduce((a, b) => a + b, 0) / tv.length);
    }
    // pitch slope across the final half second, in semitones per second
    const xs = [], ys = [];
    for (let i = 0; i < times.length; i++) {
      if (times[i] >= p.end - 0.5 && times[i] <= p.end) { xs.push(times[i]); ys.push(12 * Math.log2(f0[i] / median)); }
    }
    if (xs.length >= 5) {
      const mx = xs.reduce((a, b) => a + b, 0) / xs.length, my = ys.reduce((a, b) => a + b, 0) / ys.length;
      let num = 0, den = 0;
      for (let i = 0; i < xs.length; i++) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) ** 2; }
      if (den > 0) slopes.push(num / den);
    }
  }
  const meanDrop = drops.length ? drops.reduce((a, b) => a + b, 0) / drops.length : NaN;
  return {
    nPhrases: ph.length,
    meanPhraseS: ph.length ? ph.reduce((a, b) => a + b.dur, 0) / ph.length : NaN,
    longestPhraseS: ph.length ? Math.max(...ph.map(p => p.dur)) : NaN,
    meanDropDb: meanDrop,
    fadingPct: drops.length ? drops.filter(d => d > 4).length / drops.length * 100 : NaN,
    risingPct: slopes.length ? slopes.filter(s => s > 2).length / slopes.length * 100 : NaN,
    fallingPct: slopes.length ? slopes.filter(s => s < -2).length / slopes.length * 100 : NaN,
    nEndings: drops.length,
  };
}

/* ------------------------------------------------------------------ clarity and shape */

/** Share of energy in the 2-8 kHz band, where most consonant information lives. A mumbled
 *  delivery loses the consonants first, and the vowels carry on regardless. */
export function consonantEnergy(x, sr, mask) {
  if (sr < 15000) return NaN;                     // below this the band does not exist in the file
  const top = Math.min(8000, sr / 2 * 0.92);      // keep clear of Nyquist so the filter behaves
  const hi = bandRms(x, sr, 2000, top, mask);
  const all = bandRms(x, sr, 60, top, mask);
  return all > 0 ? (hi * hi) / (all * all) * 100 : NaN;
}

/** Least-squares trend of a per-frame series over speech frames, per minute.
 *  Negative loudness drift is a speaker running out of air or conviction as they go. */
export function trend(values, mask, perMinute = true) {
  const xs = [], ys = [];
  for (let i = 0; i < values.length && i < mask.length; i++) {
    if (mask[i] && isFinite(values[i])) { xs.push(i * HOP); ys.push(values[i]); }
  }
  if (xs.length < 30) return NaN;
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length, my = ys.reduce((a, b) => a + b, 0) / ys.length;
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) ** 2; }
  if (den <= 0) return NaN;
  return num / den * (perMinute ? 60 : 1);
}

/** Normalised pairwise variability of the gaps between syllables, the standard measure of speech
 *  rhythm. Near zero is a metronome. English conversation sits around 55-65; too low reads as
 *  chanting, too high as lurching. */
export function rhythmVariability(nuclei) {
  if (!nuclei || nuclei.length < 8) return NaN;
  const d = [];
  for (let i = 1; i < nuclei.length; i++) {
    const gap = nuclei[i] - nuclei[i - 1];
    if (gap > 0.04 && gap < 1.2) d.push(gap);       // ignore gaps that span a pause
  }
  if (d.length < 6) return NaN;
  let sum = 0, n = 0;
  for (let i = 0; i + 1 < d.length; i++) {
    const m = (d[i] + d[i + 1]) / 2;
    if (m > 0) { sum += Math.abs(d[i] - d[i + 1]) / m; n++; }
  }
  return n ? sum / n * 100 : NaN;
}

/* ------------------------------------------------------------------ placing pauses in the text */

/** Map a position in the script onto a moment in the recording.
 *
 *  No transcription is involved. The passage's syllable count is known and the syllable nuclei
 *  were detected acoustically, so the two can be laid against each other: the reader who has
 *  produced 120 of 240 detected nuclei is halfway through the text. Detection under-counts a
 *  little on real speech, which the ratio absorbs because it scales both sides. */
export function alignToScript(nuclei, script) {
  const n = nuclei ? nuclei.length : 0, total = script && script.syllables;
  if (n < 8 || !total) return null;
  const scale = n / total;
  const timeAt = j => nuclei[Math.max(0, Math.min(n - 1, Math.round(j * scale)))];
  /** How many script syllables had been spoken by time t. */
  const syllableAt = t => {
    let i = -1;
    while (i + 1 < n && nuclei[i + 1] < t) i++;
    return (i + 1) / scale;
  };
  return {
    detected: n, expected: total, scale, nuclei, timeAt, syllableAt,
    boundaryTimes: (script.boundaries || []).map(timeAt),
    commaTimes: (script.commas || []).map(timeAt),
  };
}

/** Sort each pause by where it falls in the sentence.
 *
 *  This is the distinction that separates a commanding reader from an anxious one. Both may be
 *  silent for the same number of seconds. The difference is that one stops after landing a
 *  sentence and the other stops in the middle of a clause, hunting for the next word. Across every
 *  great speaker measured for this project, at least 57% of pauses fell after a full stop, and the
 *  median was 68%. */
export function classifyPauses(pauseList, align, script, tolSyll = 1.5) {
  if (!align || !pauseList.length) return null;
  // Compared in syllables rather than seconds: a pause is judged by how far through the text the
  // reader was when they stopped, which does not change with how long the pause itself lasted.
  const near = (pos, arr) => arr.length ? Math.min(...arr.map(v => Math.abs(pos - v))) : Infinity;
  const stops = script.boundaries || [], commas = script.commas || [];
  let stop = 0, clause = 0, mid = 0;
  const midExamples = [];
  for (const p of pauseList) {
    const pos = align.syllableAt(p.start);
    if (near(pos, stops) <= tolSyll) stop++;
    else if (near(pos, commas) <= tolSyll) clause++;
    else { mid++; midExamples.push({ ...p, syllable: pos }); }
  }
  const n = stop + clause + mid;
  return {
    atFullStop: stop, atClause: clause, midClause: mid,
    atFullStopPct: n ? stop / n * 100 : NaN,
    midClausePct: n ? mid / n * 100 : NaN,
    longestMid: midExamples.sort((a, b) => b.dur - a.dur).slice(0, 4),
    tolerance: tolSyll,
  };
}

/** Per-sentence pace and loudness, so a reader can see which sentence ran away from them. */
export function perSentence(align, script, db, mask, f0, times) {
  if (!align || !script.sentenceList) return [];
  const out = [];
  for (const s of script.sentenceList) {
    const t0 = align.timeAt(s.from), t1 = align.timeAt(s.to);
    if (!(t1 > t0)) continue;
    const i0 = Math.round(t0 / HOP), i1 = Math.round(t1 / HOP);
    const lv = [];
    for (let i = i0; i < Math.min(i1, db.length); i++) if (mask[i]) lv.push(db[i]);
    const pv = [];
    for (let i = 0; i < times.length; i++) if (times[i] >= t0 && times[i] < t1) pv.push(f0[i]);
    const sorted = pv.slice().sort((a, b) => a - b);
    out.push({
      text: s.text, from: t0, to: t1, syllables: s.syllables,
      rate: s.syllables / (t1 - t0),
      level: lv.length ? lv.reduce((a, b) => a + b, 0) / lv.length : NaN,
      pitchRange: sorted.length > 10 ? 12 * Math.log2(pct(sorted, 90) / pct(sorted, 10)) : NaN,
    });
  }
  return out;
}

/* ------------------------------------------------------------------ the whole measurement */
/**
 * @param x   mono samples, already high-passed
 * @param sr  sample rate
 * @param script { words, sentences, syllables } of the passage that was read
 */
export async function analyse(x, sr, script, onProgress = () => {}) {
  const db = frameLevels(x, sr);
  const { mask, floor } = speechMask(db);
  const total = db.length * HOP;
  if (total < 8) throw new Error(`That take is only ${total.toFixed(1)} seconds. Read the whole passage so pace and pauses can be measured.`);
  const speech = mask.reduce((a, b) => a + (b ? 1 : 0), 0) * HOP;
  if (speech < 4) throw new Error("No speech found. Check the microphone was picking you up and that the take is not silent.");

  onProgress(0.15);
  const ps = pauses(mask);
  const dyn = dynamicRange(db, mask);
  const syl = syllablePeaks(db, mask, floor);
  const { f0, times, clarity } = await pitchTrack(x, sr, mask, p => onProgress(0.15 + 0.6 * p));
  const p = pitchStats(f0, times);
  const balance = lowEndBalance(x, sr, mask);
  onProgress(0.8);

  // the deeper reading: voice quality, what happens at the end of each phrase, and where in the
  // text each pause actually fell
  const ph = phrases(mask, ps);
  const endings = phraseEndings(db, mask, ph, f0, times, p.median);
  const voicedDb = [];
  for (let i = 0; i < times.length; i++) {
    const fi = Math.min(db.length - 1, Math.round(times[i] / HOP));
    voicedDb.push(Math.pow(10, db[fi] / 20));
  }
  const align = alignToScript(syl.at, script);
  const placement = align ? classifyPauses(ps, align, script) : null;
  const sentences = align ? perSentence(align, script, db, mask, f0, times) : [];
  onProgress(0.95);

  // The passage is known, so its words and syllables need no transcription. Pace is measured over
  // the real extent of speech, first voiced frame to last, so dead air at either end is not
  // counted against it.
  const firstF = mask.indexOf(true), lastF = mask.lastIndexOf(true);
  const extent = Math.max(1, (lastF - firstF + 1) * HOP);
  const words = script?.words || 0, sylCount = script?.syllables || 0;

  return {
    dur: total, speech, extent,
    wpm: words ? words / (extent / 60) : 0,
    artic: sylCount && speech ? sylCount / speech : 0,
    pitch: p.range, median: p.median, pitchSd: p.sd, mono: p.mono,
    dyn,
    ppm: ps.length / (extent / 60),
    plen: ps.length ? ps.reduce((a, b) => a + b.dur, 0) / ps.length : 0,
    nPauses: ps.length,
    longest: ps.length ? Math.max(...ps.map(q => q.dur)) : 0,
    pauseList: ps,
    completeness: sylCount ? syl.count / sylCount : 0,
    detectedSyllables: syl.count,
    floor, balance, bandLimited: balance < -12,

    // deeper reading
    hnr: hnr(clarity),
    pitchSteadiness: perturbation(f0, times, PITCH_HOP),
    levelSteadiness: perturbation(voicedDb, times, PITCH_HOP),
    consonant: consonantEnergy(x, sr, mask),
    levelDrift: trend(Array.from(db), mask),
    rhythm: rhythmVariability(syl.at),
    // the share of pauses that land after a full stop, scored alongside the rest: across every
    // great speaker measured for this project it never fell below 57%
    stops: placement ? placement.atFullStopPct : NaN,
    endings, placement, sentences,
    align: align ? { detected: align.detected, expected: align.expected, scale: align.scale } : null,

    series: { db, mask, f0, times, nuclei: syl.at,
              boundaries: align ? align.boundaryTimes : [] },
    scriptWords: words, scriptSentences: script?.sentences || 0,
  };
}

/* ------------------------------------------------------------------ scoring */
export function scoreBand(v, lo, hi) {
  if (!isFinite(v)) return null;
  if (v >= lo && v <= hi) return 100;
  const w = (hi - lo) || 1;
  const d = v < lo ? lo - v : v - hi;
  return Math.max(0, Math.round(100 - 100 * (d / (w * 0.9))));
}
export const grade = s => s >= 85 ? "good" : s >= 55 ? "fair" : "poor";
