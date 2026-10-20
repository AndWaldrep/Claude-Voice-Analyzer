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
  const f0 = [], times = [];
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
    if (hz >= 70 && hz <= 400) { f0.push(hz); times.push(t); }
  }
  return { f0, times };
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
  const { f0, times } = await pitchTrack(x, sr, mask, p => onProgress(0.15 + 0.7 * p));
  const p = pitchStats(f0, times);
  const balance = lowEndBalance(x, sr, mask);
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
    series: { db, mask, f0, times },
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
