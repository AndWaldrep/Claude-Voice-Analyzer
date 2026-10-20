/* Test signals with answers known in advance, plus a tiny runner and WAV io.
   Shared by the unit tests and the live-microphone test. */
import fs from "node:fs";

export function writeWav(path, x, sr) {
  const n = x.length, buf = Buffer.alloc(44 + n * 2);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write("WAVE", 8);
  buf.write("fmt ", 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(sr * 2, 28);
  buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36); buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) {
    buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, x[i])) * 32767), 44 + i * 2);
  }
  fs.writeFileSync(path, buf);
}

/** A glottal-style pulse train: harmonically rich like speech, so autocorrelation has a real
 *  chance to make an octave error. A pure sine would not test that. */
export function voiced(f0, dur, sr, amp = 0.3) {
  const n = Math.round(dur * sr), x = new Float32Array(n);
  let phase = 0;
  for (let i = 0; i < n; i++) {
    phase += f0 / sr;
    if (phase >= 1) phase -= 1;
    let s = 0;
    for (let h = 1; h <= 12; h++) s += Math.sin(2 * Math.PI * h * phase) / h;
    x[i] = amp * s / 1.6;
  }
  return x;
}

export function noise(dur, sr, amp) {
  const n = Math.round(dur * sr), x = new Float32Array(n);
  let s = 12345;
  for (let i = 0; i < n; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; x[i] = amp * ((s / 0x3fffffff) - 1); }
  return x;
}

export function concat(parts) {
  const n = parts.reduce((a, p) => a + p.length, 0), x = new Float32Array(n);
  let o = 0;
  for (const p of parts) { x.set(p, o); o += p.length; }
  return x;
}

/** Alternating voiced nuclei and near-silent closures, at a known syllable rate. */
export function syllableTrain(f0, nSyll, syllDur, sr, amp = 0.3) {
  const parts = [];
  for (let i = 0; i < nSyll; i++) {
    parts.push(voiced(f0 * (1 + 0.05 * Math.sin(i)), syllDur * 0.62, sr, amp));
    parts.push(noise(syllDur * 0.38, sr, amp * 0.02));
  }
  return concat(parts);
}

/** Speech-like passage with real sentence pauses, for the live test. */
export function passage(f0, sr, { phrases = 8, syllPerPhrase = 9, syllDur = 0.24, gap = 0.7 } = {}) {
  const parts = [];
  for (let p = 0; p < phrases; p++) {
    parts.push(syllableTrain(f0, syllPerPhrase, syllDur, sr, p % 3 === 0 ? 0.36 : 0.26));
    if (p < phrases - 1) parts.push(noise(gap, sr, 0.0005));
  }
  return concat(parts);
}

let pass = 0, fail = 0;
const lines = [];
export function check(name, ok, detail) {
  ok ? pass++ : fail++;
  lines.push((ok ? "  ok   " : "  FAIL ") + name + (detail ? "   " + detail : ""));
}
export function near(name, got, want, tol, unit = "") {
  const ok = Number.isFinite(got) && Math.abs(got - want) <= tol;
  check(name, ok, `got ${typeof got === "number" ? got.toFixed(2) : got}${unit}, want ${want}${unit} ±${tol}`);
  return ok;
}
export function report(title) {
  console.log("\n=== " + title + " ===");
  lines.forEach(l => console.log(l));
  lines.length = 0;
}
export function finish() {
  console.log(`\n${pass} passed, ${fail} failed\n`);
  return fail;
}
