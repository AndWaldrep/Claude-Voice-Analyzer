/* Microphone capture to raw mono samples.
 *
 * Two things here matter more than the rest of the file.
 *
 * 1. The browser's own voice processing is switched OFF. Automatic gain control flattens loud
 *    against quiet, which is precisely the volume contrast being measured; noise suppression
 *    strips the low end, which removes the fundamental of a low voice and makes pitch read an
 *    octave high. Left on, the defaults would quietly corrupt three of the seven metrics.
 *
 * 2. Nothing is encoded. Samples are tapped straight off the graph, so the analysis sees what the
 *    microphone produced and playback is built from the same array. This also sidesteps
 *    MediaRecorder, whose container support still differs between Safari and Chrome.
 */

export const CONSTRAINTS = {
  audio: {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    channelCount: 1,
  },
};

export class Recorder {
  constructor() {
    this.chunks = [];
    this.frames = 0;
    this.sampleRate = 0;
    this.state = "idle";
    this.onLevel = null;          // called with { rms, dbfs, pitchHz, seconds }
    this._wake = null;
  }

  /** Whether this browser can record at all. */
  static supported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia &&
              (window.AudioContext || window.webkitAudioContext));
  }

  /** Must be called from inside a click or tap: iOS will not grant audio otherwise. */
  async start() {
    if (this.state === "recording") return;
    this.chunks = [];
    this.frames = 0;

    this.stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
    // A browser that hands back a stream with nothing in it has refused without saying so.
    if (!this.stream.getAudioTracks || this.stream.getAudioTracks().length === 0) {
      try { this.stream.getTracks().forEach(t => t.stop()); } catch (e) {}
      const err = new Error("the microphone returned no audio");
      err.name = "NotReadableError";
      throw err;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AC();
    // Safari hands back a suspended context; it only starts inside the user gesture.
    if (this.ctx.state === "suspended") await this.ctx.resume();
    this.sampleRate = this.ctx.sampleRate;

    const src = this.ctx.createMediaStreamSource(this.stream);

    // live meter, read on animation frames rather than in the audio path
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.2;
    src.connect(this.analyser);

    let tapped = false;
    if (this.ctx.audioWorklet) {
      try {
        await this.ctx.audioWorklet.addModule("./worklet.js");
        this.node = new AudioWorkletNode(this.ctx, "tap");
        this.node.port.onmessage = e => { this.chunks.push(e.data); this.frames += e.data.length; };
        src.connect(this.node);
        // a worklet with no output still needs a sink in some engines
        const silent = this.ctx.createGain();
        silent.gain.value = 0;
        this.node.connect(silent).connect(this.ctx.destination);
        tapped = true;
      } catch (e) { tapped = false; }
    }
    if (!tapped) {
      // older iOS: deprecated but universally present
      const size = 4096;
      this.node = this.ctx.createScriptProcessor(size, 1, 1);
      this.node.onaudioprocess = e => {
        const d = e.inputBuffer.getChannelData(0);
        this.chunks.push(new Float32Array(d));
        this.frames += d.length;
      };
      src.connect(this.node);
      const silent = this.ctx.createGain();
      silent.gain.value = 0;
      this.node.connect(silent).connect(this.ctx.destination);
    }
    this.usedWorklet = tapped;

    this.state = "recording";
    this._requestWakeLock();
    this._meter();
    return { sampleRate: this.sampleRate, worklet: tapped };
  }

  /** Stop and return the take as one mono Float32Array. */
  async stop() {
    if (this.state !== "recording") return null;
    this.state = "idle";
    cancelAnimationFrame(this._raf);
    this._releaseWakeLock();
    try { this.stream.getTracks().forEach(t => t.stop()); } catch (e) {}
    try { if (this.node) this.node.disconnect(); } catch (e) {}
    const out = new Float32Array(this.frames);
    let o = 0;
    for (const c of this.chunks) { out.set(c, o); o += c.length; }
    const sr = this.sampleRate;
    try { await this.ctx.close(); } catch (e) {}
    return { samples: out, sampleRate: sr };
  }

  /* live level and a coarse pitch, for feedback while reading */
  _meter() {
    const buf = new Float32Array(this.analyser.fftSize);
    const tick = () => {
      if (this.state !== "recording") return;
      this.analyser.getFloatTimeDomainData(buf);
      let s = 0;
      for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
      const rms = Math.sqrt(s / buf.length);
      if (this.onLevel) {
        this.onLevel({
          rms,
          dbfs: 20 * Math.log10(rms + 1e-10),
          pitchHz: rms > 0.004 ? quickPitch(buf, this.sampleRate) : 0,
          seconds: this.frames / this.sampleRate,
        });
      }
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  async _requestWakeLock() {
    try { this._wake = await navigator.wakeLock?.request("screen"); } catch (e) { this._wake = null; }
  }
  _releaseWakeLock() {
    try { this._wake?.release(); } catch (e) {}
    this._wake = null;
  }
}

/** A cheap autocorrelation for the live readout only. The reported figures come from dsp.js. */
export function quickPitch(buf, sr) {
  const minLag = Math.floor(sr / 400), maxLag = Math.min(Math.floor(sr / 70), buf.length - 1);
  let mean = 0;
  for (let i = 0; i < buf.length; i++) mean += buf[i];
  mean /= buf.length;
  let best = 0, bestLag = -1;
  for (let lag = minLag; lag <= maxLag; lag += 2) {
    let num = 0, d1 = 0, d2 = 0;
    for (let i = 0; i < buf.length - lag; i += 2) {
      const a = buf[i] - mean, b = buf[i + lag] - mean;
      num += a * b; d1 += a * a; d2 += b * b;
    }
    const r = num / (Math.sqrt(d1 * d2) + 1e-12);
    if (r > best) { best = r; bestLag = lag; }
  }
  if (bestLag < 0 || best < 0.4) return 0;
  const hz = sr / bestLag;
  return hz >= 70 && hz <= 400 ? hz : 0;
}

/** Wrap mono samples as a 16-bit WAV, for playback and for saving a take. */
export function toWav(samples, sr) {
  const n = samples.length;
  const b = new ArrayBuffer(44 + n * 2), v = new DataView(b);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, Math.round(s * 32767), true);
  }
  return new Blob([b], { type: "audio/wav" });
}

/** Read a dropped or chosen audio file into mono samples. */
export async function decodeFile(file) {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  try {
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    const n = buf.length, out = new Float32Array(n);
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < n; i++) out[i] += d[i];
    }
    if (buf.numberOfChannels > 1) {
      const k = 1 / buf.numberOfChannels;
      for (let i = 0; i < n; i++) out[i] *= k;
    }
    return { samples: out, sampleRate: buf.sampleRate };
  } finally { try { await ctx.close(); } catch (e) {} }
}
