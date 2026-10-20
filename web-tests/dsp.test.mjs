/* Unit tests for the shipped measurement code: node web-tests/dsp.test.mjs */
import * as DSP from "../docs/dsp.js";
import { MODES, METRICS, syllables, scriptStats, sentencesOf, words } from "../docs/modes.js";
import { voiced, noise, concat, syllableTrain, check, near, report, finish } from "./signals.mjs";

const SR = 16000;

/* ---------- text ---------- */
{
  const REF = [["organization",5],["complacency",4],["information",4],["management",3],["statement",2],
    ["movement",2],["the",1],["strength",1],["accountability",6],["responsibility",6],["opportunity",5],
    ["people",2],["business",2],["idea",3],["area",3],["hour",1],["fire",1],["crispian",3],["ocean",2],
    ["christian",2],["remembered",3],["freedom",2],["liberty",3],["nation",2],["dedicated",4],
    ["walked",1],["wanted",2],["needed",2],["plea",1],["sea",1],["audacity",4],["senate",2],["happiness",3]];
  const wrong = REF.filter(([w, n]) => syllables(w) !== n).map(([w, n]) => `${w}=${syllables(w)}/${n}`);
  check(`syllables exact on ${REF.length} reference words`, wrong.length === 0, wrong.join(", ") || "all exact");
  const got = REF.reduce((a, [w]) => a + syllables(w), 0), want = REF.reduce((a, [, n]) => a + n, 0);
  near("no net syllable bias", got, want, 1, " syllables");

  for (const [k, m] of Object.entries(MODES)) {
    const s = scriptStats(m.text);
    check(`${k}: text parses`, s.words > 80 && s.sentences >= 5,
      `${s.words} words, ${s.sentences} sentences, ${s.syllables} syllables, ${(s.syllables / s.words).toFixed(2)} per word`);
    const missing = METRICS.map(x => x.key).filter(x => !Array.isArray(m.bands[x]));
    check(`${k}: every target band present`, missing.length === 0, missing.join(",") || "complete");
    const bad = METRICS.map(x => x.key).filter(x => m.bands[x] && m.bands[x][0] >= m.bands[x][1]);
    check(`${k}: bands ordered low<high`, bad.length === 0, bad.join(",") || "ok");
    const badFocus = (m.focus || []).filter(f => !METRICS.some(x => x.key === f));
    check(`${k}: focus keys valid`, badFocus.length === 0, badFocus.join(",") || "ok");
  }
  report("text, scripts and modes");
}

/* ---------- levels ---------- */
{
  const n = SR * 3, x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = 0.5 * Math.sin(2 * Math.PI * 200 * i / SR);
  near("frame level of a 0.5 sine", DSP.frameLevels(x, SR)[75], -9.03, 0.15, " dBFS");

  const loud = new Float32Array(SR * 3), quiet = new Float32Array(SR * 3);
  for (let i = 0; i < SR * 3; i++) {
    loud[i] = 0.5 * Math.sin(2 * Math.PI * 150 * i / SR);
    quiet[i] = 0.1255 * Math.sin(2 * Math.PI * 150 * i / SR);
  }
  const sig = concat([loud, noise(0.5, SR, 0.0005), quiet]);
  const db = DSP.frameLevels(sig, SR);
  near("dynamic range of two blocks 12 dB apart", DSP.dynamicRange(db, DSP.speechMask(db).mask), 12, 1.5, " dB");
  report("levels");
}

/* ---------- pauses ---------- */
{
  const parts = [];
  for (let i = 0; i < 4; i++) { parts.push(voiced(120, 2, SR)); if (i < 3) parts.push(noise(1.0, SR, 0.0004)); }
  const mask = DSP.speechMask(DSP.frameLevels(DSP.highpass(concat(parts), SR), SR)).mask;
  const ps = DSP.pauses(mask);
  check("finds exactly the 3 interior pauses", ps.length === 3, `found ${ps.length}`);
  if (ps.length) near("mean pause length", ps.reduce((a, b) => a + b.dur, 0) / ps.length, 1.0, 0.12, " s");

  const longGap = DSP.highpass(concat([voiced(120, 2, SR), noise(7, SR, 0.0004), voiced(120, 2, SR)]), SR);
  check("a 7 s gap counts as a stop, not a pause",
    DSP.pauses(DSP.speechMask(DSP.frameLevels(longGap, SR)).mask).length === 0);
  const edges = DSP.highpass(concat([noise(2, SR, 0.0004), voiced(120, 3, SR), noise(2, SR, 0.0004)]), SR);
  check("silence at the ends is not a pause",
    DSP.pauses(DSP.speechMask(DSP.frameLevels(edges, SR)).mask).length === 0);
  report("pauses");
}

/* ---------- pitch ---------- */
{
  for (const f0 of [85, 100, 120, 150, 185, 220]) {
    const x = DSP.highpass(voiced(f0, 4, SR), SR);
    const mask = DSP.speechMask(DSP.frameLevels(x, SR)).mask;
    const { f0: tr, times } = await DSP.pitchTrack(x, SR, mask);
    near(`median pitch of a ${f0} Hz pulse train`, DSP.pitchStats(tr, times).median, f0, Math.max(3, f0 * 0.04), " Hz");
    const off = tr.filter(v => v < f0 * 0.6 || v > f0 * 1.7).length;
    check(`  no octave jumps at ${f0} Hz`, off / Math.max(1, tr.length) < 0.05, `${off}/${tr.length} frames`);
  }
  const two = DSP.highpass(concat([voiced(100, 3, SR), voiced(200, 3, SR)]), SR);
  const m2 = DSP.speechMask(DSP.frameLevels(two, SR)).mask;
  const t2 = await DSP.pitchTrack(two, SR, m2);
  near("100 to 200 Hz reads as one octave", DSP.pitchStats(t2.f0, t2.times).range, 12, 2, " st");

  const flat = DSP.highpass(voiced(120, 12, SR), SR);
  const mf = DSP.speechMask(DSP.frameLevels(flat, SR)).mask;
  const tf = await DSP.pitchTrack(flat, SR, mf);
  near("constant pitch is ~100% monotone", DSP.pitchStats(tf.f0, tf.times).mono, 100, 12, "%");
  report("pitch");
}

/* ---------- syllable nuclei ---------- */
{
  for (const [n, d] of [[40, 0.15], [30, 0.20], [24, 0.25], [20, 0.30], [17, 0.35]]) {
    const x = DSP.highpass(syllableTrain(130, n, d, SR), SR);
    const db = DSP.frameLevels(x, SR);
    const { mask, floor } = DSP.speechMask(db);
    const got = DSP.syllablePeaks(db, mask, floor).count;
    check(`detects ~${n} nuclei at ${d}s each`, Math.abs(got - n) / n < 0.25, `got ${got} (${Math.round(got / n * 100)}%)`);
  }
  report("syllable nuclei");
}

/* ---------- scoring ---------- */
{
  check("at band low scores 100", DSP.scoreBand(10, 10, 20) === 100);
  check("at band high scores 100", DSP.scoreBand(20, 10, 20) === 100);
  check("one band-width below scores 0", DSP.scoreBand(1, 10, 20) === 0);
  check("just outside is high but under 100", DSP.scoreBand(9.5, 10, 20) > 90 && DSP.scoreBand(9.5, 10, 20) < 100);
  check("NaN is not scored", DSP.scoreBand(NaN, 10, 20) === null);
  check("zero-width band does not divide by zero", Number.isFinite(DSP.scoreBand(5, 10, 10)));
  report("scoring");
}

/* ---------- end to end, known pace ---------- */
{
  const n = 60, d = 0.25;
  const script = { words: 40, sentences: 5, syllables: 60 };
  const r = await DSP.analyse(DSP.highpass(syllableTrain(120, n, d, SR), SR), SR, script);
  near("duration", r.dur, n * d, 0.3, " s");
  near("words per minute", r.wpm, 40 / ((n * d) / 60), 12, " wpm");
  check("completeness near 1", Math.abs(r.completeness - 1) < 0.3, `${r.completeness.toFixed(2)}`);

  const SR2 = 48000;
  const r2 = await DSP.analyse(DSP.highpass(syllableTrain(120, n, d, SR2), SR2), SR2, script);
  near("48 kHz gives the same pitch", r2.median, r.median, 6, " Hz");
  near("48 kHz gives the same pace", r2.wpm, r.wpm, 6, " wpm");
  near("48 kHz gives the same articulation rate", r2.artic, r.artic, 0.5, " syll/s");
  report("end to end");
}

/* ---------- recording quality and refusals ---------- */
{
  const script = { words: 40, sentences: 5, syllables: 60 };
  const expectThrow = async (name, x, sr) => {
    try { await DSP.analyse(x, sr, script); check(name, false, "did not throw"); }
    catch (e) { check(name, true, e.message.slice(0, 56)); }
  };
  await expectThrow("pure silence is refused", new Float32Array(SR * 20), SR);
  await expectThrow("a 3 s take is refused as too short", DSP.highpass(voiced(120, 3, SR), SR), SR);

  const intact = await DSP.analyse(DSP.highpass(syllableTrain(110, 60, 0.25, SR), SR), SR, script);
  check("full-bandwidth voice is not flagged", intact.bandLimited === false,
    `balance ${intact.balance.toFixed(1)} dB, median ${intact.median.toFixed(0)} Hz`);
  near("  and reads its true pitch", intact.median, 110, 6, " Hz");

  let gutted = syllableTrain(110, 60, 0.25, SR);
  for (let i = 0; i < 4; i++) gutted = DSP.biquad(gutted, SR, "hp", 200);
  const low = await DSP.analyse(gutted, SR, script);
  check("stripping the low end lowers the balance measure", low.balance < intact.balance - 5,
    `${intact.balance.toFixed(1)} dB -> ${low.balance.toFixed(1)} dB`);

  const high = await DSP.analyse(DSP.highpass(syllableTrain(215, 60, 0.25, SR), SR), SR, script);
  near("a genuine high voice still measures correctly", high.median, 215, 10, " Hz");
  check("  and is not falsely flagged", high.bandLimited === false, `balance ${high.balance.toFixed(1)} dB`);

  const quiet = DSP.highpass(syllableTrain(120, 60, 0.25, SR).map(v => v * 0.01), SR);
  const rq = await DSP.analyse(quiet, SR, script);
  check("a very quiet take still analyses", Number.isFinite(rq.median), `median ${rq.median.toFixed(0)} Hz, floor ${rq.floor.toFixed(0)} dBFS`);

  const clipped = DSP.highpass(syllableTrain(120, 60, 0.25, SR).map(v => Math.max(-1, Math.min(1, v * 8))), SR);
  check("a clipped take still analyses", Number.isFinite((await DSP.analyse(clipped, SR, script)).dyn));

  const rw = await DSP.analyse(DSP.highpass(noise(20, SR, 0.2), SR), SR, script);
  check("unvoiced noise does not crash", true, `pitch ${Number.isFinite(rw.pitch) ? rw.pitch.toFixed(1) + " st" : "not measurable"}`);

  const rz = await DSP.analyse(DSP.highpass(syllableTrain(120, 40, 0.25, SR), SR), SR, { words: 0, sentences: 0, syllables: 0 });
  check("an empty script does not produce Infinity", Number.isFinite(rz.wpm) && Number.isFinite(rz.artic), `wpm ${rz.wpm}, artic ${rz.artic}`);
  report("recording quality and refusals");
}


/* ---------- the deeper reading ---------- */
{
  const { MODES: MM, scriptStats: ss } = await import("../docs/modes.js");
  const script = ss(MM.oratory.text);

  // read the real script structure aloud, pausing where the text tells you to
  const build = (where, { fade = 0, syllDur = 0.22, gap = 0.9 } = {}) => {
    const marks = where === "stops" ? script.boundaries : script.boundaries.map(b => Math.max(3, b - 5));
    const parts = []; let spoken = 0, k = 0;
    for (const m of marks) {
      const need = m - spoken;
      if (need > 0) {
        const amp = 0.3 * Math.pow(10, -(fade * k) / 20);
        parts.push(syllableTrain(120, need, syllDur, SR, amp));
        spoken = m;
      }
      parts.push(noise(gap, SR, 0.0005)); k++;
    }
    const tail = script.syllables - spoken;
    if (tail > 0) parts.push(syllableTrain(120, tail, syllDur, SR, 0.3));
    return DSP.highpass(concat(parts), SR);
  };

  const atStops = await DSP.analyse(build("stops"), SR, script);
  const midSent = await DSP.analyse(build("mid"), SR, script);

  check("alignment recovers nearly every syllable",
    Math.abs(atStops.align.detected - atStops.align.expected) / atStops.align.expected < 0.1,
    `${atStops.align.detected} detected of ${atStops.align.expected} in the text`);
  near("pauses at full stops are placed there", atStops.placement.atFullStopPct, 100, 1, "%");
  check("pauses inside sentences are not", midSent.placement.atFullStopPct < 20,
    `${midSent.placement.atFullStopPct.toFixed(0)}% at full stops, ${midSent.placement.midClause} mid-clause`);
  check("the mid-clause pauses are reported with their position",
    midSent.placement.longestMid.length > 0 && isFinite(midSent.placement.longestMid[0].syllable));

  check("phrases match the sentences they were built from",
    Math.abs(atStops.endings.nPhrases - script.sentences) <= 1,
    `${atStops.endings.nPhrases} phrases, ${script.sentences} sentences`);
  check("one row per sentence is reported", atStops.sentences.length === script.sentences,
    `${atStops.sentences.length} rows`);
  check("each sentence row carries a rate and a level",
    atStops.sentences.every(s => isFinite(s.rate) && isFinite(s.level)));

  // a reading that fades 2 dB per phrase should register as fading, a steady one should not
  const fading = await DSP.analyse(build("stops", { fade: 2 }), SR, script);
  check("a steady reading does not read as trailing off", atStops.endings.fadingPct < 20,
    `${atStops.endings.fadingPct.toFixed(0)}% of endings fade`);
  check("a fading reading has negative loudness drift", fading.levelDrift < atStops.levelDrift - 1,
    `steady ${atStops.levelDrift.toFixed(2)} dB/min vs fading ${fading.levelDrift.toFixed(2)} dB/min`);

  // voice quality: a clean tone is harmonic, noise is not
  const clean = await DSP.analyse(DSP.highpass(syllableTrain(120, 60, 0.25, SR), SR), SR, script);
  check("a clean voice has high harmonics-to-noise", clean.hnr > 12, `${clean.hnr.toFixed(1)} dB`);
  check("a steady pitch reads as steady", clean.pitchSteadiness < 3, `${clean.pitchSteadiness.toFixed(2)}%`);
  const breathy = concat([syllableTrain(120, 60, 0.25, SR), noise(0.1, SR, 0)]).map((v, i, a) => v + 0.05 * Math.sin(i * 12.9898) * Math.cos(i * 0.137));
  const br = await DSP.analyse(DSP.highpass(Float32Array.from(breathy), SR), SR, script);
  check("added noise lowers harmonics-to-noise", br.hnr < clean.hnr,
    `${clean.hnr.toFixed(1)} dB clean vs ${br.hnr.toFixed(1)} dB noisy`);

  // rhythm: a metronomic train is regular, varied syllable lengths are not
  const varied = [];
  for (let i = 0; i < 60; i++) varied.push(syllableTrain(120, 1, 0.14 + 0.22 * ((i * 7) % 5) / 4, SR));
  const rv = await DSP.analyse(DSP.highpass(concat(varied), SR), SR, script);
  check("varied syllable lengths read as less regular than a metronome",
    rv.rhythm > clean.rhythm, `metronome ${clean.rhythm.toFixed(0)} vs varied ${rv.rhythm.toFixed(0)}`);

  check("consonant energy is measured at 16 kHz", isFinite(clean.consonant), `${clean.consonant.toFixed(2)}%`);
  report("the deeper reading");
}

/* ---------- benchmarks ---------- */
{
  const B = await import("../docs/benchmarks.js");
  for (const [m, pts] of Object.entries(B.PERCENTILES)) {
    const ordered = pts.every((v, i) => i === 0 || v >= pts[i - 1]);
    check(`${m}: corpus deciles are ordered`, ordered, pts.join(", "));
    const lo = B.percentileOf(m, pts[0] * 0.5), mid = B.percentileOf(m, pts[2]), hi = B.percentileOf(m, pts[4] * 1.5);
    check(`${m}: percentile rises with the value`, lo < mid && mid < hi,
      `${lo?.toFixed(0)} < ${mid?.toFixed(0)} < ${hi?.toFixed(0)}`);
  }
  check("a median value reads as mid-range", B.compare("wpm", 153.29).verdict === "mid-range for the corpus");
  check("a very low value reads as below the corpus", B.compare("wpm", 60).verdict === "below the whole corpus");
  check("for monotone, high is the bad direction",
    B.compare("mono", 50).verdict === "higher than the whole corpus");
  check("an unknown metric is not scored", B.percentileOf("nonsense", 5) === null);
  check("the client's 55% pause placement sits near the corpus floor",
    B.percentileOf("atFullStopPct", 55) < 12, `p${B.percentileOf("atFullStopPct", 55)?.toFixed(0)}`);
  report("benchmarks");
}

process.exit(finish() ? 1 : 0);
