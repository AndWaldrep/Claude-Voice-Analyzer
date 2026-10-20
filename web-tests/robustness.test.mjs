/* Does pause placement survive an imperfect reader?
 *
 * Placing a pause in the text needs no transcript: the passage's syllables are known and the
 * reader's are detected, so the two can be laid against each other. That works until the reader
 * departs from the page, and these are the cases that matter, because a confidently wrong number
 * is worse than no number. Each case below reads the real Henry V passage with one specific
 * imperfection and asserts either an accurate measurement or a visible caveat.
 *
 *   node web-tests/robustness.test.mjs
 */
import * as DSP from "../docs/dsp.js";
import { MODES, scriptStats } from "../docs/modes.js";
import { syllableTrain, noise, concat, check, near, report, finish } from "./signals.mjs";

const SR = 16000;
const script = scriptStats(MODES.oratory.text);

/** Read the passage aloud with one chosen imperfection. */
function read({ skip = -1, repeat = -1, extra = 0, syllDur = 0.22, gap = 0.9, mid = false } = {}) {
  const parts = [];
  if (mid) {
    // pause often, but always well clear of a full stop. Near a comma is fine: a breath at a
    // comma is its own category, and what must never happen is counting one as a landed stop.
    let spoken = 0;
    for (let pos = 6; pos < script.syllables - 6; pos += 7) {
      if (Math.min(...script.boundaries.map(a => Math.abs(pos - a))) < 8) continue;
      const need = pos - spoken;
      if (need > 0) { parts.push(syllableTrain(120, need, syllDur, SR)); spoken = pos; }
      parts.push(noise(gap, SR, 0.0005));
    }
    const tail = script.syllables - spoken;
    if (tail > 0) parts.push(syllableTrain(120, tail, syllDur, SR));
    return DSP.highpass(concat(parts), SR);
  }
  script.sentenceList.forEach((s, i) => {
    if (i === skip) return;
    parts.push(syllableTrain(120, s.syllables * (i === repeat ? 2 : 1) + (i === 0 ? extra : 0), syllDur, SR));
    parts.push(noise(gap, SR, 0.0005));
  });
  return DSP.highpass(concat(parts), SR);
}

const faithful = async (name, opts, want) => {
  const r = await DSP.analyse(read(opts), SR, script);
  check(`${name}: measured`, r.placementTrusted && !r.placementCaveat,
    `caveat ${r.placementCaveat || "none"}, completeness ${(r.completeness * 100).toFixed(0)}%`);
  near(`  ${name}: pauses placed correctly`, r.placement.atFullStopPct, want, 15, "%");
};

const deviating = async (name, opts, expectCaveat) => {
  const r = await DSP.analyse(read(opts), SR, script);
  check(`${name}: flagged`, !r.placementTrusted || r.placementCaveat === expectCaveat,
    `caveat ${r.placementCaveat || "none"}, completeness ${(r.completeness * 100).toFixed(0)}%, ` +
    `would have reported ${r.placement ? r.placement.atFullStopPct.toFixed(0) + "%" : "nothing"}`);
};

// read as written, at every rate a person speaks at: the pauses must be placed accurately
await faithful("steady reading", {}, 100);
await faithful("fast reading", { syllDur: 0.13 }, 100);
await faithful("slow, grave reading", { syllDur: 0.34 }, 100);
await faithful("very slow reading", { syllDur: 0.40 }, 100);
await faithful("brief pauses", { gap: 0.45 }, 100);

// the finding the whole measurement exists for: read faithfully, but never land a pause.
// This must be reported, not explained away and not refused.
await faithful("every pause mid-clause", { mid: true }, 0);
report("a reader who follows the page");

// departures from the text cannot be placed against it, and must say so rather than guess
await deviating("skips a sentence", { skip: 2 }, "incomplete");
await deviating("skips a later sentence", { skip: 5 }, "incomplete");
await deviating("repeats a sentence", { repeat: 3 }, "extra");
await deviating("ad-libs a short opener", { extra: 12 }, "extra");
await deviating("ad-libs a long opener", { extra: 30 }, "extra");
report("a reader who departs from the page");

/* Nobody reads a passage word for word. They miss a word, swap one, rephrase a clause. None of
   that should move the measurement, and before this was fixed it moved it enormously: dropping
   four syllables out of 221, under two percent, took a flawless reading from 100% to 50%. */
function readNatural(n, delta, seed = 1) {
  const parts = [];
  let s = seed;
  const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const touched = new Set();
  while (touched.size < Math.min(n, script.sentenceList.length)) {
    touched.add(Math.floor(rnd() * script.sentenceList.length));
  }
  script.sentenceList.forEach((sent, i) => {
    parts.push(syllableTrain(120, Math.max(3, sent.syllables + (touched.has(i) ? delta : 0)), 0.22, SR));
    parts.push(noise(0.9, SR, 0.0005));
  });
  return DSP.highpass(concat(parts), SR);
}

for (const [n, delta, what] of [
  [1, -2, "misses a word"], [3, -2, "misses a word in three sentences"],
  [5, -2, "misses a word in five"], [9, -2, "misses a word in every sentence"],
  [4, -4, "drops a few words in four sentences"], [9, -4, "drops a few words throughout"],
  [1, 2, "adds a word"], [3, 2, "adds a word in three sentences"],
  [5, 2, "adds a word in five"], [3, 4, "rephrases three clauses longer"],
]) {
  const r = await DSP.analyse(readNatural(n, delta), SR, script);
  near(`${what} (${n > 0 ? n * delta : 0} syllables)`, r.placement.atFullStopPct, 100, 13, "%");
}
report("a reader who is human about it");

/* A reader who pauses the same short distance before every full stop has gaps that match the text
   perfectly, so matching on gaps alone would accept them. They are not landing their pauses. */
{
  const build = points => {
    const parts = [];
    let spoken = 0;
    for (const p of points) {
      const need = p - spoken;
      if (need > 0) { parts.push(syllableTrain(120, need, 0.22, SR)); spoken = p; }
      parts.push(noise(0.9, SR, 0.0005));
    }
    const tail = script.syllables - spoken;
    if (tail > 0) parts.push(syllableTrain(120, tail, 0.22, SR));
    return DSP.highpass(concat(parts), SR);
  };
  for (const early of [5, 7]) {
    const r = await DSP.analyse(build(script.boundaries.map(b => Math.max(3, b - early))), SR, script);
    near(`stopping ${early} syllables short of every full stop`, r.placement.atFullStopPct, 0, 5, "%");
  }
  report("a reader who stops just short, every time");
}

// the alignment must not be free enough to explain anything it is shown
{
  const x = read({ mid: true });
  const db = DSP.frameLevels(x, SR);
  const { mask, floor } = DSP.speechMask(db);
  const align = DSP.alignToScript(DSP.syllablePeaks(db, mask, floor).at, script);
  const ps = DSP.pauses(mask);
  const tight = DSP.classifyPauses(ps, align, script);
  const loose = DSP.classifyPauses(ps, align, script, { scaleSpan: 0.15, offSpan: 4 });
  check("a free scale lets the alignment explain anything", loose.atFullStopPct > tight.atFullStopPct + 20,
    `nailed down ${tight.atFullStopPct.toFixed(0)}%, given freedom ${loose.atFullStopPct.toFixed(0)}%`);
  near("  which is why it is nailed down", tight.atFullStopPct, 0, 5, "%");
  report("why the search is constrained");
}

process.exit(finish() ? 1 : 0);
