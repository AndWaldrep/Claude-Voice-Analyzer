/* Measured distributions from a corpus of recognised speakers, so a reading can be placed against
   what good speaking actually looks like rather than against an invented ideal.

   Thirteen recordings, all public domain or Creative Commons, every one processed through this same
   measurement code: Brené Brown, Simon Sinek, Chimamanda Ngozi Adichie, Ken Robinson, Susan Cain,
   Amy Cuddy, Sarah Kay, Rita Pierson, Tony Robbins, Angela Duckworth, Derek Sivers, Seth Godin, and
   Kennedy's 1961 inaugural. See docs/BENCHMARKS.md for the provenance and the limits.

   The corpus disagrees with itself on nearly everything. Pace runs from 93 to 218 words per minute;
   pitch range from 9 to 17 semitones. So a percentile here is a question worth asking, not a verdict.
   Three things did converge, and those are the ones worth paying attention to:

     - where the pauses fall: every speaker put most of them after a full stop, median 68%
     - sentence-length contrast: all thirteen between 0.63 and 0.83
     - density of rhetorical figures: twelve of thirteen between 11 and 29 per thousand words
*/

export const CORPUS_SIZE = 13;

/* metric -> (p10, p25, p50, p75, p90) */
export const PERCENTILES = {
  wpm:           [140.01, 141.88, 153.29, 175.64, 178.41],
  artic:         [4.29, 4.90, 5.09, 5.37, 5.46],
  pitch:         [9.42, 10.67, 11.77, 13.57, 14.83],
  mono:          [1.54, 9.27, 12.62, 17.50, 23.32],
  dyn:           [4.49, 4.58, 7.87, 10.06, 10.85],
  ppm:           [14.05, 14.74, 15.77, 17.42, 17.86],
  plen:          [0.72, 0.82, 0.86, 1.06, 1.10],
  atFullStopPct: [58.17, 63.25, 68.08, 74.23, 77.15],
};

export const LABELS = {
  wpm: "Pace", artic: "Articulation rate", pitch: "Pitch range",
  mono: "Monotone stretches", dyn: "Volume contrast", ppm: "Pauses per minute",
  plen: "Mean pause", atFullStopPct: "Pauses landing after a full stop",
};

/** Metrics where a higher reading is the worse one. */
export const LOWER_IS_BETTER = new Set(["mono"]);

/** Where a value sits in the corpus, by interpolation between the stored decile points. */
export function percentileOf(metric, value) {
  const pts = PERCENTILES[metric];
  if (!pts || !isFinite(value)) return null;
  const qs = [10, 25, 50, 75, 90];
  if (value <= pts[0]) return pts[0] > 0 ? Math.max(0, 10 * value / pts[0]) : 0;
  if (value >= pts[4]) return 95;
  for (let i = 0; i < 4; i++) {
    if (value >= pts[i] && value <= pts[i + 1]) {
      const span = pts[i + 1] - pts[i];
      return qs[i] + (qs[i + 1] - qs[i]) * (span > 0 ? (value - pts[i]) / span : 0);
    }
  }
  return null;
}

/** Percentile plus a phrase that agrees with it, so the words and the number never contradict. */
export function compare(metric, value) {
  const p = percentileOf(metric, value);
  if (p === null) return { p: null, verdict: "" };
  const inverted = LOWER_IS_BETTER.has(metric);
  if (p < 10) return { p, verdict: inverted ? "more than the whole corpus" : "below the whole corpus" };
  if (p < 30) return { p, verdict: "lower than most of the corpus" };
  if (p <= 70) return { p, verdict: "mid-range for the corpus" };
  if (p <= 90) return { p, verdict: "high end of the corpus" };
  return { p, verdict: inverted ? "higher than the whole corpus" : "above the whole corpus" };
}
