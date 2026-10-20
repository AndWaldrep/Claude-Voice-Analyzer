/* End-to-end test of the real microphone path.
 *
 * Chromium is given a synthetic speech-like file as its microphone, so everything the app does on
 * a phone runs here for real: getUserMedia with processing disabled, the audio worklet tap, the
 * raw sample buffer, the analysis, and the rendered score. The signal's pitch is known in advance,
 * which is what makes the assertion meaningful.
 *
 *   node web-tests/live.test.mjs
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
// playwright is installed globally here, which ESM resolution does not search
const { chromium } = createRequire(import.meta.url)("playwright");
import { passage, writeWav, check, near, report, finish } from "./signals.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.join(HERE, "..", "docs");
const TMP = path.join(HERE, ".tmp");
const PORT = 8911;
const CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const F0 = 130;                       // the pitch the fake microphone will produce

const MIME = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json",
  ".webmanifest": "application/manifest+json", ".png": "image/png", ".wav": "audio/wav",
  ".css": "text/css", ".md": "text/markdown" };

function serve(root, port) {
  const s = http.createServer((q, res) => {
    let f = path.join(root, decodeURIComponent(q.url.split("?")[0]));
    if (f.endsWith("/") || f.endsWith(path.sep)) f = path.join(f, "index.html");
    fs.readFile(f, (e, d) => {
      if (e) { res.writeHead(404); res.end("not found"); return; }
      res.writeHead(200, { "Content-Type": MIME[path.extname(f)] || "application/octet-stream",
                           "Service-Worker-Allowed": "/", "Cache-Control": "no-store" });
      res.end(d);
    });
  });
  return new Promise(r => s.listen(port, () => r(s)));
}

const main = async () => {
  fs.mkdirSync(TMP, { recursive: true });
  // 24 s of speech-like audio at a known pitch, with real gaps between phrases
  const wav = path.join(TMP, "mic.wav");
  writeWav(wav, passage(F0, 16000, { phrases: 10, syllPerPhrase: 9, syllDur: 0.24, gap: 0.7 }), 16000);

  const server = await serve(DOCS, PORT);
  const browser = await chromium.launch({
    executablePath: CHROME,
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
           "--mute-audio", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
           "--use-file-for-fake-audio-capture=" + wav,
           "--autoplay-policy=no-user-gesture-required"],
  });
  let failed = 0;
  try {
    const ctx = await browser.newContext({ permissions: ["microphone"] });
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", e => errors.push(e.message));
    page.on("console", m => { if (m.type() === "error") errors.push("console: " + m.text()); });

    await page.goto(`http://127.0.0.1:${PORT}/index.html`, { waitUntil: "load", timeout: 30000 });
    await page.waitForFunction(() => !!window.__promptbook, null, { timeout: 15000 });
    check("the page loads and the module initialises", true);
    check("no errors on load", errors.length === 0, errors.join(" | ") || "clean");

    // the first screen is usable before anything is recorded
    check("a mode is selected on arrival",
      await page.locator('.mode[aria-pressed="true"]').count() === 1);
    const wc = +(await page.locator("#wc").textContent());
    check("the model text is counted", wc > 80, `${wc} words`);
    check("target ranges are shown", await page.locator("#targets div").count() >= 6);

    // record through the fake microphone
    await page.click("#recbtn");
    await page.waitForFunction(() => window.__promptbook.rec.state === "recording", null, { timeout: 15000 });
    check("recording starts from a click", true);
    check("the live panel replaces the instructions", await page.locator("#livepanel").isVisible());

    await page.waitForTimeout(1500);
    const live = await page.evaluate(() => ({
      lvl: document.getElementById("lvl").textContent,
      pitch: document.getElementById("lpitch").textContent,
      width: document.getElementById("mfill").style.width,
    }));
    check("the live meter moves", live.width !== "" && live.width !== "0%", JSON.stringify(live));
    const livePitch = parseInt(live.pitch, 10);
    near("the live pitch readout is right", livePitch, F0, 14, " Hz");

    const constraints = await page.evaluate(() => {
      const t = window.__promptbook.rec.stream.getAudioTracks()[0];
      return t.getSettings ? t.getSettings() : {};
    });
    check("gain control and noise suppression are off",
      constraints.autoGainControl !== true && constraints.noiseSuppression !== true &&
      constraints.echoCancellation !== true,
      JSON.stringify({ agc: constraints.autoGainControl, ns: constraints.noiseSuppression, ec: constraints.echoCancellation }));
    check("capture uses the audio worklet", await page.evaluate(() => window.__promptbook.rec.usedWorklet === true));

    await page.waitForTimeout(13000);         // ~15 s of audio in total
    await page.click("#recbtn");
    await page.waitForFunction(() => window.__promptbook.getLast() !== null, null, { timeout: 60000 });
    check("stopping produces an analysis", true);

    const r = await page.evaluate(() => {
      const r = window.__promptbook.getLast();
      return { dur: r.dur, speech: r.speech, median: r.median, pitch: r.pitch, dyn: r.dyn,
               ppm: r.ppm, plen: r.plen, nPauses: r.nPauses, wpm: r.wpm, artic: r.artic,
               floor: r.floor, balance: r.balance, bandLimited: r.bandLimited,
               completeness: r.completeness, sampleRate: r.sampleRate };
    });
    console.log("\n  captured through the fake microphone:");
    console.log("   ", JSON.stringify(r, (k, v) => typeof v === "number" ? +v.toFixed(2) : v));

    near("the whole take was captured", r.dur, 14.5, 3, " s");
    near("pitch survives capture and analysis", r.median, F0, 12, " Hz");
    check("the sample rate is the device's own", r.sampleRate >= 16000, `${r.sampleRate} Hz`);
    // the signal repeats a 2.86 s cycle (9 syllables of 0.24 s, then a 0.7 s gap), and gaps at the
    // very start and end are excluded by design, so expect one fewer than the phrases captured
    const cycle = 9 * 0.24 + 0.7;
    const expectGaps = Math.max(1, Math.round(r.dur / cycle) - 1);
    check("phrase gaps are found as pauses", Math.abs(r.nPauses - expectGaps) <= 1,
      `${r.nPauses} pauses, expected about ${expectGaps}`);
    near("mean pause matches the 0.7 s gaps", r.plen, 0.7, 0.3, " s");
    check("volume contrast is measured", Number.isFinite(r.dyn) && r.dyn > 0, `${r.dyn.toFixed(1)} dB`);
    check("a full-bandwidth take is not flagged band-limited", r.bandLimited === false, `balance ${r.balance.toFixed(1)} dB`);
    check("every reported metric is finite", [r.wpm, r.artic, r.pitch, r.dyn, r.ppm, r.plen].every(Number.isFinite));

    // the score actually renders
    const ui = await page.evaluate(() => ({
      ring: document.querySelector(".ring b")?.textContent,
      rows: document.querySelectorAll(".metrics .m").length,
      audio: !!document.querySelector("#results audio"),
      hist: document.querySelectorAll(".hist tbody tr").length,
      charts: document.querySelectorAll("#results canvas").length,
    }));
    check("an overall score is rendered", /^\d+$/.test(ui.ring || ""), "score " + ui.ring);
    check("every metric row is rendered", ui.rows === 7, `${ui.rows} rows`);
    check("both charts are drawn", ui.charts === 2, `${ui.charts} canvases`);
    check("playback of the take is offered", ui.audio);
    check("the attempt is saved to history", ui.hist === 1, `${ui.hist} rows`);

    // switching form rescores the same take, which is the point of having forms
    const before = ui.ring;
    await page.click('.mode:has-text("Eulogy")');
    const after = await page.evaluate(() => document.querySelector(".ring b")?.textContent);
    check("switching form rescores the same reading", before !== after, `${before} as oratory, ${after} as eulogy`);

    check("still no page errors", errors.length === 0, errors.join(" | ") || "clean");
  } finally {
    await browser.close();
    server.close();
  }
  report("live microphone, end to end");
  failed = finish();
  process.exit(failed ? 1 : 0);
};

main().catch(e => { console.error("harness error:", e); process.exit(1); });
