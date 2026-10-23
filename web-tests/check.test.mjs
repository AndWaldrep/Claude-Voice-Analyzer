/* The diagnostics page, which exists for the case where the app itself cannot explain itself.
 *
 * It is the last thing standing when the record button does nothing, so it has to work under the
 * worst conditions: no modules, no microphone, and a phone screen. Both outcomes are exercised —
 * a microphone that answers and one that refuses — and the page must say which, in words, within
 * the first screenful.
 *
 *   node web-tests/check.test.mjs
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { passage, writeWav, check, report, finish } from "./signals.mjs";

const { chromium } = createRequire(import.meta.url)("playwright");
const HERE = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.join(HERE, "..", "docs");
const TMP = path.join(HERE, ".tmp");
const PORT = 8941;
const CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const PHONE = { width: 390, height: 844 };

const MIME = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json",
  ".webmanifest": "application/manifest+json", ".png": "image/png", ".wav": "audio/wav",
  ".md": "text/markdown" };

function serve(root, port) {
  const s = http.createServer((q, res) => {
    let f = path.join(root, decodeURIComponent(q.url.split("?")[0]));
    if (f.endsWith("/") || f.endsWith(path.sep)) f = path.join(f, "index.html");
    fs.readFile(f, (e, d) => {
      if (e) { res.writeHead(404); res.end("not found"); return; }
      res.writeHead(200, { "Content-Type": MIME[path.extname(f)] || "application/octet-stream",
                           "Cache-Control": "no-store" });
      res.end(d);
    });
  });
  return new Promise(r => s.listen(port, () => r(s)));
}

/** Everything the page is saying, as one string, plus where the verdict landed on screen. */
const readPage = page => page.evaluate(() => {
  const rows = [...document.querySelectorAll("#out tbody tr")]
    .map(tr => [...tr.children].map(td => td.textContent.trim()));
  const advice = document.querySelector("#advice .note");
  const r = advice ? advice.getBoundingClientRect() : null;
  return {
    rows,
    table: rows.map(r => r.join(": ")).join("\n"),
    advice: advice ? advice.textContent.trim() : "",
    adviceTop: r ? Math.round(r.top) : -1,
    viewport: window.innerHeight,
    live: document.getElementById("live").textContent.trim(),
  };
});
const valueOf = (rows, label) => (rows.find(r => r[0] === label) || [])[1] || "";

const main = async () => {
  fs.mkdirSync(TMP, { recursive: true });
  const wav = path.join(TMP, "check-mic.wav");
  writeWav(wav, passage(130, 16000, { phrases: 3, syllPerPhrase: 8, syllDur: 0.24, gap: 0.6 }), 16000);
  const server = await serve(DOCS, PORT);
  const base = ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
                "--disable-gpu", "--mute-audio"];

  /* ---------------------------------------------- the microphone answers */
  let browser = await chromium.launch({
    executablePath: CHROME,
    args: [...base, "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
           "--use-file-for-fake-audio-capture=" + wav, "--autoplay-policy=no-user-gesture-required"],
  });
  try {
    const ctx = await browser.newContext({ viewport: PHONE, permissions: ["microphone"] });
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", e => errors.push(e.message));
    await page.goto(`http://127.0.0.1:${PORT}/check.html`, { waitUntil: "load", timeout: 30000 });

    const first = await readPage(page);
    check("the environment is reported before anything is pressed", first.rows.length >= 8,
      `${first.rows.length} rows`);
    check("it names the browser", /Mozilla|Chrome|Safari/i.test(valueOf(first.rows, "Browser")),
      valueOf(first.rows, "Browser").slice(0, 50));
    check("it reports the secure context", valueOf(first.rows, "Secure connection").length > 0);
    check("it finds mediaDevices in a browser that has it",
      valueOf(first.rows, "navigator.mediaDevices") === "available");
    check("it finds Web Audio", valueOf(first.rows, "Web Audio") === "available");
    check("nothing is claimed about the microphone yet",
      !first.table.includes("Result"), first.table.slice(0, 60));

    await page.locator("#go").click();
    await page.waitForFunction(() => [...document.querySelectorAll("#out td")]
      .some(td => td.textContent === "Result"), null, { timeout: 20000 });
    await page.waitForTimeout(600);
    const got = await readPage(page);
    check("a granted microphone is reported as granted", valueOf(got.rows, "Result") === "granted",
      valueOf(got.rows, "Result"));
    check("the audio track is counted", Number(valueOf(got.rows, "Audio tracks")) >= 1,
      valueOf(got.rows, "Audio tracks"));
    check("the sample rate is reported", /\d/.test(valueOf(got.rows, "Sample rate")),
      valueOf(got.rows, "Sample rate"));
    check("gain control is confirmed off", valueOf(got.rows, "Gain control off") === "yes",
      valueOf(got.rows, "Gain control off"));
    check("the audio context opens", /running|suspended/.test(valueOf(got.rows, "Audio context")),
      valueOf(got.rows, "Audio context"));
    check("the verdict is that it works", /microphone is working/i.test(got.advice),
      got.advice.slice(0, 70));
    check("and the verdict is on the screen being looked at",
      got.adviceTop >= 0 && got.adviceTop < got.viewport,
      `advice at ${got.adviceTop}px in a ${got.viewport}px viewport`);

    // the meter must actually move: that is what proves sound is arriving, not just permission
    let moved = false;
    for (let i = 0; i < 40 && !moved; i++) {
      const live = await page.evaluate(() => document.getElementById("live").textContent);
      if (/dB/.test(live)) moved = true;
      else await page.waitForTimeout(100);
    }
    check("the level meter shows sound arriving", moved,
      await page.evaluate(() => document.getElementById("live").textContent));

    await page.locator("#copy").click();
    await page.waitForTimeout(1200);
    const copied = await page.evaluate(() => document.getElementById("copy").textContent);
    check("the report can be copied", /Copied|Select/.test(copied), copied);
    check("no uncaught errors with a microphone", errors.length === 0, errors.join(" | ") || "clean");
  } finally { await browser.close(); }

  /* ------------------------- refused before asking, as Brave on an iPhone did in use */
  browser = await chromium.launch({
    executablePath: CHROME,
    args: [...base, "--use-fake-device-for-media-stream"],
  });
  try {
    const ctx = await browser.newContext({
      viewport: PHONE, permissions: [],
      userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 " +
                 "(KHTML, like Gecko) Version/18.7 Mobile/15E148 Safari/604.1",
    });
    await ctx.addInitScript(() => {
      Object.defineProperty(navigator, "brave", { value: { isBrave: () => Promise.resolve(true) } });
    });
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", e => errors.push(e.message));
    await page.goto(`http://127.0.0.1:${PORT}/check.html`, { waitUntil: "load", timeout: 30000 });
    await page.locator("#go").click();
    await page.waitForFunction(() => [...document.querySelectorAll("#out td")]
      .some(td => td.textContent === "Plain request (no settings)"), null, { timeout: 20000 });
    const ios = await readPage(page);
    check("an iPhone is recognised", /^yes/.test(valueOf(ios.rows, "iPhone or iPad")));
    check("Brave is recognised", valueOf(ios.rows, "Looks like Brave") === "yes");
    check("the refusal is NotAllowedError", valueOf(ios.rows, "Result") === "refused: NotAllowedError",
      valueOf(ios.rows, "Result"));
    check("an instant refusal is called out as no prompt shown",
      /no prompt was shown/.test(valueOf(ios.rows, "Answered after")), valueOf(ios.rows, "Answered after"));
    check("the stored permission is reported", valueOf(ios.rows, "Stored permission for this site").length > 0,
      valueOf(ios.rows, "Stored permission for this site"));
    check("the microphones visible are counted",
      /^\d+$/.test(valueOf(ios.rows, "Microphones visible to the page")),
      valueOf(ios.rows, "Microphones visible to the page"));
    check("the plain request is tried too", /refused|granted/.test(valueOf(ios.rows, "Plain request (no settings)")),
      valueOf(ios.rows, "Plain request (no settings)"));
    check("the advice sends them to the iPhone's Settings for Brave",
      /Settings → Apps → Brave → Microphone/.test(ios.advice), ios.advice.slice(0, 90));
    check("and to Brave Shields", /Shields/.test(ios.advice));
    check("and to Safari as the fallback", /Safari/.test(ios.advice));
    check("the advice is on screen", ios.adviceTop >= 0 && ios.adviceTop < ios.viewport,
      `advice at ${ios.adviceTop}px`);
    check("no uncaught errors on an iPhone refusal", errors.length === 0, errors.join(" | ") || "clean");
  } finally { await browser.close(); }

  /* ---------------------------------------------- the microphone refuses */
  browser = await chromium.launch({ executablePath: CHROME, args: base });
  try {
    const ctx = await browser.newContext({ viewport: PHONE, permissions: [] });
    await ctx.clearPermissions();
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", e => errors.push(e.message));
    await page.goto(`http://127.0.0.1:${PORT}/check.html`, { waitUntil: "load", timeout: 30000 });
    await page.locator("#go").click();
    await page.waitForFunction(() => [...document.querySelectorAll("#out td")]
      .some(td => td.textContent === "Result"), null, { timeout: 20000 });
    const no = await readPage(page);

    check("a refusal is reported as a refusal", /refused/.test(valueOf(no.rows, "Result")),
      valueOf(no.rows, "Result"));
    check("the error is named", /Error/.test(valueOf(no.rows, "Result")), valueOf(no.rows, "Result"));
    check("the browser's own message is kept", valueOf(no.rows, "Message").length > 0,
      valueOf(no.rows, "Message").slice(0, 60));
    check("the advice says what to try", /shield|settings|safari|chrome|permission/i.test(no.advice),
      no.advice.slice(0, 80));
    check("the advice is on screen", no.adviceTop >= 0 && no.adviceTop < no.viewport,
      `advice at ${no.adviceTop}px in a ${no.viewport}px viewport`);
    check("the button comes back so it can be tried again",
      await page.evaluate(() => !document.getElementById("go").disabled));
    check("no uncaught errors on refusal", errors.length === 0, errors.join(" | ") || "clean");

    /* the main page must send a stuck user here */
    const app = await ctx.newPage();
    await app.goto(`http://127.0.0.1:${PORT}/index.html`, { waitUntil: "load", timeout: 30000 });
    await app.waitForFunction(() => !!window.__promptbook, null, { timeout: 15000 });
    await app.locator("#dockbtn").click();
    await app.waitForTimeout(2500);
    const link = await app.evaluate(() => {
      const a = document.querySelector("#dockmsg a");
      if (!a) return null;
      const r = a.getBoundingClientRect();
      return { href: a.getAttribute("href"), text: a.textContent,
               onScreen: r.top >= 0 && r.bottom <= window.innerHeight, top: Math.round(r.top) };
    });
    check("a refused microphone offers the check page", !!link && link.href === "./check.html",
      JSON.stringify(link));
    check("and that offer is within reach of the thumb", !!link && link.onScreen,
      link ? `link at ${link.top}px` : "no link");
    const reachable = await app.evaluate(async () =>
      (await fetch("./check.html")).status);
    check("the check page is actually there", reachable === 200, String(reachable));
    report("the microphone diagnostics page");
  } finally {
    await browser.close();
    server.close();
  }
  process.exit(finish() ? 1 : 0);
};

main().catch(e => { console.error("harness error:", e); process.exit(1); });
