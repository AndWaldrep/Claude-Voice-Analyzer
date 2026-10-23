/* What a phone sees when the microphone does not come back.
 *
 * The happy path is covered elsewhere. This is the other half, and it is where the app actually
 * failed in use: the microphone was refused, the explanation rendered 1668 pixels down an 844
 * pixel screen because it sat in a card below the passage, and the button simply looked dead.
 * Anything the app needs to say has to arrive within reach of the thumb that pressed the button.
 *
 *   node web-tests/mobile.test.mjs
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { check, report, finish } from "./signals.mjs";

const { chromium } = createRequire(import.meta.url)("playwright");
const HERE = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.join(HERE, "..", "docs");
const PORT = 8931;
const CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const PHONE = { width: 390, height: 844 };

const MIME = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json",
  ".webmanifest": "application/manifest+json", ".png": "image/png", ".md": "text/markdown" };

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

const main = async () => {
  const server = await serve(DOCS, PORT);
  const browser = await chromium.launch({
    executablePath: CHROME,
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--mute-audio"],
  });
  try {
    // no fake device and no permission: the microphone will not be granted
    const ctx = await browser.newContext({ viewport: PHONE, permissions: [] });
    await ctx.clearPermissions();
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", e => errors.push(e.message));
    await page.goto(`http://127.0.0.1:${PORT}/index.html`, { waitUntil: "load", timeout: 30000 });
    await page.waitForFunction(() => !!window.__promptbook, null, { timeout: 15000 });

    check("the dock control is within reach on a phone", await page.locator("#dockbtn").isVisible());
    const before = await page.evaluate(() => document.getElementById("dockmsg").hidden);
    check("nothing is shouted before the button is pressed", before);

    await page.locator("#dockbtn").click();
    await page.waitForTimeout(2500);

    const seen = await page.evaluate(() => {
      const d = document.getElementById("dockmsg"), r = d.getBoundingClientRect();
      return {
        shown: !d.hidden,
        onScreen: !d.hidden && r.top >= 0 && r.bottom <= window.innerHeight,
        top: Math.round(r.top), viewport: window.innerHeight,
        text: (d.textContent || "").trim(),
        alsoInCard: !document.getElementById("micnote").hidden,
        label: document.getElementById("dockLabel").textContent,
        state: window.__promptbook.rec.state,
      };
    });
    check("pressing it says something", seen.shown, seen.text.slice(0, 80));
    check("and says it on the screen the user is looking at", seen.onScreen,
      `message at ${seen.top}px in a ${seen.viewport}px viewport`);
    check("the same thing is said in the card for a wider screen", seen.alsoInCard);
    check("the message explains what to do", /shield|allow|settings|safari|chrome|https|voice memo|found/i.test(seen.text),
      seen.text.slice(0, 110));
    check("the button returns to its resting state", seen.label === "Record" && seen.state === "idle",
      `label "${seen.label}", recorder ${seen.state}`);
    check("no uncaught errors", errors.length === 0, errors.join(" | ") || "clean");

    // a second press must behave the same, not wedge
    await page.locator("#dockbtn").click();
    await page.waitForTimeout(2000);
    const again = await page.evaluate(() => ({
      shown: !document.getElementById("dockmsg").hidden,
      state: window.__promptbook.rec.state,
      label: document.getElementById("dockLabel").textContent,
    }));
    check("pressing again behaves the same way", again.shown && again.state === "idle" && again.label === "Record",
      JSON.stringify(again));

    // the page must still be useful without a microphone
    check("analysing a recorded file is still offered", await page.locator('input[type="file"]').count() === 1);
    report("a phone with no microphone");
  } finally {
    await browser.close();
    server.close();
  }
  process.exit(finish() ? 1 : 0);
};

main().catch(e => { console.error("harness error:", e); process.exit(1); });
