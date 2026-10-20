import { MODES, METRICS, scriptStats, sentencesOf } from "./modes.js";
import * as DSP from "./dsp.js";
import { Recorder, toWav, decodeFile } from "./capture.js";

const $ = id => document.getElementById(id);
const clock = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

let modeKey = "oratory";
let stats = { words: 0, sentences: 0, syllables: 0 };
let last = null;
let history = [];
const rec = new Recorder();

/* ------------------------------------------------------------------ script */
function renderModes() {
  const ul = $("modes");
  ul.innerHTML = "";
  for (const [k, m] of Object.entries(MODES)) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.className = "mode"; b.type = "button";
    b.setAttribute("aria-pressed", String(k === modeKey));
    b.innerHTML = `<b>${m.label}</b><span>${m.tag}</span>`;
    b.addEventListener("click", () => {
      modeKey = k; renderModes(); renderScript();
      if (last) renderResults(last);     // rescore the same take against the new form
      renderHistory();
    });
    li.appendChild(b); ul.appendChild(li);
  }
}

function renderScript() {
  const m = MODES[modeKey];
  $("cite").innerHTML = m.cite;
  const host = $("script");
  host.innerHTML = "";
  for (const para of m.text.split(/\n+/)) {
    const p = document.createElement("p");
    const ss = sentencesOf(para);
    ss.forEach((sent, i) => {
      p.appendChild(document.createTextNode(sent + " "));
      if (i < ss.length - 1) {
        const b = document.createElement("span");
        b.className = "beat"; b.textContent = "¶"; b.setAttribute("aria-hidden", "true");
        p.appendChild(b);
      }
    });
    host.appendChild(p);
  }
  stats = scriptStats(m.text);
  $("wc").textContent = stats.words;
  $("sc").textContent = stats.sentences;
  const mid = (m.bands.wpm[0] + m.bands.wpm[1]) / 2;
  $("tt").textContent = clock(stats.words / mid * 60);

  const t = $("targets");
  t.innerHTML = "";
  for (const x of METRICS) {
    if (x.key === "mono") continue;
    const b = m.bands[x.key];
    const d = document.createElement("div");
    d.innerHTML = `<dt>${x.label}</dt><dd>${b[0].toFixed(x.dec)}&ndash;${b[1].toFixed(x.dec)}${x.unit}</dd>`;
    t.appendChild(d);
  }
}

/* ------------------------------------------------------------------ recording */
function targetSeconds() {
  const b = MODES[modeKey].bands.wpm;
  return stats.words / ((b[0] + b[1]) / 2) * 60;
}

function setLive(on) {
  $("livepanel").hidden = !on;
  $("howto").hidden = on;
  for (const id of ["recbtn", "dockbtn"]) {
    $(id).classList.toggle("recording", on);
    $(id).classList.toggle("stop", on);
  }
  $("recLabel").textContent = on ? "Stop and score" : "Record";
  $("dockLabel").textContent = on ? "Stop" : "Record";
  if (on) {
    // the good-level zone: -26 to -9 dBFS maps onto the meter's -60..0 span
    for (const id of ["mzone", "dzone"]) {
      $(id).style.left = ((-26 + 60) / 60 * 100) + "%";
      $(id).style.width = (17 / 60 * 100) + "%";
    }
    $("ltarget").textContent = clock(targetSeconds());
    $("dtarget").textContent = "of " + clock(targetSeconds());
  } else {
    $("dlvl").textContent = "ready";
  }
}

rec.onLevel = ({ dbfs, pitchHz, seconds }) => {
  const pos = Math.max(0, Math.min(100, (dbfs + 60) / 60 * 100));
  $("mfill").style.width = pos + "%";
  $("dfill").style.width = pos + "%";
  $("delapsed").textContent = clock(seconds);
  $("dlvl").textContent = dbfs > -59 ? dbfs.toFixed(0) + " dB" : "quiet";
  $("lvl").textContent = dbfs > -59 ? dbfs.toFixed(0) + " dB" : "quiet";
  $("lpitch").textContent = pitchHz ? Math.round(pitchHz) + " Hz" : "—";
  $("lelapsed").textContent = clock(seconds);
  const tgt = targetSeconds();
  const frac = Math.min(1.3, seconds / tgt);
  const f = $("pacefill");
  f.style.width = Math.min(100, frac * 100) + "%";
  f.className = frac > 1.08 ? "fast" : frac < 0.55 ? "slow" : "";
};

async function toggleRecord() {
  const btns = [$("recbtn"), $("dockbtn")];
  if (rec.state === "recording") {
    btns.forEach(b => b.disabled = true);
    const take = await rec.stop();
    setLive(false);
    btns.forEach(b => b.disabled = false);
    if (take && take.samples.length) await runAnalysis(take.samples, take.sampleRate, take);
    return;
  }
  if (!Recorder.supported()) {
    showNote("This browser cannot record audio. Use the file option, or open the page in Safari or Chrome.");
    return;
  }
  try {
    $("micnote").hidden = true;
    const info = await rec.start();
    setLive(true);
    if (info.sampleRate < 16000) {
      showNote(`The microphone is running at ${info.sampleRate} Hz, which is low. Pitch may be coarse.`);
    }
  } catch (e) {
    const n = e && e.name;
    showNote(
      n === "NotAllowedError" ? "Microphone access was refused. Allow it for this page in your browser settings, then press record again."
      : n === "NotFoundError" ? "No microphone was found on this device."
      : n === "NotReadableError" ? "The microphone is in use by another app. Close it and try again."
      : "Could not start recording: " + (e && e.message ? e.message : "unknown error"));
  }
}

function showNote(msg) {
  const n = $("micnote");
  n.textContent = msg;
  n.hidden = false;
}

/* ------------------------------------------------------------------ analysis */
async function runAnalysis(samples, sampleRate, take) {
  const bar = $("bar"), fill = bar.firstElementChild, err = $("err");
  err.hidden = true; bar.hidden = false; fill.style.width = "5%";
  try {
    const x = DSP.highpass(samples, sampleRate, 80);
    const r = await DSP.analyse(x, sampleRate, stats, p => { fill.style.width = Math.round(p * 100) + "%"; });
    r.sampleRate = sampleRate;
    renderResults(r);
    if (take) {
      const url = URL.createObjectURL(toWav(samples, sampleRate));
      const a = document.createElement("audio");
      a.controls = true; a.src = url;
      $("results").appendChild(a);
    }
    saveAttempt(r);
  } catch (e) {
    err.textContent = e && e.message ? e.message : "That audio could not be analysed.";
    err.hidden = false;
  } finally {
    setTimeout(() => { bar.hidden = true; fill.style.width = "0"; }, 500);
  }
}

function renderResults(r) {
  last = r;
  const m = MODES[modeKey];
  const scores = {};
  for (const x of METRICS) scores[x.key] = DSP.scoreBand(r[x.key], m.bands[x.key][0], m.bands[x.key][1]);
  const vals = METRICS.map(x => scores[x.key]).filter(v => v !== null);
  const overall = Math.round(vals.reduce((a, b) => a + b, 0) / Math.max(1, vals.length));
  const worst = METRICS.filter(x => scores[x.key] !== null).sort((a, b) => scores[a.key] - scores[b.key])[0];
  const C = 2 * Math.PI * 32;

  let html = `<div class="overall">
    <div class="ring">
      <svg width="76" height="76" viewBox="0 0 76 76" aria-hidden="true">
        <circle cx="38" cy="38" r="32" fill="none" stroke="var(--ground)" stroke-width="7"></circle>
        <circle cx="38" cy="38" r="32" fill="none" stroke="var(--${DSP.grade(overall)})" stroke-width="7"
          stroke-linecap="round" stroke-dasharray="${C.toFixed(1)}"
          stroke-dashoffset="${(C * (1 - overall / 100)).toFixed(1)}"></circle>
      </svg><b>${overall}</b>
    </div>
    <div class="verdict">
      <strong>${overall >= 85 ? "Close to the mark for " + m.label.toLowerCase()
        : overall >= 55 ? "Workable, with one thing to fix" : "Some way off what this form calls for"}</strong>
      <span>${worst && scores[worst.key] < 85 ? "Weakest element: " + worst.label.toLowerCase() + "." : "Nothing stands out as off."}
      ${clock(r.dur)} read &middot; ${Math.round(r.speech / r.dur * 100)}% of it speaking
      ${r.sampleRate ? "&middot; " + (r.sampleRate / 1000).toFixed(1) + " kHz" : ""}</span>
    </div></div>`;

  if (r.bandLimited) {
    html += `<p class="note">Little energy below 150 Hz (${r.balance.toFixed(0)} dB under the formant band).
      If this is a low voice its fundamental may be missing, and the pitch figures below then read about an
      octave high. Switch off noise cancelling and record closer to the mouth.</p>`;
  }
  if (r.completeness < 0.75) {
    html += `<p class="note">Only about ${Math.round(r.completeness * 100)}% of the passage's syllables were
      detected, so pace may be understated. Pace assumes all ${r.scriptWords} words were read.</p>`;
  }

  html += `<div class="metrics">`;
  for (const x of METRICS) {
    const v = r[x.key], s = scores[x.key], band = m.bands[x.key];
    if (s === null) {
      html += `<div class="m"><div class="mtop"><b>${x.label}</b><i>not measurable</i></div></div>`;
      continue;
    }
    const [a0, a1] = x.axis;
    const cl = n => Math.max(0, Math.min(100, (n - a0) / (a1 - a0) * 100));
    const z0 = cl(band[0]), z1 = cl(band[1]);
    const note = s < 85 ? (v < band[0] ? x.low : x.high) : "";
    html += `<div class="m">
      <div class="mtop">
        <b>${x.label}${m.focus.includes(x.key) ? ' <span class="pill fair">key for this form</span>' : ""}</b>
        <i>${v.toFixed(x.dec)}${x.unit} <span class="pill ${DSP.grade(s)}">${s}</span></i>
      </div>
      <div class="track" role="img" aria-label="${x.label} ${v.toFixed(x.dec)}${x.unit}, target ${band[0]} to ${band[1]}">
        <span class="zone" style="left:${z0}%;width:${Math.max(1.5, z1 - z0)}%"></span>
        <span class="pin" style="left:calc(${cl(v)}% - 1.25px)"></span>
      </div>
      ${note ? `<p class="mnote off">${note}</p>`
             : `<p class="mnote">Target ${band[0].toFixed(x.dec)}&ndash;${band[1].toFixed(x.dec)}${x.unit}.</p>`}
    </div>`;
  }
  html += `</div>
    <div class="chartwrap"><h3>Loudness and silence</h3><canvas id="cvL" height="92"></canvas></div>
    <div class="chartwrap"><h3>Pitch contour</h3><canvas id="cvP" height="92"></canvas></div>
    <details><summary>What was measured</summary><ul>
      <li>Pace assumes the full passage: ${r.scriptWords} words over ${r.extent.toFixed(1)} seconds of speech.</li>
      <li>${r.nPauses} pauses of 0.3 s or more; longest ${r.longest.toFixed(1)} s. Gaps over 5 s count as stops, not delivery.</li>
      <li>Median pitch ${isFinite(r.median) ? Math.round(r.median) + " Hz" : "not found"}; noise floor ${Math.round(r.floor)} dBFS; low-end balance ${r.balance.toFixed(0)} dB.</li>
      <li>${r.detectedSyllables} syllable nuclei detected against ${stats.syllables} in the text.</li>
      <li>The passage has ${r.scriptSentences} full stops. Aim for at least that many pauses, landed on them.</li>
    </ul></details>`;

  $("results").innerHTML = html;
  drawLevel($("cvL"), r);
  drawPitch($("cvP"), r);
}

/* ------------------------------------------------------------------ charts */
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function drawLevel(cv, r) {
  if (!cv || !r.series) return;
  const W = cv.clientWidth || 420, H = 92, dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const pad = { l: 30, r: 6, t: 8, b: 16 }, iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const db = r.series.db, lo = r.floor - 2;
  let mx = -Infinity;
  for (let i = 0; i < db.length; i++) if (db[i] > mx) mx = db[i];
  const hi = mx + 2, Y = v => pad.t + ih * (1 - (v - lo) / (hi - lo));
  g.font = "9px 'IBM Plex Mono', monospace"; g.textAlign = "right";
  for (const v of [hi, (hi + lo) / 2, lo]) {
    const y = Math.round(Y(v)) + .5;
    g.strokeStyle = css("--rule"); g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.fillStyle = css("--muted"); g.fillText(Math.round(v) + "", pad.l - 5, y + 3);
  }
  g.fillStyle = css("--band");
  let run = 0;
  for (let i = 0; i <= db.length; i++) {
    if (i < db.length && !r.series.mask[i]) { run++; continue; }
    if (run * DSP.HOP >= DSP.MIN_PAUSE) {
      g.fillRect(pad.l + iw * ((i - run) / db.length), pad.t, Math.max(1, iw * (run / db.length)), ih);
    }
    run = 0;
  }
  g.strokeStyle = css("--brass"); g.lineWidth = 1; g.beginPath();
  for (let i = 0; i < db.length; i++) {
    const x = pad.l + iw * (i / db.length), y = Y(Math.max(lo, db[i]));
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.stroke();
  g.fillStyle = css("--muted");
  g.textAlign = "left"; g.fillText("0:00", pad.l, H - 4);
  g.textAlign = "right"; g.fillText(clock(r.dur), W - pad.r, H - 4);
  g.textAlign = "center"; g.fillText("shaded = silence", pad.l + iw / 2, H - 4);
}

function drawPitch(cv, r) {
  if (!cv || !r.series) return;
  const W = cv.clientWidth || 420, H = 92, dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const pad = { l: 30, r: 6, t: 8, b: 16 }, iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  if (!r.series.f0.length) {
    g.fillStyle = css("--muted"); g.font = "11px 'IBM Plex Sans', sans-serif"; g.textAlign = "center";
    g.fillText("No voiced pitch found", W / 2, H / 2); return;
  }
  const s = r.series.f0.slice().sort((a, b) => a - b);
  const lo = Math.max(60, DSP.pct(s, 2) - 10), hi = DSP.pct(s, 98) + 10;
  const Y = v => pad.t + ih * (1 - (v - lo) / (hi - lo));
  g.font = "9px 'IBM Plex Mono', monospace";
  for (const v of [hi, r.median, lo]) {
    if (!isFinite(v)) continue;
    const y = Math.round(Y(v)) + .5;
    g.strokeStyle = css("--rule"); g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.fillStyle = css("--muted"); g.textAlign = "right"; g.fillText(Math.round(v) + "", pad.l - 5, y + 3);
  }
  g.fillStyle = css("--brass");
  for (let i = 0; i < r.series.f0.length; i++) {
    g.fillRect(pad.l + iw * (r.series.times[i] / r.dur), Y(r.series.f0[i]) - 1, 1.6, 2);
  }
  g.fillStyle = css("--muted"); g.textAlign = "center";
  g.fillText("Hz over time", pad.l + iw / 2, H - 4);
}

let rt;
window.addEventListener("resize", () => {
  clearTimeout(rt);
  rt = setTimeout(() => { if (last) { drawLevel($("cvL"), last); drawPitch($("cvP"), last); } }, 160);
});

/* ------------------------------------------------------------------ history */
const KEY = "promptbook.v1";
function loadHistory() {
  try { history = JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (e) { history = []; }
}
function saveAttempt(r) {
  const m = MODES[modeKey];
  const scores = METRICS.map(x => DSP.scoreBand(r[x.key], m.bands[x.key][0], m.bands[x.key][1])).filter(v => v !== null);
  history.push({
    at: Date.now(), mode: modeKey,
    overall: Math.round(scores.reduce((a, b) => a + b, 0) / Math.max(1, scores.length)),
    wpm: r.wpm, pitch: r.pitch, dyn: r.dyn, ppm: r.ppm, plen: r.plen,
  });
  history = history.slice(-200);
  try { localStorage.setItem(KEY, JSON.stringify(history)); } catch (e) {}
  renderHistory();
}
function renderHistory() {
  const host = $("hist");
  const rows = history.filter(h => h.mode === modeKey).slice(-8).reverse();
  if (!rows.length) { host.hidden = true; return; }
  host.hidden = false;
  host.innerHTML = `<h3>Your ${MODES[modeKey].label.toLowerCase()} attempts</h3>
    <div style="overflow-x:auto"><table><thead><tr>
      <th>When</th><th>Score</th><th>Pace</th><th>Pitch</th><th>Contrast</th><th>Pauses</th>
    </tr></thead><tbody>${rows.map(h => `<tr>
      <td>${new Date(h.at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</td>
      <td>${h.overall}</td><td>${h.wpm.toFixed(0)}</td><td>${h.pitch.toFixed(1)}</td>
      <td>${h.dyn.toFixed(1)}</td><td>${h.ppm.toFixed(1)}</td></tr>`).join("")}</tbody></table></div>`;
}

/* ------------------------------------------------------------------ wiring */
$("recbtn").addEventListener("click", toggleRecord);
$("dockbtn").addEventListener("click", toggleRecord);
$("audio").addEventListener("change", async e => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    const { samples, sampleRate } = await decodeFile(f);
    await runAnalysis(samples, sampleRate, { samples, sampleRate });
  } catch (err) {
    $("err").textContent = "That file could not be decoded. Try exporting it as m4a or wav.";
    $("err").hidden = false;
  }
});
let size = 19;
const setSize = d => {
  size = Math.max(14, Math.min(28, size + d));
  $("script").style.setProperty("--scriptsize", size + "px");
};
$("bigger").addEventListener("click", () => setSize(1));
$("smaller").addEventListener("click", () => setSize(-1));

renderModes();
renderScript();
loadHistory();
renderHistory();
if (!Recorder.supported()) showNote("This browser cannot reach a microphone. You can still analyse a recorded file.");
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("./sw.js").catch(() => {}));
}

// exposed so the test harness can drive the real analysis path
window.__promptbook = { DSP, MODES, METRICS, rec, runAnalysis, getStats: () => stats,
                        setMode: k => { modeKey = k; renderModes(); renderScript(); },
                        getLast: () => last };
