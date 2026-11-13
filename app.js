import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.all.min.mjs";
import { DemucsProcessor, CONSTANTS } from "./index.js";
import { extractAmbience } from "./ambience.js";

const $ = id => document.getElementById(id), SR = CONSTANTS.SAMPLE_RATE;
const NAMES = { vocal: "Vocal", drum: "Drum", bass: "Bass", other: "Other", ambience: "Ambience", music: "Music" };
const status = (t, pct) => { $("stage").textContent = t; if (pct != null) $("fill").style.width = pct + "%"; };
const tick = () => new Promise(r => setTimeout(r, 0));
let processor = null, backend = "CPU";

async function getModel() {
  const url = CONSTANTS.DEFAULT_MODEL_URL; let cache = null;
  try { cache = await caches.open("stem-model-v1"); const hit = await cache.match(url); if (hit) return await hit.arrayBuffer(); } catch {}
  const res = await fetch(url); if (!res.ok) throw new Error("Model download failed (HTTP " + res.status + ")");
  const total = +res.headers.get("content-length") || 180e6, reader = res.body.getReader(), parts = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length;
    status(`Downloading AI model (first time only): ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`, Math.min(100, got / total * 100)); }
  const blob = new Blob(parts); try { await cache?.put(url, new Response(blob)); } catch {}
  return await blob.arrayBuffer();
}
async function getProcessor() {
  if (processor) return processor;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  const buf = await getModel(); status("Starting AI engine...", 3); await tick();
  let gpu = false; try { gpu = !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch {}
  for (const eps of gpu ? [["webgpu", "wasm"], ["wasm"]] : [["wasm"]]) {
    try {
      backend = eps[0] === "webgpu" ? "GPU" : "CPU";
      const p = new DemucsProcessor({ ort, sessionOptions: { executionProviders: eps, enableCpuMemArena: false, enableMemPattern: false },
        onProgress: ({ progress }) => status(`Separating on ${backend}... ${(progress * 100).toFixed(0)}%`, 5 + progress * 80) });
      await p.loadModel(buf.slice(0)); processor = p; return p;
    } catch (e) { console.warn("engine failed", eps, e); }
  }
  throw new Error("Could not start the AI engine in this browser. Try the latest Chrome or Edge on a desktop PC.");
}
async function decode(file) {
  const ctx = new AudioContext({ sampleRate: SR });
  const ab = await ctx.decodeAudioData(await file.arrayBuffer()); ctx.close();
  const l = ab.getChannelData(0); return [l, ab.numberOfChannels > 1 ? ab.getChannelData(1) : l];
}
const sum = (...a) => { const o = new Float32Array(a[0].length); for (const x of a) for (let i = 0; i < o.length; i++) o[i] += x[i]; return o; };
const diff = (a, b) => { const o = new Float32Array(a.length); for (let i = 0; i < o.length; i++) o[i] = a[i] - b[i]; return o; };

function wav16(l, r, g) {
  const n = l.length, buf = new ArrayBuffer(44 + n * 4), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + n * 4, true); w(8, "WAVEfmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
  v.setUint32(24, SR, true); v.setUint32(28, SR * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, n * 4, true);
  for (let i = 0, o = 44; i < n; i++, o += 4) {
    v.setInt16(o, Math.round(Math.max(-1, Math.min(1, l[i] * g)) * 32767), true);
    v.setInt16(o + 2, Math.round(Math.max(-1, Math.min(1, r[i] * g)) * 32767), true);
  }
  return new Blob([buf], { type: "audio/wav" });
}
const crcT = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
const crc32 = u => { let c = ~0; for (let i = 0; i < u.length; i++) c = crcT[(c ^ u[i]) & 255] ^ (c >>> 8); return ~c >>> 0; };
async function zipBlob(files) {
  const parts = [], cen = []; let off = 0, cs = 0;
  for (const f of files) {
    const nm = new TextEncoder().encode(f.name), d = new Uint8Array(await f.blob.arrayBuffer()), crc = crc32(d), sz = d.length;
    const h = new DataView(new ArrayBuffer(30)); h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint32(14, crc, true);
    h.setUint32(18, sz, true); h.setUint32(22, sz, true); h.setUint16(26, nm.length, true); parts.push(h.buffer, nm, d);
    const c = new DataView(new ArrayBuffer(46)); c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true);
    c.setUint32(16, crc, true); c.setUint32(20, sz, true); c.setUint32(24, sz, true); c.setUint16(28, nm.length, true); c.setUint32(42, off, true);
    cen.push(c.buffer, nm); off += 30 + nm.length + sz; cs += 46 + nm.length;
  }
  const e = new DataView(new ArrayBuffer(22)); e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true);
  e.setUint16(10, files.length, true); e.setUint32(12, cs, true); e.setUint32(16, off, true);
  return new Blob([...parts, ...cen, e.buffer], { type: "application/zip" });
}

async function go(file) {
  $("err").textContent = ""; $("drop").style.display = "none"; $("prog").style.display = "block";
  try {
    const p = await getProcessor();
    status("Reading audio...", 4); await tick();
    const [l, r] = await decode(file);
    if (l.length / SR > 900) throw new Error("File is longer than 15 minutes. Please use a shorter file (memory limit).");
    const res = await p.separate(l, r);
    status("Building Ambience and Music stems...", 88); await tick();
    const amb = extractAmbience(res.other.left, res.other.right, f => status("Building Ambience stem... " + (f * 100).toFixed(0) + "%", 88 + f * 8));
    const S = {
      vocal: [res.vocals.left, res.vocals.right], drum: [res.drums.left, res.drums.right], bass: [res.bass.left, res.bass.right],
      ambience: amb, other: [diff(res.other.left, amb[0]), diff(res.other.right, amb[1])],
      music: [sum(res.drums.left, res.bass.left, res.other.left), sum(res.drums.right, res.bass.right, res.other.right)]
    };
    let peak = 0; for (const [a, b] of Object.values(S)) for (let i = 0; i < a.length; i += 7) peak = Math.max(peak, Math.abs(a[i]), Math.abs(b[i]));
    const g = peak > 0.99 ? 0.99 / peak : 1, blobs = {};
    for (const k of Object.keys(NAMES)) { blobs[k] = wav16(S[k][0], S[k][1], g); await tick(); }
    status("Done", 100); show(blobs, file.name.replace(/\.[^.]+$/, ""));
  } catch (e) {
    console.error(e); $("err").textContent = e.message || String(e); $("prog").style.display = "none"; $("drop").style.display = "block";
  }
}
function show(blobs, base) {
  $("prog").style.display = "none"; $("res").style.display = "block";
  $("list").innerHTML = Object.keys(NAMES).map(k => { const u = URL.createObjectURL(blobs[k]);
    return `<div class="st" id="r-${k}"><h3>${NAMES[k]}</h3><audio controls preload="auto" src="${u}"></audio>
    <button class="g mu" data-k="${k}">Mute</button><a class="b g" href="${u}" download="${base}_${k}.wav">⬇ WAV</a></div>`; }).join("");
  const auds = [...document.querySelectorAll("audio")]; let playing = false;
  document.querySelectorAll(".mu").forEach(b => b.onclick = () => { const row = $("r-" + b.dataset.k), a = row.querySelector("audio");
    a.muted = !a.muted; b.textContent = a.muted ? "Unmute" : "Mute"; row.classList.toggle("m", a.muted); });
  $("pp").onclick = () => { if (!playing) { const t = auds[0].currentTime; auds.forEach(a => { a.currentTime = t; a.play(); }); } else auds.forEach(a => a.pause());
    playing = !playing; $("pp").textContent = playing ? "⏸ Pause all" : "▶ Play all"; };
  $("zip").onclick = async () => { $("zip").textContent = "Preparing zip..."; const z = await zipBlob(Object.keys(NAMES).map(k => ({ name: `${base}_${k}.wav`, blob: blobs[k] })));
    const a = document.createElement("a"); a.href = URL.createObjectURL(z); a.download = base + "_stems.zip"; a.click(); $("zip").textContent = "⬇ Download all (.zip)"; };
}
const drop = $("drop"), file = $("file");
drop.onclick = () => file.click(); file.onchange = () => file.files[0] && go(file.files[0]);
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); }; drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); e.dataTransfer.files[0] && go(e.dataTransfer.files[0]); };
