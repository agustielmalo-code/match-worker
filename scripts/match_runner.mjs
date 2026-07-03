// match_runner.mjs — versión FARM (cross-platform) del matcher CLIP. Corre en un runner
// de GitHub Actions: procesa un SHARD de beats (i % N === idx) del match list, puntúa con
// CLIP local y escribe out/match_part_<idx>.json. El agregador los une.
//
// v2 — misma lógica de búsqueda mejorada que matchclip.mjs (búsqueda ancha + re-rank por
// título, escaneo del video entero coarse→fine, penalización de texto graduada). Diferencia:
// usa `yt-dlp`/`ffmpeg` del PATH (no el .exe de bin/ ni el ffmpeg de Remotion).
//
// Uso:  node scripts/match_runner.mjs <slug> <idx> <total> [matchListPath]
import fs from "fs";
import path from "path";
import { spawnSync } from "child_process";
import { pipeline } from "@huggingface/transformers";

const [slug, idxArg, totalArg, listArg] = process.argv.slice(2);
if (!slug || idxArg == null || !totalArg) {
  console.error("Uso: node scripts/match_runner.mjs <slug> <idx> <total> [matchList]");
  process.exit(1);
}
const IDX = +idxArg, TOTAL = +totalArg;
const YTDLP = process.env.YTDLP || "yt-dlp";
const FF = process.env.FFMPEG || "ffmpeg";
const LIST = listArg || `match_${slug}.json`;
if (!fs.existsSync(LIST)) { console.error("No existe match list:", LIST); process.exit(1); }
const allBeats = JSON.parse(fs.readFileSync(LIST, "utf8").replace(/^﻿/, ""));
const beats = allBeats.filter((_, i) => i % TOTAL === IDX);
console.log(`shard ${IDX}/${TOTAL}: ${beats.length} beats`);

const CANDS = +(process.env.MATCH_CANDS || 6);
const POOL = +(process.env.MATCH_POOL || 16);
const MAXSPAN = +(process.env.MATCH_MAXSPAN || 360);
const COARSE = +(process.env.MATCH_COARSE || 44);
const FINE = +(process.env.MATCH_FINE || 0.5);
const HEAD = +(process.env.MATCH_HEAD || 6);
const TAIL = 0.06;
// Texto QUEMADO grande (carteles/títulos/subtítulos) se penaliza FUERTE en TODOS los nichos
// (ver scoreFrame). ★ MODO FAUNA (NO_WATERMARK=1) además rechaza DURO logos/marcas de agua,
// para que el documental se sienta UNA sola fuente (storytelling sin cortes).
const WM = process.env.NO_WATERMARK === "1";

const TMP = "_match_" + IDX; // temp PROPIO por shard → seguro para correr en paralelo (matchlocal.mjs)
fs.mkdirSync(TMP, { recursive: true });

// COOKIES de cuentas QUEMADAS (logueadas en YouTube) para esquivar el throttle. Rota por
// shard: cada uno usa una cuenta distinta de cookies/*.txt → el límite se reparte entre
// cuentas y se puede ir a full sin que YouTube marque bot. Sin carpeta cookies/ → como antes.
const cookieDir = process.env.COOKIE_DIR || "cookies";
let COOKIE = [];
try {
  const cf = fs.readdirSync(cookieDir).filter((f) => f.endsWith(".txt") && f !== "proxies.txt").sort();
  if (cf.length) { COOKIE = ["--cookies", path.join(cookieDir, cf[IDX % cf.length])]; console.log(`shard ${IDX}: cookies ${cf[IDX % cf.length]}`); }
} catch { /* sin cookies */ }
// PROXY DEDICADO al scraping (NO el de tus canales ni tu IP de casa). yt-dlp lo usa solo
// para esto. Se EMPAREJA con la cuenta: cookies/proxies.txt = 1 proxy por línea, MISMO orden
// que cookies/1.txt,2.txt… → cada cuenta sale por SU IP fija (sticky, ideal para no marcar).
// Fallback: YTPROXY (uno solo para todos). Vacío → IP local.
let PROXY = process.env.YTPROXY ? ["--proxy", process.env.YTPROXY] : [];
try {
  const px = fs.readFileSync(path.join(cookieDir, "proxies.txt"), "utf8").split(/\r?\n/).map((s) => s.trim()).filter((s) => s && !s.startsWith("#"));
  if (px.length) { PROXY = ["--proxy", px[IDX % px.length]]; console.log(`shard ${IDX}: proxy ${px[IDX % px.length].replace(/\/\/[^@]*@/, "//***@")}`); }
} catch { /* sin proxies.txt → YTPROXY o IP local */ }
const vidId = (u) => (u.match(/(?:v=|youtu\.be\/|shorts\/|embed\/)([A-Za-z0-9_-]{11})/) || u.match(/([A-Za-z0-9_-]{11})/) || [])[1];

const STOP = new Set(("a an the of to in on at by for with and or from into is are was were be " +
  "being this that it its as close up shot footage video clip hd 4k uhd full real best top").split(/\s+/));
const kw = (s) => [...new Set((s || "").toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/)
  .filter((w) => w.length > 2 && !STOP.has(w)))];
const BAD = /(reaction|react|podcast|full episode|interview|tier list|gameplay|let'?s play|trailer|unboxing|vlog|prank|lyric|lyrics|music video|official video|official music|\bsong\b|cover|remix|karaoke|instrumental|playlist|audio only|full album)/i;
// canales muy marcados: su footage casi siempre trae logo/bug → en fauna los penalizamos
const BRANDED = /(national geographic|nat ?geo|bbc|discovery|animal planet|pbs|smithsonian|earth ?touch)/i;

const searchRanked = (queries, concept, limit = CANDS) => {
  const want = new Set([...kw(concept), ...queries.flatMap(kw)]);
  const seen = new Map();
  for (const q of queries) {
    if (!q) continue;
    const r = spawnSync(YTDLP, [
      `ytsearch${POOL}:${q}`, "--skip-download", "--no-warnings", ...COOKIE, ...PROXY,
      "--socket-timeout", "30", "--match-filter", "duration>40 & duration<3000 & !is_live",
      "--print", "%(id)s\t%(title)s\t%(duration)s",
    ], { encoding: "utf8", maxBuffer: 1 << 26, timeout: 60000, killSignal: "SIGKILL" });
    const lines = (r.stdout || "").trim().split(/\r?\n/).filter(Boolean);
    if (!lines.length && process.env.MATCH_DEBUG) console.error(`  [dbg] q="${q}" status=${r.status} err=${r.error && r.error.message} stderr=${(r.stderr || "").slice(0, 200)}`);
    lines.forEach((line, ri) => {
      const [id, title = "", durS = ""] = line.split("\t");
      if (!id) return;
      if (seen.has(id)) { const e = seen.get(id); e.rank = Math.min(e.rank, ri); return; }
      const t = title.toLowerCase();
      const lex = [...want].reduce((n, w) => n + (t.includes(w) ? 1 : 0), 0);
      const pen = (BAD.test(t) ? 3 : 0) + (WM && BRANDED.test(t) ? 2 : 0);
      seen.set(id, { url: `https://youtu.be/${id}`, id, title, dur: +durS || 0, rank: ri, lex: lex - pen });
    });
  }
  return [...seen.values()].sort((a, b) => (b.lex - a.lex) || (a.rank - b.rank)).slice(0, limit);
};

const dlProxy = (url, a, b, tag) => {
  const proxy = path.join(TMP, `${tag}.mp4`);
  // --ffmpeg-location SOLO si FFMPEG es una ruta (local: ffmpeg de Remotion). En el farm
  // FFMPEG="ffmpeg" (en el PATH) → se omite. Sin esto, yt-dlp no puede cortar la sección.
  const ffloc = /[\\/]/.test(FF) ? ["--ffmpeg-location", path.dirname(FF)] : [];
  const dl = spawnSync(YTDLP, [
    url, ...COOKIE, ...PROXY, "--socket-timeout", "30", "--download-sections", `*${Math.max(0, a)}-${b}`, "--force-keyframes-at-cuts",
    "-f", "bv*[height<=240]/wv*/w", ...ffloc, "--merge-output-format", "mp4", "--no-playlist",
    "-o", proxy, "--force-overwrites", "--quiet", "--no-warnings",
  ], { encoding: "utf8", timeout: 120000, killSignal: "SIGKILL" });
  return (dl.status === 0 && fs.existsSync(proxy)) ? proxy : null;
};
const extract = (proxy, baseT, step, maxF, tag, ss = 0) => {
  const fdir = path.join(TMP, `${tag}_f`);
  fs.rmSync(fdir, { recursive: true, force: true });
  fs.mkdirSync(fdir, { recursive: true });
  const args = ["-y"];
  if (ss > 0) args.push("-ss", String(ss));
  args.push("-i", proxy, "-r", String(1 / step), "-frames:v", String(maxF), path.join(fdir, "f%04d.png"));
  spawnSync(FF, args, { encoding: "utf8", timeout: 90000, killSignal: "SIGKILL" });
  const files = fs.existsSync(fdir) ? fs.readdirSync(fdir).filter((f) => f.endsWith(".png")).sort() : [];
  return files.map((f, i) => ({ file: path.join(fdir, f), t: +(baseT + i * step).toFixed(1) }));
};

// Default = CPU q8 (idéntico a antes, no rompe runs en curso). Opt-in GPU/modelo grande:
//   CLIP_DEVICE=dml      → corre en la GPU (DirectML, sin CUDA). Sirve p/ modelos GRANDES.
//   CLIP_MODEL=...       → modelo distinto (ej. clip-vit-large) para MÁS calidad de match.
//   CLIP_DTYPE=fp32|fp16 → precisión (en GPU default fp32, en CPU default q8).
const CLIP_MODEL = process.env.CLIP_MODEL || "Xenova/clip-vit-base-patch32";
const CLIP_DEVICE = process.env.CLIP_DEVICE || "";
const CLIP_DTYPE = process.env.CLIP_DTYPE || (CLIP_DEVICE ? "fp32" : "q8");
console.log(`cargando CLIP (${CLIP_MODEL} · ${CLIP_DEVICE || "cpu"} · ${CLIP_DTYPE})...`);
const clf = await pipeline("zero-shot-image-classification", CLIP_MODEL, CLIP_DEVICE ? { device: CLIP_DEVICE, dtype: CLIP_DTYPE } : { dtype: CLIP_DTYPE });
const DISTRACTORS = [
  "a person talking to the camera, a talking head",
  "a blurry or unrelated indoor scene",
];
// dos preocupaciones SEPARADAS (misma llamada CLIP → sin recalibrar):
//   TEXT = texto QUEMADO grande (cartel/título/subtítulos) → FEO EN TODOS LOS NICHOS.
//   MARK = logo/marca de agua de canal → solo molesta en FAUNA (rompe el storytelling);
//          en misterio/otros las marcas se toleran (directiva del usuario).
const TEXT_LABELS = [
  "a title card or large bold text overlay across the screen",
  "big subtitles or captions text on screen",
  "karaoke style word-by-word captions with a colored highlight box",
  "a red YouTube subscribe button or social media UI overlay",
];
const MARK_LABELS = [
  "a channel logo, watermark or banner",
  "a semi-transparent tv channel logo bug in the corner",
];
// curva de penalización: score×1 si la confianza ≤LO, ×FLOOR si ≥HI, lineal en medio.
const curve = (x, LO, HI, FLOOR) => (x <= LO ? 1 : x >= HI ? FLOOR : 1 - ((x - LO) / (HI - LO)) * (1 - FLOOR));
const scoreFrame = async (file, concept) => {
  const out = await clf(file, [concept, ...DISTRACTORS, ...TEXT_LABELS, ...MARK_LABELS]);
  const get = (l) => out.find((o) => o.label === l)?.score || 0;
  // devolvemos CRUDO: concepto puro + confianza de texto + de marca. La penalización se
  // aplica DESPUÉS a nivel candidato (ranking), no acá, para no contaminar el _score.
  return { c: get(concept), text: Math.max(...TEXT_LABELS.map(get)), mark: Math.max(...MARK_LABELS.map(get)) };
};

const analyze = async (cand, beat, tag) => {
  const explicitWindow = beat.after != null || beat.scan != null;
  const a0 = explicitWindow ? (beat.after ?? 0) : HEAD;
  const dur = cand.dur || 0;
  const end = explicitWindow
    ? a0 + (beat.scan ?? 180)
    : Math.min(a0 + MAXSPAN, dur ? Math.floor(dur * (1 - TAIL)) : a0 + MAXSPAN);
  const span = Math.max(2, end - a0);
  const proxy = dlProxy(cand.url, a0, end, tag);
  if (!proxy) return null;
  const coarseStep = explicitWindow ? (beat.step || 2) : Math.max(2, +(span / COARSE).toFixed(2));
  const coarse = extract(proxy, a0, coarseStep, Math.ceil(span / coarseStep) + 2, tag + "_c");
  if (!coarse.length) return null;
  let bf = { c: -1, t: a0 };
  let texty = 0, total = 0, markHi = 0;
  for (const fr of coarse) { const r = await scoreFrame(fr.file, beat.concept); total++; if (r.text > 0.12) texty++; markHi = Math.max(markHi, r.mark); if (r.c > bf.c) bf = { c: r.c, t: fr.t }; }
  if (coarseStep > FINE) {
    const fa = Math.max(a0, +(bf.t - coarseStep).toFixed(2));
    const fb = Math.min(end, +(bf.t + coarseStep).toFixed(2));
    const fine = extract(proxy, fa, FINE, Math.ceil((fb - fa) / FINE) + 2, tag + "_x", fa - a0);
    for (const fr of fine) { const r = await scoreFrame(fr.file, beat.concept); total++; if (r.text > 0.12) texty++; markHi = Math.max(markHi, r.mark); if (r.c > bf.c) bf = { c: r.c, t: fr.t }; }
  }
  // TEXTO a nivel VENTANA: la fracción de frames con texto quemado RANKEA candidatos
  // (preferir el más limpio) SIN destruir el score de concepto → el filtro por _score sigue
  // siendo "qué tan on-topic" (no perdemos densidad). winPen SUAVE: full-texty ⇒ ×0.3 (demota).
  const frac = total ? texty / total : 0;
  const winPen = curve(frac, 0.25, 0.65, 0.3);
  const markPen = WM ? curve(markHi, 0.05, 0.14, 0.05) : 1; // logo/marca: solo fauna, en el RANK
  return { score: +bf.c.toFixed(4), rank: +(bf.c * winPen * markPen).toFixed(4), t: bf.t, url: cand.url, id: cand.id, text: +frac.toFixed(2) };
};

const results = [];
const usedIds = new Set();
for (const b of beats) {
  const { name, concept, dur = 6, lead = 1.0 } = b;
  let cands;
  if (b.urls) cands = b.urls.map((u) => ({ url: u, id: vidId(u), title: "", dur: 0 }));
  else cands = searchRanked(Array.isArray(b.query) ? b.query : [b.query], concept);
  if (!cands.length) { console.log(`✗ ${name}: sin candidatos`); continue; }
  const scored = [];
  for (let i = 0; i < cands.length; i++) {
    const res = await analyze(cands[i], b, `${name}_${i}`);
    if (res) scored.push(res);
  }
  if (!scored.length) { console.log(`✗ ${name}: sin frames`); continue; }
  // RANK = concepto × limpieza-de-texto (elige el candidato más limpio y on-topic).
  scored.sort((a, b2) => b2.rank - a.rank);
  let best = scored[0];
  const fresh = scored.find((c) => c.id && !usedIds.has(c.id) && c.rank >= scored[0].rank * 0.88);
  if (fresh) best = fresh;
  if (best.id) usedIds.add(best.id);
  const start = Math.max(0, +(best.t - lead).toFixed(1));
  console.log(`  ${name}: ${best.score.toFixed(3)} @ ${best.t}s (${best.id})${best.text > 0.3 ? ` ⚠txt${best.text}` : ""}`);
  // _score = concepto puro (para filtrar on-topic). _text = fracción de frames con texto.
  results.push({ name, url: best.url, start, dur, _score: +best.score.toFixed(3), _text: best.text });
}

fs.mkdirSync("out", { recursive: true });
const outPath = `out/match_part_${IDX}.json`;
fs.writeFileSync(outPath, JSON.stringify(results, null, 2));
console.log(`→ ${outPath} (${results.length} beats)`);
fs.rmSync(TMP, { recursive: true, force: true });
