// match_sb.mjs — matcher CLIP que puntúa con los STORYBOARDS de YouTube (NO baja video).
// Las IPs de GitHub Actions están bloqueadas para BAJAR video, pero NO para las imágenes
// de storyboard del CDN i.ytimg.com. Esto permite correr el match gratis y paralelo en la nube.
//
// Mecanismo (validado a mano):
//  1) yt-dlp -J <id> → info JSON. En .formats están los storyboards sb0(mejor) sb1 sb2 sb3.
//     Cada uno: columns, rows, fragments[].url, width/height = tamaño de UN tile. Arriba .duration.
//  2) total_tiles = columns*rows*fragments.length ; interval = duration/total_tiles.
//     tile global k (0-idx) → timestamp = (k+0.5)*interval.
//  3) bajar cada fragments[i].url (curl — CDN i.ytimg.com, NO bloqueado).
//  4) cada fragmento es grilla (cols*tileW) x (rows*tileH). Cortar con ffmpeg crop. Orden row-major;
//     índice global = fragIdx*(cols*rows) + row*cols + col.
//  5) puntuar cada tile con CLIP (transformers.js) contra el concept. Mejor tile → mejor video + ts.
//
// FALLBACK: si yt-dlp -J falla o no trae storyboards, usar frames públicos
//   https://i.ytimg.com/vi/<ID>/{1,2,3}.jpg (25/50/75%) → 3 tiles coarse. Siempre hay con qué puntuar.
//
// Uso:  node match_sb.mjs <slug> <shardIdx> <N> [listPath]  → out/match_part_<shardIdx>.json
// MISMO formato de salida que clips_<slug>_matched.json para que fetch_clips.mjs lo baje igual.
import fs from "fs";
import path from "path";
import { spawnSync } from "child_process";
import { pipeline } from "@huggingface/transformers";

// .env mínimo: carga YT_API_KEY (y cualquier VAR=valor) sin dependencia. process.env gana.
try {
  for (const line of fs.readFileSync(".env", "utf8").split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)$/i);
    if (m && process.env[m[1]] === undefined) process.env[m[1]] = m[2].trim().replace(/^["']|["']$/g, "");
  }
} catch { /* sin .env */ }
// MULTI-KEY: YT_API_KEY, YT_API_KEY2..N (las que existan). Round-robin por request + failover
// ante quotaExceeded (cuota diaria de una key agotada → siguiente key). 1 sola key = no rota.
const YT_API_KEYS = [
  process.env.YT_API_KEY, process.env.YT_API_KEY2, process.env.YT_API_KEY3, process.env.YT_API_KEY4,
  process.env.YT_API_KEY5, process.env.YT_API_KEY6,
].map((k) => (k || "").trim()).filter(Boolean);
const YT_API_KEY = YT_API_KEYS[0] || ""; // compat: el resto del código chequea "hay key?"

// yt-dlp / ffmpeg: como match_runner, via env YTDLP/FFMPEG o defaults. En local apuntá al
// .exe de bin/ y al ffmpeg de Remotion; en el farm quedan "yt-dlp"/"ffmpeg" del PATH.
const YTDLP = process.env.YTDLP || "yt-dlp";
const FF = process.env.FFMPEG || "ffmpeg";
const CANDS = +(process.env.MATCH_CANDS || 6);
const POOL = +(process.env.MATCH_POOL || 16);
const SB_MAXFRAGS = +(process.env.SB_MAXFRAGS || 24); // cap de fragmentos a bajar por candidato
// SB_NO_INFO=1 → simula la NUBE (GitHub): fuerza que ytInfo devuelva null (yt-dlp -J bloqueado),
// probando la rama de fallback API-duration + frames públicos 25/50/75%.
const SB_NO_INFO = process.env.SB_NO_INFO === "1";
// SB_ALL_QUERIES=1 → en modo API, usar TODAS las variantes de query (más candidatos, más cuota).
// Default: solo la 1ª variante por concepto (ahorra cuota; 150 conceptos × 1 key alcanza).
const SB_ALL_QUERIES = process.env.SB_ALL_QUERIES === "1";

// ── MODO REFINE: node match_sb.mjs --refine <clips_<slug>_matched.json> ──
// 2ª capa (LOCAL): sobre cada ganador ya elegido por la nube, corre el flujo COMPLETO de
// storyboard (-J, que en LOCAL sí anda) sobre ESE mismo video y actualiza `start` al segundo
// exacto (fino). La nube eligió el VIDEO; local afina el MOMENTO. Se maneja abajo (isRefine).
const argv = process.argv.slice(2);
const isRefine = argv[0] === "--refine";

let IDX = 0, TOTAL = 1, beats = [], refineList = null, refinePath = null;
if (isRefine) {
  refinePath = argv[1];
  if (!refinePath || !fs.existsSync(refinePath)) { console.error("Uso: node match_sb.mjs --refine <clips_<slug>_matched.json>"); process.exit(1); }
  refineList = JSON.parse(fs.readFileSync(refinePath, "utf8").replace(/^﻿/, ""));
  console.log(`refine: ${refineList.length} entradas de ${refinePath}`);
} else {
  const [slug, idxArg, totalArg, listArg] = argv;
  if (!slug || idxArg == null || !totalArg) {
    console.error("Uso: node match_sb.mjs <slug> <idx> <total> [matchList]  |  --refine <matched.json>");
    process.exit(1);
  }
  IDX = +idxArg; TOTAL = +totalArg;
  const LIST = listArg || `match_${slug}.json`;
  if (!fs.existsSync(LIST)) { console.error("No existe match list:", LIST); process.exit(1); }
  const allBeats = JSON.parse(fs.readFileSync(LIST, "utf8").replace(/^﻿/, ""));
  beats = allBeats.filter((_, i) => i % TOTAL === IDX);
  console.log(`shard ${IDX}/${TOTAL}: ${beats.length} beats`);
  console.log(`búsqueda: ${YT_API_KEY ? `YouTube Data API (${YT_API_KEYS.length} key${YT_API_KEYS.length > 1 ? "s" : ""}, retry+rotación, ${SB_ALL_QUERIES ? "todas las queries" : "1 query/concepto"})` : "yt-dlp ytsearch (sin YT_API_KEY)"}`);
}

const TMP = "_sb_" + (isRefine ? "refine" : IDX); // temp PROPIO por shard → seguro en paralelo
fs.rmSync(TMP, { recursive: true, force: true });
fs.mkdirSync(TMP, { recursive: true });

// COOKIES (cuentas logueadas) para esquivar throttle en los yt-dlp -J / search. Rota por shard.
// OJO: acá NO usamos proxies (solo search+metadata, y las imgs son del CDN abierto).
const cookieDir = process.env.COOKIE_DIR || "cookies";
let COOKIE = [];
try {
  const cf = fs.readdirSync(cookieDir).filter((f) => f.endsWith(".txt") && f !== "proxies.txt").sort();
  if (cf.length) { COOKIE = ["--cookies", path.join(cookieDir, cf[IDX % cf.length])]; console.log(`shard ${IDX}: cookies ${cf[IDX % cf.length]}`); }
} catch { /* sin cookies */ }

const vidId = (u) => (u.match(/(?:v=|youtu\.be\/|shorts\/|embed\/)([A-Za-z0-9_-]{11})/) || u.match(/([A-Za-z0-9_-]{11})/) || [])[1];

// ── BÚSQUEDA (copiada de match_runner.mjs): ytsearch ancha + re-rank por título + filtro BAD ──
const STOP = new Set(("a an the of to in on at by for with and or from into is are was were be " +
  "being this that it its as close up shot footage video clip hd 4k uhd full real best top").split(/\s+/));
const kw = (s) => [...new Set((s || "").toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/)
  .filter((w) => w.length > 2 && !STOP.has(w)))];
const BAD = /(reaction|react|podcast|full episode|interview|tier list|gameplay|let'?s play|trailer|unboxing|vlog|prank|lyric|lyrics|music video|official video|official music|\bsong\b|cover|remix|karaoke|instrumental|playlist|audio only|full album)/i;

// re-rank compartido (mismo criterio léxico + BAD para AMBAS fuentes de búsqueda). Recibe
// filas {id,title,dur} en orden de relevancia de la fuente → dedup + rankea + corta a limit.
const rerank = (rows, queries, concept, limit) => {
  const want = new Set([...kw(concept), ...queries.flatMap(kw)]);
  const seen = new Map();
  rows.forEach(({ id, title = "", dur = 0 }, ri) => {
    if (!id) return;
    if (seen.has(id)) { const e = seen.get(id); e.rank = Math.min(e.rank, ri); return; }
    const t = title.toLowerCase();
    const lex = [...want].reduce((n, w) => n + (t.includes(w) ? 1 : 0), 0);
    const pen = BAD.test(t) ? 3 : 0;
    seen.set(id, { url: `https://youtu.be/${id}`, id, title, dur: +dur || 0, rank: ri, lex: lex - pen });
  });
  return [...seen.values()].sort((a, b) => (b.lex - a.lex) || (a.rank - b.rank)).slice(0, limit);
};

// ── HTTP a la YouTube Data API con RETRY + MULTI-KEY + failover ──
// Problema real observado en la nube: 20 shards contra 1 key → 429 rateLimitExceeded (límite POR
// SEGUNDO, no cuota diaria) → caían al fallback ytsearch (bloqueado en GitHub) → 25/150.
// Estrategia:
//  · rateLimitExceeded (429): TRANSITORIO → backoff exponencial + jitter (1s,2s,4s,8s) y reintentar;
//    también probamos con OTRA key por si esa está caliente.
//  · quotaExceeded (403): cuota DIARIA de esa key agotada → marcar key muerta y pasar a la siguiente.
//  · jitter corto entre llamadas para no ametrallar.
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const jitter = (base, spread) => base + Math.floor(Math.random() * spread);
let keyPtr = ((+process.env.SHARD_IDX || IDX || 0)) % Math.max(1, YT_API_KEYS.length); // arranca distinto por shard
const deadKeys = new Set(); // keys con quotaExceeded (cuota diaria) → no reusar en este run
const nextKey = () => {
  for (let i = 0; i < YT_API_KEYS.length; i++) {
    const k = YT_API_KEYS[keyPtr % YT_API_KEYS.length];
    keyPtr++;
    if (!deadKeys.has(k)) return k;
  }
  return null; // todas muertas
};
const kTag = (k) => "…" + (k || "").slice(-4);

// GET con reintentos. Devuelve el JSON parseado (con .items) o null si se agotó todo.
const apiGet = async (buildUrl, ctx) => {
  const MAX = 4;
  for (let attempt = 0; attempt < MAX; attempt++) {
    const key = nextKey();
    if (!key) { console.error(`  [API] ${ctx}: sin keys vivas (todas quotaExceeded)`); return null; }
    try {
      await sleep(jitter(150, 250)); // ~150-400ms anti-ráfaga
      const res = await fetch(buildUrl(key), { signal: AbortSignal.timeout(30000) });
      const j = await res.json().catch(() => ({}));
      if (!j.error) return j;
      const reason = j.error.errors?.[0]?.reason || j.error.status || j.error.code;
      if (reason === "quotaExceeded" || reason === "dailyLimitExceeded") {
        deadKeys.add(key);
        console.error(`  [API] ${ctx}: quota agotada en key ${kTag(key)} → rotando`);
        continue; // NO cuenta como retry-con-espera: probamos otra key ya
      }
      if (res.status === 429 || reason === "rateLimitExceeded" || reason === "userRateLimitExceeded") {
        const wait = jitter(1000 * 2 ** attempt, 500); // 1s,2s,4s,8s (+jitter)
        console.error(`  [API] ${ctx}: 429 ${reason} key ${kTag(key)} → backoff ${wait}ms (intento ${attempt + 1}/${MAX})`);
        await sleep(wait);
        continue;
      }
      console.error(`  [API] ${ctx}: error ${res.status} ${reason} → fallback`);
      return null; // error no recuperable (400 badRequest, keyInvalid, etc.)
    } catch (e) {
      const wait = jitter(1000 * 2 ** attempt, 500);
      console.error(`  [API] ${ctx}: fetch falló (${e.message}) → backoff ${wait}ms (intento ${attempt + 1}/${MAX})`);
      await sleep(wait);
    }
  }
  console.error(`  [API] ${ctx}: agotados ${MAX} reintentos → fallback`);
  return null;
};

// CACHE de búsquedas POR PROCESO (por shard). search.list cuesta 100 unidades (10k/día = 100
// búsquedas) → memoizar es clave. Si varios beats comparten la MISMA query (ej N ángulos del mismo
// dulce con igual query[0]) se hace UNA sola llamada y todos reusan el pool; el dedup usedIds
// reparte videos/momentos distintos del pool (estilo docextract). No se persiste a disco.
const searchCache = new Map(); // queryNorm → filas [] (o [] si falló, para no repegar)
const qNorm = (q) => (q || "").trim().toLowerCase().replace(/\s+/g, " ");

// FUENTE A: YouTube Data API oficial (con retry/rotación + cache). Una llamada por variante de
// query, items en orden de relevancia. Devuelve filas {id,title} o null (para caer a ytsearch).
const apiSearch = async (q) => {
  const key = qNorm(q);
  if (searchCache.has(key)) {
    const cached = searchCache.get(key);
    console.log(`  [API] cache HIT q="${q}" (${cached.length} cands, 0 unidades)`);
    return cached;
  }
  const j = await apiGet(
    (k) => `https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=25`
      + `&relevanceLanguage=es&q=${encodeURIComponent(q)}&key=${k}`,
    `search q="${q}"`,
  );
  if (!j) return null; // error → NO cachear (que reintente / caiga a ytsearch en el próximo beat)
  const rows = (j.items || []).map((it) => ({ id: it.id?.videoId, title: it.snippet?.title || "", dur: 0 }))
    .filter((r) => r.id);
  searchCache.set(key, rows);
  return rows;
};

// FUENTE B (fallback): yt-dlp ytsearch (rate-limitable). Devuelve filas {id,title,dur}.
const ytsearch = (q) => {
  const r = spawnSync(YTDLP, [
    `ytsearch${POOL}:${q}`, "--skip-download", "--no-warnings", ...COOKIE,
    "--socket-timeout", "30", "--match-filter", "duration>40 & duration<3000 & !is_live",
    "--print", "%(id)s\t%(title)s\t%(duration)s",
  ], { encoding: "utf8", maxBuffer: 1 << 26, timeout: 60000, killSignal: "SIGKILL" });
  const lines = (r.stdout || "").trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length && process.env.MATCH_DEBUG) console.error(`  [dbg] ytsearch q="${q}" status=${r.status} err=${r.error && r.error.message} stderr=${(r.stderr || "").slice(0, 200)}`);
  return lines.map((line) => { const [id, title = "", durS = ""] = line.split("\t"); return { id, title, dur: +durS || 0 }; });
};

const searchRanked = async (queries, concept, limit = CANDS) => {
  // En modo API usamos SOLO la 1ª variante por defecto (ahorra cuota; 150 conceptos × 1 key
  // alcanza). SB_ALL_QUERIES=1 vuelve a todas. Sin key (fallback ytsearch) sí usa todas.
  const apiQueries = (YT_API_KEY && !SB_ALL_QUERIES) ? queries.slice(0, 1) : queries;
  const rows = [];
  for (const q of apiQueries) {
    if (!q) continue;
    let got = null;
    if (YT_API_KEY) got = await apiSearch(q); // API oficial (con retry/rotación) primero
    if (got == null) got = ytsearch(q);        // fallback: sin-key / error no recuperable → ytsearch
    rows.push(...got);
  }
  return rerank(rows, apiQueries, concept, limit);
};

// ── STORYBOARD: metadata + descarga de fragmentos + corte de tiles ──
// yt-dlp -J acá va SIN cookies a propósito: las cuentas logueadas suelen estar rate-limiteadas
// ("This content isn't available, try again later"), pero la metadata pública (incl. storyboards
// + duration) sí sale sin login. Con cookie fallaba → todo caía al fallback con dur=0 (@0s).
// La búsqueda ya usa la API oficial, así que las cookies no aportan nada aquí.
const ytInfo = (id) => {
  if (SB_NO_INFO) return null; // simula la nube: -J bloqueado
  const r = spawnSync(YTDLP, [
    `https://youtu.be/${id}`, "-J", "--skip-download", "--no-warnings",
    "--socket-timeout", "30",
  ], { encoding: "utf8", maxBuffer: 1 << 28, timeout: 60000, killSignal: "SIGKILL" });
  if (r.status !== 0 || !r.stdout) return null;
  try { return JSON.parse(r.stdout); } catch { return null; }
};

// ISO-8601 de YouTube (contentDetails.duration, ej "PT1H4M30S") → segundos.
const iso8601ToSec = (s) => {
  const m = /^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$/.exec(s || "");
  if (!m) return 0;
  return (+m[1] || 0) * 3600 + (+m[2] || 0) * 60 + (+m[3] || 0);
};
// DURACIÓN vía YouTube Data API (batch hasta 50 ids = 1 unidad de cuota). Devuelve Map id→segundos.
// Es lo que salva a la NUBE: -J bloqueado ⇒ sin duration ⇒ fallback caía a @0s. Con esto el
// fallback mapea 25/50/75% a segundos REALES. {} si no hay key o error (loguea reason).
const apiDurations = async (ids) => {
  const out = new Map();
  if (!YT_API_KEY || !ids.length) return out;
  for (let i = 0; i < ids.length; i += 50) {
    const batch = ids.slice(i, i + 50);
    // mismo apiGet → hereda retry/backoff en 429 y rotación de key en quotaExceeded.
    const j = await apiGet(
      (key) => `https://www.googleapis.com/youtube/v3/videos?part=contentDetails&id=${batch.join(",")}&key=${key}`,
      `videos.list (${batch.length} ids)`,
    );
    if (!j) continue;
    for (const it of (j.items || [])) out.set(it.id, iso8601ToSec(it.contentDetails?.duration));
  }
  return out;
};

// devuelve el mejor storyboard (sb0 preferido, luego más tiles). null si no hay.
const pickStoryboard = (info) => {
  const sbs = (info.formats || []).filter((f) => String(f.format_id || "").startsWith("sb") && (f.fragments || []).length && f.columns && f.rows);
  if (!sbs.length) return null;
  // preferir sb0 (mejor resolución de tile); si no, el de más tiles totales
  sbs.sort((a, b) => {
    if (a.format_id === "sb0") return -1;
    if (b.format_id === "sb0") return 1;
    return (b.columns * b.rows * b.fragments.length) - (a.columns * a.rows * a.fragments.length);
  });
  return sbs[0];
};

const curl = (url, dest) => {
  const r = spawnSync("curl", ["-s", "-L", "--max-time", "30", url, "-o", dest], { encoding: "utf8", timeout: 40000 });
  return r.status === 0 && fs.existsSync(dest) && fs.statSync(dest).size > 500;
};

// corta un fragmento (grilla cols*rows) en tiles → array de {file, k}. tag único por fragmento.
const cropTiles = (fragFile, cols, rows, tileW, tileH, fragIdx, tag) => {
  const out = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const k = fragIdx * (cols * rows) + row * cols + col;
      const file = path.join(TMP, `${tag}_t${k}.jpg`);
      const r = spawnSync(FF, ["-y", "-hide_banner", "-loglevel", "error", "-i", fragFile,
        "-vf", `crop=${tileW}:${tileH}:${col * tileW}:${row * tileH}`, file],
        { encoding: "utf8", timeout: 30000 });
      if (r.status === 0 && fs.existsSync(file)) out.push({ file, k });
    }
  }
  return out;
};

// genera todos los tiles candidatos de un video via storyboard. [] si falla.
const storyboardTiles = (info, id, tag) => {
  const sb = pickStoryboard(info);
  if (!sb) return [];
  const duration = info.duration || sb.duration || 0;
  if (!duration) return [];
  const cols = sb.columns, rows = sb.rows;
  const frags = sb.fragments.slice(0, SB_MAXFRAGS);
  const totalTiles = cols * rows * sb.fragments.length; // usar el TOTAL real para el interval
  const interval = duration / totalTiles;
  const tileW = sb.width, tileH = sb.height;
  const tiles = [];
  for (let fi = 0; fi < frags.length; fi++) {
    const frag = path.join(TMP, `${tag}_f${fi}.jpg`);
    if (!curl(frags[fi].url, frag)) continue;
    const ts = cropTiles(frag, cols, rows, tileW, tileH, fi, tag);
    for (const t of ts) tiles.push({ file: t.file, t: +((t.k + 0.5) * interval).toFixed(1) });
    fs.rmSync(frag, { force: true });
  }
  return tiles;
};

// FALLBACK: frames públicos /vi/<ID>/{1,2,3}.jpg (25/50/75%). Siempre accesibles.
const fallbackTiles = (id, duration, tag) => {
  const tiles = [];
  const fracs = [[1, 0.25], [2, 0.5], [3, 0.75]];
  for (const [n, fr] of fracs) {
    const dest = path.join(TMP, `${tag}_vi${n}.jpg`);
    if (curl(`https://i.ytimg.com/vi/${id}/${n}.jpg`, dest)) {
      tiles.push({ file: dest, t: +((duration || 0) * fr).toFixed(1) });
    }
  }
  return tiles;
};

// ── CLIP (copiado de match_runner.mjs) ──
const CLIP_MODEL = process.env.CLIP_MODEL || "Xenova/clip-vit-base-patch32";
const CLIP_DEVICE = process.env.CLIP_DEVICE || "";
const CLIP_DTYPE = process.env.CLIP_DTYPE || (CLIP_DEVICE ? "fp32" : "q8");
console.log(`cargando CLIP (${CLIP_MODEL} · ${CLIP_DEVICE || "cpu"} · ${CLIP_DTYPE})...`);
const clf = await pipeline("zero-shot-image-classification", CLIP_MODEL, CLIP_DEVICE ? { device: CLIP_DEVICE, dtype: CLIP_DTYPE } : { dtype: CLIP_DTYPE });
const DISTRACTORS = [
  "a person talking to the camera, a talking head",
  "a blurry or unrelated indoor scene",
];
// TEXT_LABELS: texto QUEMADO grande / end-cards / botón de suscribir / UI social. En storyboards
// esto es CRÍTICO: los tiles caen a menudo en la pantalla final (grilla "SUSCRIBITE" + iconos)
// o carátulas de título, que CLIP (con pocos distractores) puntúa alto para el concepto. Los
// metemos como labels COMPETIDORES y penalizamos el tile por su confianza de texto → así el
// mejor tile es una toma limpia, no un end-screen. (Misma idea que scoreFrame de match_runner.)
const TEXT_LABELS = [
  "a title card or large bold text overlay across the screen",
  "big subtitles or captions text on screen",
  "a red YouTube subscribe button or social media UI overlay",
  "a grid collage of several small thumbnails, a video end screen",
];
// curva: score×1 si confianza de texto ≤LO, ×FLOOR si ≥HI, lineal en medio.
const curve = (x, LO, HI, FLOOR) => (x <= LO ? 1 : x >= HI ? FLOOR : 1 - ((x - LO) / (HI - LO)) * (1 - FLOOR));
const scoreTile = async (file, concept) => {
  const out = await clf(file, [concept, ...DISTRACTORS, ...TEXT_LABELS]);
  const get = (l) => out.find((o) => o.label === l)?.score || 0;
  const c = get(concept);
  const textHi = Math.max(...TEXT_LABELS.map(get));
  // penaliza tiles con texto/end-card quemado: textHi≥0.35 ⇒ ×0.15 (los demota fuerte).
  return { c, rank: c * curve(textHi, 0.12, 0.35, 0.15), text: textHi };
};

// Puntúa TODOS los tiles de un video (storyboard si hay -J; si no, fallback con duración de API)
// y devuelve el mejor {score, rank, t, text} o null. `apiDur` = duración por API (segundos) para
// el fallback (clave para que la NUBE dé un timestamp coarse REAL y no @0s).
const scoreVideo = async (id, concept, tag, apiDur = 0) => {
  const info = ytInfo(id);
  let tiles = [];
  let dur0 = apiDur || 0;
  if (info) {
    dur0 = info.duration || dur0;
    tiles = storyboardTiles(info, id, tag);
  }
  if (!tiles.length) tiles = fallbackTiles(id, dur0, tag); // NUBE: 25/50/75% × dur(API) → coarse real
  if (!tiles.length) return null;
  let bf = { c: -1, rank: -1, t: 0, text: 0 };
  for (const tl of tiles) {
    const r = await scoreTile(tl.file, concept);
    if (r.rank > bf.rank) bf = { c: r.c, rank: r.rank, t: tl.t, text: r.text };
  }
  for (const tl of tiles) fs.rmSync(tl.file, { force: true });
  return { score: +bf.c.toFixed(4), rank: +bf.rank.toFixed(4), t: bf.t, text: +bf.text.toFixed(2), fine: !!info };
};

// ── MODO REFINE: afina el `start` de cada ganador con el storyboard FINO (local) ──
if (isRefine) {
  let refined = 0;
  for (const e of refineList) {
    const id = vidId(e.url || "");
    if (!id) { console.log(`  ✗ ${e.name}: sin id`); continue; }
    const r = await scoreVideo(id, e.concept || "", `refine_${id}`);
    if (!r) { console.log(`  ✗ ${e.name}: sin tiles (${id})`); continue; }
    const lead = e.lead || 0;
    const newStart = Math.max(0, +(r.t - lead).toFixed(1));
    const old = e.start;
    e.start = newStart;
    if (r.fine) e._sb = true; // afinado con storyboard real
    refined++;
    console.log(`  ${e.name}: ${old}s → ${newStart}s  (${r.fine ? "fino" : "coarse"} ${r.score.toFixed(3)}, ${id})`);
  }
  fs.writeFileSync(refinePath, JSON.stringify(refineList, null, 2));
  console.log(`→ ${refinePath} (${refined}/${refineList.length} afinados)`);
  fs.rmSync(TMP, { recursive: true, force: true });
  process.exit(0);
}

// ── loop principal ──
const results = [];
const usedIds = new Set();
for (const b of beats) {
  const { name, concept, dur = 5, lead = 0 } = b;
  let cands;
  if (b.urls) cands = b.urls.map((u) => ({ url: u, id: vidId(u), title: "", dur: 0 }));
  else cands = await searchRanked(Array.isArray(b.query) ? b.query : [b.query], concept);
  if (!cands.length) { console.log(`✗ ${name}: sin candidatos`); continue; }

  // Prefetch de DURACIONES por API (1 unidad de cuota por lote de 50) → el fallback de la NUBE
  // mapea 25/50/75% a segundos reales. Barato y evita @0s cuando -J está bloqueado.
  const durMap = await apiDurations([...new Set(cands.map((c) => c.id).filter(Boolean))]);

  const scored = [];
  for (let i = 0; i < cands.length; i++) {
    const cand = cands[i];
    if (!cand.id) continue;
    const apiDur = durMap.get(cand.id) || cand.dur || 0;
    const r = await scoreVideo(cand.id, concept, `${name}_${i}`, apiDur);
    if (!r) { if (process.env.MATCH_DEBUG) console.error(`  [dbg] ${name} cand ${cand.id}: 0 tiles`); continue; }
    scored.push({ ...r, url: cand.url, id: cand.id });
  }
  if (!scored.length) { console.log(`✗ ${name}: sin tiles`); continue; }
  scored.sort((a, b2) => b2.rank - a.rank);
  let best = scored[0];
  // preferir un video FRESCO (no reusado) si está dentro del 88% del mejor rank
  const fresh = scored.find((c) => c.id && !usedIds.has(c.id) && c.rank >= scored[0].rank * 0.88);
  if (fresh) best = fresh;
  if (best.id) usedIds.add(best.id);
  const start = Math.max(0, +(best.t - lead).toFixed(1));
  console.log(`  ${name}: ${best.score.toFixed(3)} @ ${best.t}s (${best.id})${best.fine ? "" : " ~coarse"}${best.text > 0.3 ? ` ⚠txt${best.text}` : ""}`);
  results.push({ name, concept, url: best.url, start, dur, _score: +best.score.toFixed(3), _sb: true });
}

fs.mkdirSync("out", { recursive: true });
const outPath = `out/match_part_${IDX}.json`;
fs.writeFileSync(outPath, JSON.stringify(results, null, 2));
console.log(`→ ${outPath} (${results.length} beats)`);
fs.rmSync(TMP, { recursive: true, force: true });
