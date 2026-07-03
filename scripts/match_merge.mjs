// match_merge.mjs — une los out/match_part_*.json de los runners en un solo
// clips_<slug>_matched.json (listo para fetch_clips.mjs).
// Uso: node scripts/match_merge.mjs <slug> <partsDir>
import fs from "fs";
import path from "path";

const [slug, dir = "parts"] = process.argv.slice(2);
if (!slug) { console.error("Uso: node scripts/match_merge.mjs <slug> <partsDir>"); process.exit(1); }

const files = fs.readdirSync(dir).filter((f) => f.startsWith("match_part_") && f.endsWith(".json"));
const all = [];
for (const f of files) {
  try { all.push(...JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"))); }
  catch (e) { console.error("salteo", f, e.message); }
}
all.sort((a, b) => a.name.localeCompare(b.name));
const out = `clips_${slug}_matched.json`;
fs.writeFileSync(out, JSON.stringify(all, null, 2));
const dud = all.filter((b) => (b._score ?? 0) < 0.55).map((b) => b.name);
console.log(`merge: ${all.length} beats desde ${files.length} partes → ${out}`);
if (dud.length) console.log(`⚠ dudosos (score<0.55): ${dud.join(", ")}`);
