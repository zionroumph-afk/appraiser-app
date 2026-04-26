// server.mjs — Creator's Companion for "Death Before The Light"
// Dual-Model Architecture: Qwen 2.5 3B (creative) + DeepSeek R1 1.5B (logic/lore)

import express  from "express";
import fs       from "fs";
import path     from "path";
import { fileURLToPath } from "url";
import fetch    from "node-fetch";
import FormData from "form-data";
import crypto   from "crypto";
import { getLlama, LlamaChatSession } from "node-llama-cpp";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app  = express();
const PORT = 3000;
const VISION_URL = "http://127.0.0.1:5001/describe";

app.use(express.json({ limit: "50mb" }));
app.use(express.static(__dirname));

// ─── STORAGE ─────────────────────────────────────────────────────
const DATA_DIR    = path.join(__dirname, "data");
const CHATS_DIR   = path.join(DATA_DIR,  "chats");
const TITLES_FILE = path.join(DATA_DIR,  "titles.json");
const MEMORY_FILE = path.join(DATA_DIR,  "memory.json");
const EXPORTS_DIR = path.join(DATA_DIR,  "exports");
const CACHE_DIR   = path.join(DATA_DIR,  "cache");

for (const dir of [DATA_DIR, CHATS_DIR, EXPORTS_DIR, CACHE_DIR]) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

// ─── DUAL-MODEL SETUP ────────────────────────────────────────────
// Model A: Qwen 2.5 3B — dialogue, persona, creative writing
// Model B: DeepSeek R1 1.5B — lore logic, verification, summarization

let llama;
let creativeModel,  creativeContext;   // Qwen 2.5 3B
let logicModel,     logicContext;      // DeepSeek R1 1.5B

const CREATIVE_MODEL_PATH = path.join(__dirname, "brain", "qwen2.5-3b.gguf");
const LOGIC_MODEL_PATH    = path.join(__dirname, "brain", "deepseek-r1-1.5b.gguf");

async function init() {
    console.log("⏳ Initializing dual-model architecture...");
    llama = await getLlama();

    // Load Creative Core (Qwen 2.5 3B)
    if (fs.existsSync(CREATIVE_MODEL_PATH)) {
        console.log("🎭 Loading Creative Core (Qwen 2.5 3B)...");
        creativeModel   = await llama.loadModel({ modelPath: CREATIVE_MODEL_PATH });
        creativeContext = await creativeModel.createContext({ contextSize: 4096 });
        console.log("✅ Creative Core ready!");
    } else {
        console.warn(`⚠️  Creative model not found at: ${CREATIVE_MODEL_PATH}`);
        console.warn("   Place Qwen2.5-3B-Instruct Q4_K_M GGUF as brain/qwen2.5-3b.gguf");
    }

    // Load Logic Layer (DeepSeek R1 1.5B)
    if (fs.existsSync(LOGIC_MODEL_PATH)) {
        console.log("🧠 Loading Logic Layer (DeepSeek R1 1.5B)...");
        logicModel   = await llama.loadModel({ modelPath: LOGIC_MODEL_PATH });
        logicContext = await logicModel.createContext({ contextSize: 2048 });
        console.log("✅ Logic Layer ready!");
    } else {
        console.warn(`⚠️  Logic model not found at: ${LOGIC_MODEL_PATH}`);
        console.warn("   Place DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M GGUF as brain/deepseek-r1-1.5b.gguf");
    }

    if (!creativeModel && !logicModel) {
        throw new Error("No models found. See warnings above for file paths.");
    }

    console.log("⚔️  Death Before The Light — Dual-Model Companion ready!");
}

// ─── SESSION FACTORIES ───────────────────────────────────────────
async function createCreativeSession() {
    const ctx = creativeModel ? creativeContext : logicContext;
    const mod = creativeModel || logicModel;
    if (!mod) throw new Error("No model loaded");
    try {
        return new LlamaChatSession({ contextSequence: ctx.getSequence() });
    } catch {
        console.log("♻️  Recreating creative context...");
        if (creativeModel) {
            creativeContext = await creativeModel.createContext({ contextSize: 4096 });
            return new LlamaChatSession({ contextSequence: creativeContext.getSequence() });
        } else {
            logicContext = await logicModel.createContext({ contextSize: 2048 });
            return new LlamaChatSession({ contextSequence: logicContext.getSequence() });
        }
    }
}

async function createLogicSession() {
    const ctx = logicContext ? logicContext : creativeContext;
    const mod = logicModel || creativeModel;
    if (!mod) throw new Error("No model loaded");
    try {
        return new LlamaChatSession({ contextSequence: ctx.getSequence() });
    } catch {
        console.log("♻️  Recreating logic context...");
        if (logicModel) {
            logicContext = await logicModel.createContext({ contextSize: 2048 });
            return new LlamaChatSession({ contextSequence: logicContext.getSequence() });
        } else {
            creativeContext = await creativeModel.createContext({ contextSize: 4096 });
            return new LlamaChatSession({ contextSequence: creativeContext.getSequence() });
        }
    }
}

// ─── HELPERS ─────────────────────────────────────────────────────
const nowIso    = () => new Date().toISOString();
const createId  = (p = "mem") => `${p}_${crypto.randomBytes(8).toString("hex")}`;
const normalize = (s = "") => s.toLowerCase().replace(/[^a-z0-9\s]/gi, " ").replace(/\s+/g, " ").trim();

function tokenize(text = "") {
    const STOP_WORDS = new Set(["the","a","an","is","are","was","were","and","or","but","in","on","at","to","for","of","with","that","this","it","its","be","been"]);
    return normalize(text).split(" ").filter(t => t.length > 2 && !STOP_WORDS.has(t));
}
function toTitleCase(v = "") {
    return String(v).trim().split(/\s+/).filter(Boolean)
        .map(s => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase()).join(" ");
}
function jaccardScore(a, b) {
    if (!a.length || !b.length) return 0;
    const sa = new Set(a), sb = new Set(b);
    let inter = 0;
    for (const t of sa) if (sb.has(t)) inter++;
    const union = sa.size + sb.size - inter;
    return union > 0 ? inter / union : 0;
}
function safeJsonRead(file, fallback) {
    try { if (!fs.existsSync(file)) return fallback; return JSON.parse(fs.readFileSync(file, "utf8")); }
    catch { return fallback; }
}
function safeJsonWrite(file, value) { fs.writeFileSync(file, JSON.stringify(value, null, 2)); }

// ─── RESPONSE CACHE (DeepSeek logic results) ─────────────────────
const cacheMap = new Map();
function getCacheKey(prompt) { return crypto.createHash("md5").update(prompt).digest("hex"); }
function getCached(key) {
    const hit = cacheMap.get(key);
    if (!hit) return null;
    if (Date.now() - hit.ts > 3600000) { cacheMap.delete(key); return null; } // 1hr TTL
    return hit.value;
}
function setCache(key, value) {
    cacheMap.set(key, { value, ts: Date.now() });
    if (cacheMap.size > 200) {
        const oldest = [...cacheMap.entries()].sort((a,b) => a[1].ts - b[1].ts)[0];
        cacheMap.delete(oldest[0]);
    }
}

// ─── VISION ──────────────────────────────────────────────────────
async function describeImage(base64, name = "image.png") {
    try {
        const raw    = base64.includes(",") ? base64.split(",")[1] : base64;
        const buffer = Buffer.from(raw, "base64");
        const form   = new FormData();
        form.append("image", buffer, { filename: name, contentType: "image/png" });
        const res  = await fetch(VISION_URL, { method: "POST", body: form, headers: form.getHeaders(), timeout: 60000 });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || "Vision failed");
        return data.description || "No description returned.";
    } catch (err) {
        if (err.code === "ECONNREFUSED") return "[Vision server offline — start vision_server.py]";
        return `[Image analysis failed: ${err.message}]`;
    }
}

// ─── IMAGE GENERATION ────────────────────────────────────────────
async function generateImage(prompt) {
    const enhanced = `dark fantasy rpg, ${prompt}, dramatic lighting, highly detailed, concept art`;
    const encoded  = encodeURIComponent(enhanced);
    const seed     = Math.floor(Math.random() * 999999);
    const url      = `https://image.pollinations.ai/prompt/${encoded}?width=512&height=512&seed=${seed}&model=flux&nologo=true`;

    const imgRes = await fetch(url, {
        headers: { "Referer": "https://pollinations.ai", "User-Agent": "CreatorsCompanion/1.0" }
    });
    if (!imgRes.ok) throw new Error(`Pollinations returned ${imgRes.status}`);
    const buffer      = await imgRes.buffer();
    const contentType = imgRes.headers.get("content-type") || "image/jpeg";
    return { url, b64: `data:${contentType};base64,${buffer.toString("base64")}` };
}

// ─── MEMORY CORE ─────────────────────────────────────────────────
const DEFAULT_GAME = {
    title: "Death Before The Light",
    characters: [], factions: [], locations: [],
    lore: [], plot: [], items: [], sessions: []
};
const DEFAULT_PREFERENCES = {
    response_length: "natural",
    story_mode: false,
    dm_perspective: "second",
    active_model: "creative"   // "creative" | "logic" | "auto"
};

function createBaseMemory() {
    return {
        schema_version: 3,
        game: { ...DEFAULT_GAME },
        preferences: { ...DEFAULT_PREFERENCES },
        project: { notes: [], pins: [] },
        memory_entries: [],
        lore_index: {}   // fast lookup: normalized_key -> entry_id[]
    };
}

function toEntry({ kind, text, entities = [], importance = 3, confidence = 0.8, source = "user", scope = {} }) {
    return {
        id: createId("mem"), kind, text: String(text || "").trim(), entities,
        scope: {
            world_id:    scope.world_id    || "death-before-the-light",
            player_id:   scope.player_id   || "creator",
            campaign_id: scope.campaign_id || "main"
        },
        importance:  Math.max(1, Math.min(5, Number(importance) || 3)),
        confidence:  Math.max(0, Math.min(1, Number(confidence) || 0.8)),
        source,
        created_at:      nowIso(),
        last_used_at:    nowIso(),
        invalidated_at:  null,
        superseded_by:   null,
        expires_at:      null
    };
}

function isActiveEntry(e) {
    if (!e || e.invalidated_at) return false;
    if (e.expires_at && new Date(e.expires_at).getTime() < Date.now()) return false;
    return true;
}

// ─── LORE INDEX: fast token → entry lookup ────────────────────────
function rebuildLoreIndex(memory) {
    const idx = {};
    for (const e of memory.memory_entries || []) {
        if (!isActiveEntry(e)) continue;
        const tokens = tokenize(`${e.kind} ${e.text} ${(e.entities || []).join(" ")}`);
        for (const t of tokens) {
            if (!idx[t]) idx[t] = [];
            if (!idx[t].includes(e.id)) idx[t].push(e.id);
        }
    }
    memory.lore_index = idx;
}

function fastRetrieve(memory, query, { limit = 24 } = {}) {
    const qTokens = tokenize(query);
    const idx     = memory.lore_index || {};
    const scores  = new Map();
    const now     = Date.now();
    const entryMap = Object.fromEntries((memory.memory_entries || []).map(e => [e.id, e]));

    // Fast candidate selection via inverted index
    for (const t of qTokens) {
        for (const id of (idx[t] || [])) {
            scores.set(id, (scores.get(id) || 0) + 1);
        }
    }

    // Score candidates
    const candidates = [...scores.entries()].map(([id, hits]) => {
        const e = entryMap[id];
        if (!e || !isActiveEntry(e)) return null;
        const eTokens   = tokenize(`${e.kind} ${e.text} ${(e.entities || []).join(" ")}`);
        const relevance = jaccardScore(qTokens, eTokens);
        const ageDays   = Math.max(0, (now - new Date(e.created_at || nowIso()).getTime()) / 86400000);
        const recency   = 1 / (1 + ageDays / 30);
        const importance = (e.importance || 3) / 5;
        const confidence = e.confidence || 0.75;
        const score = 0.5 * relevance + 0.2 * recency + 0.2 * importance + 0.1 * confidence;
        return { ...e, score };
    }).filter(Boolean);

    // If few candidates, fallback to full scan for top entries by importance
    if (candidates.length < 6) {
        const fallback = (memory.memory_entries || [])
            .filter(isActiveEntry)
            .sort((a,b) => (b.importance||3) - (a.importance||3))
            .slice(0, 8)
            .map(e => ({ ...e, score: (e.importance||3)/5 }));
        const ids = new Set(candidates.map(x=>x.id));
        for (const e of fallback) if (!ids.has(e.id)) candidates.push(e);
    }

    return candidates.sort((a,b) => b.score - a.score).slice(0, limit)
        .map(e => ({
            id: e.id, kind: e.kind, text: e.text,
            entities: e.entities, importance: e.importance,
            score: Number(e.score.toFixed(4))
        }));
}

function migrateMemory(mem) {
    const out = (mem && typeof mem === "object") ? mem : createBaseMemory();
    if (!out.schema_version) out.schema_version = 1;
    out.game = out.game || { ...DEFAULT_GAME };
    for (const k of Object.keys(DEFAULT_GAME)) if (!out.game[k]) out.game[k] = Array.isArray(DEFAULT_GAME[k]) ? [] : DEFAULT_GAME[k];
    out.preferences = { ...DEFAULT_PREFERENCES, ...(out.preferences || {}) };
    out.project = out.project || { notes: [], pins: [] };
    out.project.notes = out.project.notes || [];
    out.project.pins  = out.project.pins  || [];
    if (!Array.isArray(out.memory_entries)) out.memory_entries = [];

    if (out.schema_version < 2) {
        const migrated = [];
        for (const lore of out.game.lore || []) { if (typeof lore === "string" && lore.trim()) migrated.push(toEntry({ kind: "lore", text: lore, importance: 4 })); }
        for (const p of out.game.plot || []) { if (typeof p === "string" && p.trim()) migrated.push(toEntry({ kind: "plot", text: p, importance: 4 })); }
        for (const c of out.game.characters || []) { if (!c?.name) continue; migrated.push(toEntry({ kind: "character", text: `${c.name}: ${c.description || ""}`.trim(), entities: [c.name], importance: 5 })); }
        for (const l of out.game.locations || []) { if (!l?.name) continue; migrated.push(toEntry({ kind: "location", text: `${l.name}: ${l.description || ""}`.trim(), entities: [l.name], importance: 4 })); }
        for (const f of out.game.factions || []) { if (!f?.name) continue; migrated.push(toEntry({ kind: "faction", text: `${f.name}: ${f.description || ""}`.trim(), entities: [f.name], importance: 4 })); }
        out.memory_entries.push(...migrated);
        out.schema_version = 2;
    }

    if (out.schema_version < 3) {
        out.lore_index = {};
        out.preferences.active_model = out.preferences.active_model || "auto";
        out.schema_version = 3;
    }

    // Dedup
    const seen = new Set();
    out.memory_entries = out.memory_entries.filter(e => {
        if (!e?.text) return false;
        const key = `${e.kind || "misc"}:${normalize(e.text)}`;
        if (seen.has(key)) return false;
        seen.add(key); return true;
    });

    rebuildLoreIndex(out);
    return out;
}

function loadMemory() {
    if (!fs.existsSync(MEMORY_FILE)) { const base = createBaseMemory(); safeJsonWrite(MEMORY_FILE, base); return base; }
    const migrated = migrateMemory(safeJsonRead(MEMORY_FILE, createBaseMemory()));
    safeJsonWrite(MEMORY_FILE, migrated);
    return migrated;
}
function saveMemory(m) {
    rebuildLoreIndex(m);
    safeJsonWrite(MEMORY_FILE, m);
}

function addMemoryEntry(memory, entry) {
    const cleanText = String(entry.text || "").trim();
    if (!cleanText || cleanText.length < 8) return null;
    const existing = (memory.memory_entries || []).find(
        e => isActiveEntry(e) && e.kind === entry.kind && normalize(e.text) === normalize(cleanText)
    );
    if (existing) {
        existing.last_used_at = nowIso();
        existing.importance   = Math.min(5, Math.max(existing.importance, entry.importance || 3));
        return existing;
    }
    const typed = toEntry(entry);
    memory.memory_entries = memory.memory_entries || [];
    memory.memory_entries.push(typed);

    // Update lore index incrementally
    const tokens = tokenize(`${typed.kind} ${typed.text} ${(typed.entities || []).join(" ")}`);
    if (!memory.lore_index) memory.lore_index = {};
    for (const t of tokens) {
        if (!memory.lore_index[t]) memory.lore_index[t] = [];
        if (!memory.lore_index[t].includes(typed.id)) memory.lore_index[t].push(typed.id);
    }

    return typed;
}

function extractStoryCandidates(userMessage) {
    const msg = String(userMessage || "").trim();
    if (!msg) return [];
    const candidates = [];
    const add = (kind, text, opts = {}) => { if (!text || text.length < 8) return; candidates.push({ kind, text, ...opts }); };

    const rem = msg.match(/remember (?:that )?(.+)/i);
    if (rem) add("lore", rem[1].trim(), { importance: 5, confidence: 0.95 });

    const charP = [
        /^([A-Za-z][A-Za-z' -]{1,30}) is (?:a|an|my|the) (.+)/i,
        /my (?:main )?character[,\s]+([A-Za-z][A-Za-z' -]{1,30})[,\s]*(.*)/i,
        /character (?:named?|called) ([A-Za-z][A-Za-z' -]{1,30})[,\s]*(.*)/i,
        /([A-Za-z][A-Za-z' -]{1,30}) is my (?:main )?character[,\s]*(.*)/i
    ];
    for (const p of charP) {
        const m = msg.match(p); if (!m) continue;
        const name = toTitleCase(m[1]); const desc = (m[2] || "").trim();
        add("character", `${name}: ${desc}`.trim(), { entities: [name], importance: 5, confidence: 0.9 });
        break;
    }

    const locP = [
        /([A-Z][a-zA-Z\s]+) is (?:a|an|the) (?:city|town|village|kingdom|realm|dungeon|land|world|region)/,
        /(?:city|town|kingdom|realm|dungeon|land|world|region) (?:of|called|named?) ([A-Z][a-zA-Z\s]+)/
    ];
    for (const p of locP) {
        const m = msg.match(p); if (!m) continue;
        add("location", msg.slice(0, 220), { entities: [(m[1] || "").trim()], importance: 4, confidence: 0.85 });
        break;
    }

    const facP = [
        /([A-Z][a-zA-Z\s]+) is (?:a|an|the) (?:faction|guild|order|clan|army|empire|cult)/,
        /(?:faction|guild|order|clan|army|empire|cult) (?:called|named?) ([A-Z][a-zA-Z\s]+)/
    ];
    for (const p of facP) {
        const m = msg.match(p); if (!m) continue;
        add("faction", msg.slice(0, 220), { entities: [(m[1] || "").trim()], importance: 4, confidence: 0.85 });
        break;
    }

    const strictLoreP = [
        /^the (?:world|story|war|curse|magic|darkness|light|void)\b.+/i,
        /^in (?:this|the) (?:world|story|realm|land)\b.+/i
    ];
    for (const p of strictLoreP) {
        if (p.test(msg) && msg.length > 20) { add("lore", msg.slice(0, 250), { importance: 4, confidence: 0.82 }); break; }
    }

    return candidates;
}

function extractExplicitMemoryCandidates(userMessage) {
    const msg = String(userMessage || "").trim();
    if (!msg) return [];
    const candidates = [];
    const add = (text, importance = 5, confidence = 0.98) => {
        if (!text || text.trim().length < 3) return;
        candidates.push({ kind: "lore", text: text.trim(), importance, confidence, source: "explicit_memory" });
    };
    const patterns = [
        /^(?:please\s+)?remember(?:\s+that)?\s+(.+)$/i,
        /^(?:please\s+)?save(?:\s+this)?\s*:\s*(.+)$/i,
        /^(?:please\s+)?save(?:\s+this)?\s+(.+)$/i,
        /^(?:please\s+)?store(?:\s+this)?\s*:\s*(.+)$/i,
        /^(?:please\s+)?note(?:\s+that)?\s+(.+)$/i,
        /^(?:this is important|important)\s*:\s*(.+)$/i
    ];
    for (const p of patterns) { const m = msg.match(p); if (m?.[1]) add(m[1]); }
    return candidates;
}

function isMemoryOnlyCommand(message) {
    const m = String(message || "").trim();
    return /^(?:please\s+)?(remember|save|store|note)\b/i.test(m) || /^(?:this is important|important)\s*:/i.test(m);
}

function isMemoryQuery(message) {
    const m = normalize(message);
    return /\b(what do you know|what is in memory|what do you remember|memory recap|summarize what you know)\b/.test(m);
}

function isLoreCheckQuery(message) {
    const m = normalize(message);
    return /\b(does .+ exist|is .+ real|check the lore|lore check|verify|is that correct|is that right|confirm|contradict|consistent|inconsistent)\b/.test(m);
}

function shouldAutoCaptureMessage(message) {
    const msg = String(message || "").trim();
    if (!msg || msg.endsWith("?")) return false;
    if (isMemoryOnlyCommand(msg)) return true;
    return /(?:\bcharacter\b|\bfaction\b|\blocation\b|\blore\b|\bplot\b|\bworld\b|\brealm\b|\bkingdom\b|\bquest\b|\bitem\b)/i.test(msg)
        || /\b(?:is|are|was|were)\b.+\b(?:character|leader|city|town|realm|faction|guild|weapon|artifact)\b/i.test(msg);
}

function maybePersistCandidates(memory, candidates) {
    const persisted = [];
    for (const c of candidates) {
        const score = 0.45 * ((c.importance || 3) / 5) + 0.35 * (c.confidence || 0.75) + 0.2 * Math.min(1, (c.text?.length || 0) / 120);
        if (score < 0.65) continue;
        const entry = addMemoryEntry(memory, c);
        if (entry) persisted.push(entry);

        if (c.kind === "lore") { memory.game.lore.push(c.text); if (memory.game.lore.length > 200) memory.game.lore = memory.game.lore.slice(-200); }
        if (c.kind === "plot") { memory.game.plot.push(c.text); if (memory.game.plot.length > 120) memory.game.plot = memory.game.plot.slice(-120); }
        if (c.kind === "character") {
            const [name, ...rest] = c.text.split(":");
            const desc = rest.join(":").trim();
            const ex = memory.game.characters.find(x => x.name?.toLowerCase() === name.trim().toLowerCase());
            if (ex) ex.description = `${ex.description || ""} ${desc}`.trim();
            else memory.game.characters.push({ name: name.trim(), role: "character", description: desc.slice(0, 250) });
        }
        if (c.kind === "location") {
            const name = c.entities?.[0] || c.text.slice(0, 40);
            if (!memory.game.locations.find(x => x.name?.toLowerCase() === name.toLowerCase()))
                memory.game.locations.push({ name, description: c.text.slice(0, 250) });
        }
        if (c.kind === "faction") {
            const name = c.entities?.[0] || c.text.slice(0, 40);
            if (!memory.game.factions.find(x => x.name?.toLowerCase() === name.toLowerCase()))
                memory.game.factions.push({ name, description: c.text.slice(0, 250) });
        }
    }
    return persisted;
}

function buildDeterministicMemoryReply(memory) {
    const game   = memory.game || DEFAULT_GAME;
    const active = (memory.memory_entries || []).filter(isActiveEntry);
    const top    = [...active].sort((a, b) => (b.importance || 3) - (a.importance || 3)).slice(0, 10);
    const lines  = [`I currently have ${active.length} active facts recorded for "${game.title}".`];
    if (game.characters?.length) {
        lines.push("\nCharacters:");
        game.characters.slice(-6).forEach(c => lines.push(`• ${c.name} (${c.role || "character"}): ${c.description || "no description"}`));
    }
    if (game.lore?.length) {
        lines.push("\nLore:");
        game.lore.slice(-6).forEach(l => lines.push(`• ${l}`));
    }
    if (top.length) {
        lines.push("\nTop entries:");
        top.forEach(e => lines.push(`• [${e.kind}] ${e.text}`));
    }
    return lines.join("\n");
}

// ─── DEEPSEEK: LORE VERIFICATION ─────────────────────────────────
async function verifyLoreWithDeepSeek(query, memory) {
    if (!logicModel && !creativeModel) return null;
    const cacheKey = getCacheKey(`lore_verify:${query}`);
    const cached   = getCached(cacheKey);
    if (cached) return cached;

    const recalled = fastRetrieve(memory, query, { limit: 12 });
    if (!recalled.length) return null;

    const loreDump = recalled.map(e => `[${e.kind}] ${e.text}`).join("\n");
    const prompt   = `Given these recorded story facts:\n\n${loreDump}\n\nQuestion: ${query}\n\nAnalyze whether this is consistent with the recorded lore. Be concise and direct.`;

    try {
        const session = await createLogicSession();
        await session.setChatHistory([{
            type: "system",
            text: `You are a lore consistency checker for "Death Before The Light". Analyze facts carefully and flag contradictions. Be brief and precise. Use <think> tags internally if needed but only output your final answer.`
        }]);
        let result = "";
        await session.prompt(prompt, { onTextChunk(c) { result += c; } });
        // Strip DeepSeek's <think>...</think> blocks from output
        result = result.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
        setCache(cacheKey, result);
        return result;
    } catch (err) {
        console.error("DeepSeek lore check error:", err.message);
        return null;
    }
}

// ─── DEEPSEEK: SESSION SUMMARIZATION ─────────────────────────────
async function summarizeSession(chatHist) {
    if (!chatHist || chatHist.length < 2) return null;
    const convo = chatHist.map(m => `${m.role === "user" ? "Creator" : "Companion"}: ${m.text}`).join("\n").slice(0, 2600);
    try {
        const session = await createLogicSession();
        await session.setChatHistory([{
            type: "system",
            text: `You are a lore archivist. Summarize the session in 3-5 bullet points. Focus on characters, plot, lore, locations. Be concise. Strip any <think> blocks.`
        }]);
        let raw = "";
        await session.prompt(`Summarize:\n\n${convo}`, { onTextChunk(c) { raw += c; } });
        raw = raw.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
        return raw;
    } catch { return null; }
}

// ─── DEEPSEEK: TITLE GENERATION ──────────────────────────────────
async function generateTitle(msg) {
    try {
        const session = await createLogicSession();
        await session.setChatHistory([{ type: "system", text: "Generate a 4-6 word chat title. Reply ONLY with the title. No punctuation at start or end." }]);
        let title = "";
        await session.prompt(msg, { onTextChunk(c) { title += c; } });
        title = title.replace(/<think>[\s\S]*?<\/think>/g, "").trim().replace(/^\W+|\W+$/g, "").replace(/\n.*/g, "").trim();
        if (!title || title.length > 60) title = msg.slice(0, 40).trim();
        return title;
    } catch { return msg.slice(0, 40).trim(); }
}

// ─── SYSTEM PROMPT (Qwen Creative) ───────────────────────────────
function buildSystemPrompt(memory, userMessage, storyMode = false, recentWrites = []) {
    const game           = memory.game || DEFAULT_GAME;
    const responseLength = memory.preferences?.response_length || "natural";
    const dmPov          = memory.preferences?.dm_perspective || "second";

    const recalled = fastRetrieve(memory, userMessage || "", { limit: 20 });
    const grouped  = {
        character: recalled.filter(x => x.kind === "character"),
        location:  recalled.filter(x => x.kind === "location"),
        faction:   recalled.filter(x => x.kind === "faction"),
        lore:      recalled.filter(x => x.kind === "lore"),
        plot:      recalled.filter(x => x.kind === "plot")
    };

    const parts = [];
    if (grouped.character.length) parts.push(`CHARACTERS:\n${grouped.character.map(x => `• ${x.text}`).join("\n")}`);
    else if (game.characters?.length) parts.push(`CHARACTERS:\n${game.characters.slice(-8).map(c => `• ${c.name} (${c.role || "character"}): ${c.description || ""}`).join("\n")}`);

    if (grouped.faction.length) parts.push(`FACTIONS:\n${grouped.faction.map(x => `• ${x.text}`).join("\n")}`);
    else if (game.factions?.length) parts.push(`FACTIONS:\n${game.factions.slice(-6).map(f => `• ${f.name}: ${f.description || ""}`).join("\n")}`);

    if (grouped.location.length) parts.push(`LOCATIONS:\n${grouped.location.map(x => `• ${x.text}`).join("\n")}`);
    else if (game.locations?.length) parts.push(`LOCATIONS:\n${game.locations.slice(-6).map(l => `• ${l.name}: ${l.description || ""}`).join("\n")}`);

    if (grouped.lore.length) parts.push(`LORE:\n${grouped.lore.map(x => `• ${x.text}`).join("\n")}`);
    else if (game.lore?.length) parts.push(`LORE:\n${game.lore.slice(-12).map(l => `• ${l}`).join("\n")}`);

    if (grouped.plot.length) parts.push(`PLOT:\n${grouped.plot.map(x => `• ${x.text}`).join("\n")}`);
    else if (game.plot?.length) parts.push(`PLOT:\n${game.plot.slice(-8).map(p => `• ${p}`).join("\n")}`);

    if (game.sessions?.length) parts.push(`PAST SESSIONS:\n${game.sessions.slice(-3).join("\n\n")}`);
    if (memory.project?.pins?.length) parts.push(`PINNED NOTES:\n${memory.project.pins.slice(-10).map(p => `• ${p}`).join("\n")}`);

    const justRecorded = recentWrites.length
        ? `\n\nJUST RECORDED THIS TURN:\n${recentWrites.slice(-8).map(x => `• ${x.text}`).join("\n")}`
        : "";

    const storyBlock = parts.length > 0
        ? `STORY MEMORY — "${game.title}":\n${parts.join("\n\n")}${justRecorded}`
        : `No story details recorded yet for "${game.title}".${justRecorded}`;

    if (storyMode) {
        const pov = dmPov === "second" ? "second person (You...)" : "third person";
        return `You are the Narrator of "${game.title}" — an immersive dark fantasy RPG.

STORY MODE:
- Narrate in ${pov}, present tense
- Only reference recorded story facts below for world details
- Create vivid, atmospheric descriptions fitting the dark tone
- End each reply with a prompt for Creator's next action
- When Creator saves something, confirm: "Saved to memory."

RESPONSE LENGTH: ${responseLength}

${storyBlock}`.trim();
    }

    return `You are a helpful, knowledgeable AI assistant and creative companion for the Creator.

GENERAL ASSISTANT:
- Answer any question freely on any topic: coding, writing, science, history, math, advice, RPG design, etc.
- Use your full knowledge just like ChatGPT or Claude would
- Be conversational, creative, and genuinely helpful

RPG COMPANION for "${game.title}":
- When the Creator discusses their game, use the story memory below
- For story details NOT in memory, say you don't have that recorded and ask Creator to share it
- When new story info is shared, confirm you're recording it
- Never invent characters, places, or events that aren't recorded

SLASH COMMANDS: /characters /lore /plot /locations /factions /sheet [name]

RESPONSE LENGTH: ${responseLength}

${storyBlock}`.trim();
}

// ─── SLASH COMMANDS ──────────────────────────────────────────────
function handleSlashCommand(cmd, memory) {
    const game = memory.game || DEFAULT_GAME;
    const c    = cmd.toLowerCase().trim();
    if (c === "/characters") { if (!game.characters.length) return "No characters recorded yet."; return `**Characters in ${game.title}:**\n\n` + game.characters.map(ch => `**${ch.name}** (${ch.role || "character"})\n${ch.description || ""}`).join("\n\n"); }
    if (c === "/lore")       { if (!game.lore.length)       return "No lore recorded yet.";       return `**Lore & Facts:**\n\n` + game.lore.slice(-60).map(l => `• ${l}`).join("\n"); }
    if (c === "/plot")       { if (!game.plot.length)       return "No plot events recorded yet."; return `**Plot Events:**\n\n` + game.plot.slice(-60).map(p => `• ${p}`).join("\n"); }
    if (c === "/locations")  { if (!game.locations.length)  return "No locations recorded yet.";  return `**Locations:**\n\n` + game.locations.map(l => `**${l.name}**: ${l.description || ""}`).join("\n\n"); }
    if (c === "/factions")   { if (!game.factions.length)   return "No factions recorded yet.";   return `**Factions:**\n\n` + game.factions.map(f => `**${f.name}**: ${f.description || ""}`).join("\n\n"); }
    if (c === "/models")     { return `**Active Models:**\n\n🎭 **Creative Core**: ${creativeModel ? "Qwen 2.5 3B ✅" : "Not loaded ❌"}\n🧠 **Logic Layer**: ${logicModel ? "DeepSeek R1 1.5B ✅" : "Not loaded ❌"}\n\nMode: ${memory.preferences?.active_model || "auto"}`; }
    if (c.startsWith("/sheet")) {
        const name = c.replace("/sheet", "").trim();
        const char = game.characters.find(ch => ch.name?.toLowerCase() === name.toLowerCase());
        if (!char) return `No character named "${name}" found.`;
        return `**CHARACTER SHEET — ${char.name}**\n\n**Role:** ${char.role || "character"}\n**Description:** ${char.description || ""}`;
    }
    return null;
}

// ─── CHAT ENDPOINT ───────────────────────────────────────────────
app.post("/chat", async (req, res) => {
    const { message = "", chatId = "default", images = [], storyMode = false } = req.body;

    const memory   = loadMemory();
    const histPath = path.join(CHATS_DIR, `${chatId}.json`);
    let chatHist   = safeJsonRead(histPath, []);
    const isFirst  = chatHist.length === 0;
    if (chatHist.length > 60) chatHist = chatHist.slice(-60);

    res.setHeader("Content-Type", "text/plain; charset=utf-8");

    try {
        // Slash commands
        if (message.startsWith("/")) {
            const result = handleSlashCommand(message, memory);
            if (result) {
                res.write(result);
                chatHist.push({ role: "user", text: message }, { role: "ai", text: result });
                safeJsonWrite(histPath, chatHist);
                return res.end();
            }
        }

        let fullMessage = message;

        // Vision
        if (images.length > 0) {
            res.write("[🔍 Analyzing image...]\n\n");
            for (const img of images) {
                const desc = await describeImage(img.base64, img.name || "image.png");
                fullMessage = `[Image: ${img.name || "uploaded image"}]\nVisual analysis: ${desc}\n\n${fullMessage}`;
            }
        }

        // Auto-capture story facts
        let persistedThisTurn = [];
        if (message && shouldAutoCaptureMessage(message)) {
            const candidates = [...extractExplicitMemoryCandidates(message), ...extractStoryCandidates(message)];
            persistedThisTurn = maybePersistCandidates(memory, candidates);
        }

        // Explicit memory command — confirm immediately
        if (message && isMemoryOnlyCommand(message)) {
            const savedFacts = persistedThisTurn.map(x => `• ${x.text}`).join("\n");
            const confirm    = persistedThisTurn.length ? `✅ Saved to memory.\n\nRecorded:\n${savedFacts}` : `✅ Understood — saved what I could.`;
            res.write(confirm);
            saveMemory(memory);
            chatHist.push({ role: "user", text: message }, { role: "ai", text: confirm });
            safeJsonWrite(histPath, chatHist);
            return res.end();
        }

        // Memory query — deterministic readout
        if (message && isMemoryQuery(message)) {
            const reply = buildDeterministicMemoryReply(memory);
            res.write(reply);
            saveMemory(memory);
            chatHist.push({ role: "user", text: message }, { role: "ai", text: reply });
            safeJsonWrite(histPath, chatHist);
            return res.end();
        }

        // Lore verification — route to DeepSeek Logic Layer
        if (message && isLoreCheckQuery(message) && (logicModel || creativeModel)) {
            res.write("[🧠 Checking lore consistency...]\n\n");
            const verification = await verifyLoreWithDeepSeek(message, memory);
            if (verification) {
                res.write(verification);
                saveMemory(memory);
                chatHist.push({ role: "user", text: message }, { role: "ai", text: verification });
                safeJsonWrite(histPath, chatHist);
                return res.end();
            }
        }

        // Normal AI response — use Creative Core (Qwen)
        const session      = await createCreativeSession();
        const systemPrompt = buildSystemPrompt(memory, message, storyMode || memory.preferences?.story_mode, persistedThisTurn);
        const history      = [{ type: "system", text: systemPrompt }];
        for (let i = 0; i < chatHist.length; i += 2) {
            const u = chatHist[i], a = chatHist[i + 1];
            if (u) history.push({ type: "user",  text: u.text });
            if (a) history.push({ type: "model", response: [a.text] });
        }
        await session.setChatHistory(history);

        let aiReply = "";
        await session.prompt(fullMessage, { onTextChunk(chunk) { aiReply += chunk; res.write(chunk); } });

        aiReply = aiReply.trim();
        saveMemory(memory);
        chatHist.push({ role: "user", text: message || `[${images.length} image(s)]` }, { role: "ai", text: aiReply });
        safeJsonWrite(histPath, chatHist);

        if (isFirst) {
            const src = message || (images.length ? `Image: ${images[0].name || "upload"}` : "New chat");
            generateTitle(src).then(title => {
                const titles = loadTitles(); titles[chatId] = title; saveTitles(titles);
            });
        }

        res.end();

        // Auto-summarize every 10 messages (uses DeepSeek logic layer)
        if (chatHist.length > 0 && chatHist.length % 10 === 0) {
            summarizeSession(chatHist).then(summary => {
                if (!summary) return;
                const mem = loadMemory();
                mem.game.sessions.push(`[${new Date().toLocaleDateString()}]\n${summary}`);
                if (mem.game.sessions.length > 20) mem.game.sessions = mem.game.sessions.slice(-20);
                addMemoryEntry(mem, { kind: "session", text: summary.slice(0, 500), importance: 4, confidence: 0.7, source: "summary" });
                saveMemory(mem);
            });
        }
    } catch (err) {
        console.error("❌ Chat error:", err);
        res.status(500).end("Server error");
    }
});

// ─── IMAGE GEN ───────────────────────────────────────────────────
app.post("/generate-image", async (req, res) => {
    const { prompt } = req.body;
    if (!prompt) return res.status(400).json({ error: "No prompt provided" });
    try { res.json(await generateImage(prompt)); }
    catch (err) { res.status(500).json({ error: err.message }); }
});

// ─── EXPORT ──────────────────────────────────────────────────────
app.get("/export-chat/:id", (req, res) => {
    const fp = path.join(CHATS_DIR, `${req.params.id}.json`);
    if (!fs.existsSync(fp)) return res.status(404).json({ error: "Chat not found" });
    try {
        const hist   = safeJsonRead(fp, []);
        const titles = loadTitles();
        const title  = titles[req.params.id] || req.params.id;
        const memory = loadMemory();
        let md = `# ${title}\n*Creator's Companion — ${memory.game.title}*\n*${new Date().toLocaleDateString()}*\n\n---\n\n`;
        for (const m of hist) md += m.role === "user" ? `**Creator:** ${m.text}\n\n` : `**Lorekeeper:** ${m.text}\n\n`;
        res.setHeader("Content-Disposition", `attachment; filename="${title.replace(/[^a-z0-9]/gi, "_")}.md"`);
        res.setHeader("Content-Type", "text/markdown");
        res.send(md);
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.get("/export-sheet/:name", (req, res) => {
    const memory = loadMemory();
    const char   = memory.game.characters.find(c => c.name?.toLowerCase() === req.params.name.toLowerCase());
    if (!char) return res.status(404).json({ error: "Character not found" });
    const related = memory.game.lore.filter(l => l.toLowerCase().includes(char.name.toLowerCase()));
    let md = `# Character Sheet — ${char.name}\n*${memory.game.title}*\n\n---\n\n**Role:** ${char.role || "character"}\n\n**Description:** ${char.description || ""}\n\n`;
    if (related.length) { md += `**Known Facts:**\n`; for (const r of related) md += `- ${r}\n`; }
    res.setHeader("Content-Disposition", `attachment; filename="${char.name}_sheet.md"`);
    res.setHeader("Content-Type", "text/markdown");
    res.send(md);
});

// ─── MEMORY ROUTES ───────────────────────────────────────────────
app.post("/set-story-mode", (req, res) => {
    try {
        const memory = loadMemory();
        memory.preferences.story_mode = Boolean(req.body.enabled);
        if (req.body.perspective) memory.preferences.dm_perspective = req.body.perspective;
        saveMemory(memory); res.json({ ok: true });
    } catch { res.json({ ok: false }); }
});

app.post("/set-active-model", (req, res) => {
    const { model } = req.body; // "creative" | "logic" | "auto"
    if (!["creative","logic","auto"].includes(model)) return res.json({ ok: false, error: "Invalid model" });
    try {
        const memory = loadMemory();
        memory.preferences.active_model = model;
        saveMemory(memory);
        res.json({ ok: true, model });
    } catch { res.json({ ok: false }); }
});

app.post("/delete-memory-item", (req, res) => {
    const { section, index, id } = req.body;
    try {
        const memory = loadMemory();
        if (id) { const e = memory.memory_entries.find(x => x.id === id); if (e) e.invalidated_at = nowIso(); }
        else if (section !== undefined && index !== undefined) {
            if (section === "pins") memory.project.pins.splice(index, 1);
            else if (memory.game[section]) memory.game[section].splice(index, 1);
        }
        saveMemory(memory);
        res.json({ ok: true });
    } catch (err) { res.json({ ok: false, error: err.message }); }
});

app.get("/story-memory",    (req, res) => { try { res.json(loadMemory().game); } catch { res.json({}); } });
app.get("/view-memory",     (req, res) => { try { res.json(loadMemory());      } catch { res.json({}); } });

app.get("/memory/search", (req, res) => {
    try {
        const q = String(req.query.q || "");
        const memory = loadMemory();
        res.json({ q, results: fastRetrieve(memory, q, { limit: Math.min(100, Number(req.query.limit || 30)) }) });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.get("/memory/stats", (req, res) => {
    try {
        const memory = loadMemory();
        const active = memory.memory_entries.filter(isActiveEntry);
        const byKind = active.reduce((acc, e) => { acc[e.kind] = (acc[e.kind] || 0) + 1; return acc; }, {});
        const indexSize = Object.keys(memory.lore_index || {}).length;
        res.json({
            schema_version: memory.schema_version,
            total_entries: memory.memory_entries.length,
            active_entries: active.length,
            lore_index_tokens: indexSize,
            cache_entries: cacheMap.size,
            models: {
                creative: creativeModel ? "Qwen 2.5 3B — loaded" : "not loaded",
                logic:    logicModel    ? "DeepSeek R1 1.5B — loaded" : "not loaded"
            },
            game: {
                lore:       memory.game.lore.length,
                plot:       memory.game.plot.length,
                characters: memory.game.characters.length,
                locations:  memory.game.locations.length,
                factions:   memory.game.factions.length,
                sessions:   memory.game.sessions.length
            },
            by_kind: byKind
        });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.get("/memory/conflicts", (req, res) => {
    try {
        const memory = loadMemory();
        const active = memory.memory_entries.filter(isActiveEntry);
        const map    = new Map();
        for (const e of active) {
            const key = `${e.kind}:${(e.entities || []).join("|").toLowerCase()}`;
            if (!key.endsWith(":")) { if (!map.has(key)) map.set(key, []); map.get(key).push(e); }
        }
        const conflicts = [];
        for (const [key, arr] of map.entries()) {
            if (arr.length < 2) continue;
            const ut = new Set(arr.map(x => normalize(x.text)));
            if (ut.size > 1) conflicts.push({ key, items: arr.slice(0, 10) });
        }
        res.json({ count: conflicts.length, conflicts });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.post("/memory/reindex", (req, res) => {
    try {
        const memory = migrateMemory(loadMemory());
        rebuildLoreIndex(memory);
        saveMemory(memory);
        res.json({ ok: true, total_entries: memory.memory_entries.length, index_tokens: Object.keys(memory.lore_index || {}).length });
    } catch (err) { res.status(500).json({ ok: false, error: err.message }); }
});

app.post("/forget", (req, res) => {
    try { if (fs.existsSync(MEMORY_FILE)) fs.unlinkSync(MEMORY_FILE); loadMemory(); res.json({ ok: true }); }
    catch { res.json({ ok: false }); }
});

app.post("/remember", (req, res) => {
    const { type, data, corrects_id } = req.body;
    if (!type || data === undefined) return res.json({ ok: false });
    try {
        const memory   = loadMemory();
        const kindMap  = { lore: "lore", plot: "plot", character: "character", location: "location", faction: "faction", item: "item" };
        const kind     = kindMap[type] || "lore";
        const text     = typeof data === "string" ? data : `${data.name || "Unknown"}: ${data.description || ""}`.trim();
        const entities = typeof data === "object" && data?.name ? [data.name] : [];
        if (corrects_id) {
            const old = memory.memory_entries.find(x => x.id === corrects_id && isActiveEntry(x));
            if (old) old.invalidated_at = nowIso();
        }
        const entry = addMemoryEntry(memory, { kind, text, entities, importance: 4, confidence: 0.9, source: "manual" });
        maybePersistCandidates(memory, [{ kind, text, entities, importance: 4, confidence: 0.9 }]);
        saveMemory(memory);
        res.json({ ok: true, entry });
    } catch (err) { res.json({ ok: false, error: err.message }); }
});

app.post("/pin-message", (req, res) => {
    const { text } = req.body;
    if (!text) return res.json({ ok: false });
    try {
        const memory = loadMemory();
        memory.project.pins.push(text);
        if (memory.project.pins.length > 40) memory.project.pins = memory.project.pins.slice(-40);
        saveMemory(memory);
        res.json({ ok: true });
    } catch { res.json({ ok: false }); }
});

// ─── CHAT FILE ROUTES ─────────────────────────────────────────────
const loadTitles = () => safeJsonRead(TITLES_FILE, {});
const saveTitles = (t) => safeJsonWrite(TITLES_FILE, t);

app.get("/list-chats", (req, res) => {
    try {
        const files  = fs.readdirSync(CHATS_DIR).filter(f => f.endsWith(".json"));
        const titles = loadTitles();
        const chats  = files.map(f => {
            const id   = f.replace(".json", "");
            const fp   = path.join(CHATS_DIR, f);
            const stat = fs.statSync(fp);
            const hist = safeJsonRead(fp, []);
            const last = [...hist].reverse().find(m => m.role === "user");
            const preview = last?.text ? last.text.slice(0, 60) + (last.text.length > 60 ? "…" : "") : "";
            return { id, title: titles[id] || preview || id, preview, updatedAt: stat.mtimeMs };
        });
        chats.sort((a, b) => b.updatedAt - a.updatedAt);
        res.json(chats);
    } catch { res.json([]); }
});

app.get("/get-chat/:id", (req, res) => {
    const fp = path.join(CHATS_DIR, `${req.params.id}.json`);
    if (!fs.existsSync(fp)) return res.json([]);
    res.json(safeJsonRead(fp, []));
});

app.delete("/delete-chat/:id", (req, res) => {
    const fp = path.join(CHATS_DIR, `${req.params.id}.json`);
    try {
        if (fs.existsSync(fp)) fs.unlinkSync(fp);
        const titles = loadTitles(); delete titles[req.params.id]; saveTitles(titles);
        res.json({ ok: true });
    } catch { res.json({ ok: false }); }
});

// ─── MODEL STATUS ────────────────────────────────────────────────
app.get("/model-status", (req, res) => {
    res.json({
        creative: { loaded: !!creativeModel, name: "Qwen 2.5 3B", role: "Dialogue & Creative" },
        logic:    { loaded: !!logicModel,    name: "DeepSeek R1 1.5B", role: "Lore & Logic" }
    });
});

// ─── START ───────────────────────────────────────────────────────
init().then(() => {
    app.listen(PORT, "0.0.0.0", () => {
        console.log(`\n🔥 Server: http://localhost:${PORT}`);
        console.log(`📱 Tablet: http://[YOUR-PC-IP]:${PORT}/node-chatbot.html`);
        console.log(`🖼️  Vision: ${VISION_URL}`);
        console.log(`⚔️  Game: Death Before The Light`);
        console.log(`\n🎭 Creative Core (Qwen 2.5 3B):    ${creativeModel ? "✅ Ready" : "❌ Not loaded"}`);
        console.log(`🧠 Logic Layer  (DeepSeek R1 1.5B): ${logicModel    ? "✅ Ready" : "❌ Not loaded"}`);
        console.log(`\nBrain files expected at:`);
        console.log(`  ${CREATIVE_MODEL_PATH}`);
        console.log(`  ${LOGIC_MODEL_PATH}\n`);
    });
}).catch(err => { console.error("❌ Failed to start:", err); process.exit(1); });