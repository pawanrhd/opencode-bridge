/**
 * opencode-proxy (bridge)
 * Translates OpenAI-compatible API requests → OpenCode REST API → response
 * Supports TRUE streaming via OpenCode's /event SSE + /prompt_async,
 * multimodal images, multi-provider routing, tool/function calling,
 * retries, session cleanup, and request deduplication.
 *
 * Based on https://github.com/crazyboy24/opencode-proxy
 *
 * Changes in this version:
 *  - Default provider is now "opencode" (the Zen free-tier provider that's
 *    actually connected in this setup) instead of "github-copilot", so you
 *    no longer need to `set OPENCODE_PROVIDER_ID=opencode` before every run.
 *  - Streaming requests now use OpenCode's GET /event SSE stream + POST
 *    /session/{id}/prompt_async instead of blocking on the synchronous
 *    POST /session/{id}/message call. This means real incremental text
 *    reaches the client instead of just heartbeat comments, which is what
 *    was causing VS Code's "Response contained no choices" error on
 *    anything that took more than ~1-2 minutes to finish.
 *  - Both streaming and non-streaming responses fall back to
 *    reasoning_content when a model puts its entire answer there and
 *    leaves `content` empty (some Zen models do this).
 */

import express from "express"
import fs      from "fs"
import nodePath from "path"

// ─── Config ──────────────────────────────────────────────────────────────────

const PORT          = parseInt(process.env.PORT                 || "5000",    10)
const OPENCODE_URL  = (process.env.OPENCODE_URL                 || "http://localhost:4096").replace(/\/$/, "")
const OPENCODE_PASS = process.env.OPENCODE_SERVER_PASSWORD      || ""
const OPENCODE_USER = process.env.OPENCODE_SERVER_USERNAME      || "opencode"
// Default changed from "github-copilot" to "opencode" — this matches the
// provider that's actually connected in this OpenCode instance, so bare
// (unprefixed) model names now route correctly out of the box.
const PROVIDER_ID   = process.env.OPENCODE_PROVIDER_ID          || "opencode"
const DEFAULT_MODEL = process.env.DEFAULT_MODEL                 || "nemotron-3.5-lightning-free"
const BRIDGE_KEY    = process.env.OPENCODE_PROXY_API_KEY        || ""
const LOG_LEVEL     = process.env.LOG_LEVEL                     || "info"
const LOG_FILE      = process.env.LOG_FILE                      || ""           // e.g. /data/logs/bridge.log
const TIMEOUT_MS    = parseInt(process.env.TIMEOUT_MS           || "600000",   10)
const HEARTBEAT_MS  = parseInt(process.env.HEARTBEAT_MS         || "15000",    10)
const RETRY_COUNT   = parseInt(process.env.RETRY_COUNT          || "2",        10)
const RETRY_DELAY   = parseInt(process.env.RETRY_DELAY_MS       || "2000",     10)
const SESSION_TTL_H = parseInt(process.env.SESSION_TTL_HOURS    || "2",        10)
const CLEANUP_EVERY = parseInt(process.env.CLEANUP_INTERVAL_MS  || "3600000",  10) // 1hr

// ─── Automatic model fallback ────────────────────────────────────────────────
// If the requested model produces no real output within PROBE_TIMEOUT_MS,
// the bridge silently starts a fresh attempt on the next model in this list
// — on the SAME open connection, so VS Code never sees a failure, just a
// slightly longer wait before content starts appearing. Comma-separated,
// override with FALLBACK_MODELS env var. The requested model is always
// tried first regardless of whether it's in this list.
const FALLBACK_MODELS = (process.env.FALLBACK_MODELS
  || "nemotron-3.5-lightning-free,jev-1.13-free,mimo-v2.5-free")
  .split(",").map(s => s.trim()).filter(Boolean)
const PROBE_TIMEOUT_MS = parseInt(process.env.PROBE_TIMEOUT_MS || "40000", 10) // per-model window to see first real token
const MAX_ATTEMPTS     = parseInt(process.env.MAX_ATTEMPTS     || "3",     10) // requested model + up to N-1 fallbacks

// ─── Session map (conversation → OpenCode session) ──────────────────────────
const sessionMap = new Map() // conversationId → { sessionId, lastUsed }
const SESSION_MAP_TTL = SESSION_TTL_H * 60 * 60 * 1000

function pruneSessionMap() {
  const cutoff = Date.now() - SESSION_MAP_TTL
  for (const [k, v] of sessionMap.entries()) {
    if (v.lastUsed < cutoff) sessionMap.delete(k)
  }
}

// ─── Logger ──────────────────────────────────────────────────────────────────

const ts = () => new Date().toISOString()

let logStream = null
if (LOG_FILE) {
  const dir = nodePath.dirname(LOG_FILE)
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true })
  logStream = fs.createWriteStream(LOG_FILE, { flags: "a" })
}

function writeLog(line) {
  console.log(line)
  if (logStream) logStream.write(line + "\n")
}

const logger = {
  info:  (...a) => LOG_LEVEL !== "silent" && writeLog(`[${ts()}] INFO  ${a.join(" ")}`),
  debug: (...a) => LOG_LEVEL === "debug"  && writeLog(`[${ts()}] DEBUG ${a.join(" ")}`),
  error: (...a) => LOG_LEVEL !== "silent" && writeLog(`[${ts()}] ERROR ${a.join(" ")}`),
}

// ─── OpenCode REST helpers ────────────────────────────────────────────────────

function withTimeout(ms) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), ms)
  return { signal: controller.signal, clear: () => clearTimeout(timer) }
}

function baseHeaders() {
  const h = { "Content-Type": "application/json" }
  if (OPENCODE_PASS) h["Authorization"] = `Basic ${Buffer.from(`${OPENCODE_USER}:${OPENCODE_PASS}`).toString("base64")}`
  return h
}

async function ocGet(path, timeoutMs = TIMEOUT_MS) {
  const { signal, clear } = withTimeout(timeoutMs)
  const res = await fetch(`${OPENCODE_URL}${path}`, {
    headers: baseHeaders(),
    signal,
  }).finally(clear)
  if (!res.ok) throw new Error(`OpenCode GET ${path} → ${res.status}`)
  return res.json()
}

async function ocPost(path, body) {
  const { signal, clear } = withTimeout(TIMEOUT_MS)
  const res = await fetch(`${OPENCODE_URL}${path}`, {
    method:  "POST",
    headers: baseHeaders(),
    body:    JSON.stringify(body),
    signal,
  }).finally(clear)
  const text = await res.text()
  if (!res.ok) throw new Error(`OpenCode POST ${path} → ${res.status}: ${text.slice(0, 300)}`)
  if (!text)   throw new Error(`OpenCode POST ${path} → empty response`)
  try { return JSON.parse(text) }
  catch { throw new Error(`OpenCode POST ${path} → invalid JSON: ${text.slice(0, 300)}`) }
}

async function ocDelete(path) {
  const { signal, clear } = withTimeout(10000)
  const res = await fetch(`${OPENCODE_URL}${path}`, {
    method:  "DELETE",
    headers: baseHeaders(),
    signal,
  }).finally(clear)
  return res.ok
}

async function ocGetList(path) {
  const { signal, clear } = withTimeout(10000)
  const res = await fetch(`${OPENCODE_URL}${path}`, {
    headers: baseHeaders(),
    signal,
  }).finally(clear)
  if (!res.ok) throw new Error(`OpenCode GET ${path} → ${res.status}`)
  return res.json()
}

// ─── Retry helper ─────────────────────────────────────────────────────────────

async function withRetry(fn, retries = RETRY_COUNT, delayMs = RETRY_DELAY) {
  let lastErr
  for (let i = 0; i <= retries; i++) {
    try {
      return await fn()
    } catch (err) {
      lastErr = err
      if (err.name === "AbortError")     throw err
      if (err.message.match(/→ 4\d\d/))  throw err
      if (i < retries) {
        logger.error(`Attempt ${i + 1} failed: ${err.message} — retrying in ${delayMs}ms`)
        await new Promise(r => setTimeout(r, delayMs))
      }
    }
  }
  throw lastErr
}

// ─── Session cleanup ──────────────────────────────────────────────────────────

async function cleanupOldSessions() {
  try {
    const data     = await ocGetList("/session")
    const sessions = Array.isArray(data) ? data : (data.sessions ?? data.data ?? [])
    const cutoff   = Date.now() - SESSION_TTL_H * 60 * 60 * 1000
    let   deleted  = 0

    for (const s of sessions) {
      const created = s.time?.created ?? s.created ?? 0
      if (created < cutoff && s.title?.startsWith("bridge-")) {
        const ok = await ocDelete(`/session/${s.id}`)
        if (ok) deleted++
      }
    }

    if (deleted > 0) logger.info(`Session cleanup: deleted ${deleted} old sessions (>${SESSION_TTL_H}h)`)
  } catch (err) {
    logger.error("Session cleanup failed:", err.message)
  }
}

// ─── Message builder ─────────────────────────────────────────────────────────

function buildParts(messages, tools) {
  const parts  = []
  let   hasImg = false

  const systemMsg = messages.find(m => m.role === "system")
  if (systemMsg) {
    const text = typeof systemMsg.content === "string"
      ? systemMsg.content
      : systemMsg.content?.map(c => c.text ?? "").join("\n") ?? ""
    parts.push({ type: "text", text: `[SYSTEM]\n${text}` })
  }

  for (const m of messages) {
    if (m.role === "system") continue
    const role = m.role.toUpperCase()

    if (m.role === "assistant" && m.tool_calls?.length) {
      parts.push({ type: "text", text: `[${role}]` })
      if (typeof m.content === "string" && m.content) {
        parts.push({ type: "text", text: m.content })
      }
      for (const tc of m.tool_calls) {
        parts.push({
          type:       "tool-call",
          toolName:   tc.function?.name ?? tc.name,
          toolArgs:   (() => { try { return JSON.parse(tc.function?.arguments ?? "{}") } catch { return {} } })(),
          toolCallId: tc.id,
        })
      }
      continue
    }

    if (m.role === "tool") {
      parts.push({
        type:       "tool-result",
        toolCallId: m.tool_call_id,
        result:     m.content ?? "",
      })
      continue
    }

    if (typeof m.content === "string") {
      parts.push({ type: "text", text: `[${role}]\n${m.content}` })
      continue
    }

    if (Array.isArray(m.content)) {
      parts.push({ type: "text", text: `[${role}]` })

      for (const c of m.content) {
        if (c.type === "text") {
          parts.push({ type: "text", text: c.text ?? "" })

        } else if (c.type === "image_url") {
          hasImg      = true
          const url   = c.image_url?.url ?? ""

          if (url.startsWith("data:")) {
            const commaIdx = url.indexOf(",")
            const meta = url.slice(0, commaIdx)
            const data = url.slice(commaIdx + 1)
            const mediaType    = meta.replace("data:", "").replace(";base64", "")
            parts.push({ type: "image", source: { type: "base64", mediaType, data } })
          } else {
            parts.push({ type: "image", source: { type: "url", url } })
          }
        }
      }
    }
  }

  return { parts, hasImg }
}

// ─── Auth middleware ──────────────────────────────────────────────────────────

function authMiddleware(req, res, next) {
  if (!BRIDGE_KEY) return next()
  const header = req.headers["authorization"] ?? ""
  const token  = header.startsWith("Bearer ") ? header.slice(7) : header
  if (token !== BRIDGE_KEY) {
    return res.status(401).json({ error: { message: "Unauthorized", type: "auth_error" } })
  }
  next()
}

// ─── App ─────────────────────────────────────────────────────────────────────

const app = express()
app.use(express.json({ limit: "50mb" }))

// ─── Health ──────────────────────────────────────────────────────────────────

app.get("/health", async (req, res) => {
  try {
    const data = await ocGet("/global/health", 10000)
    res.json({ status: "ok", bridge_version: "1.3.0", opencode: { connected: true, ...data }, provider: PROVIDER_ID, active_sessions: sessionMap.size })
  } catch (err) {
    res.json({ status: "ok", bridge_version: "1.3.0", opencode: { connected: false, error: err.message }, provider: PROVIDER_ID, active_sessions: sessionMap.size })
  }
})

// ─── Models ──────────────────────────────────────────────────────────────────

app.get("/v1/models", authMiddleware, async (req, res) => {
  try {
    const data      = await ocGet("/provider")
    const connected = data.connected ?? []
    const models    = []

    for (const provider of data.all ?? []) {
      if (!connected.includes(provider.id)) continue
      if (!provider.models) continue
      for (const modelId of Object.keys(provider.models)) {
        models.push({ id: `${provider.id}/${modelId}`, object: "model", owned_by: provider.id, created: 0 })
      }
    }

    if (models.length === 0) throw new Error("No connected providers found")
    logger.debug(`Returning ${models.length} models from ${connected.length} connected providers`)
    return res.json({ object: "list", data: models })

  } catch (err) {
    logger.error("Failed to fetch models:", err.message)
    const fallback = [
      `${PROVIDER_ID}/nemotron-3.5-lightning-free`,
      `${PROVIDER_ID}/jev-1.13-free`,
    ].map(id => ({ id, object: "model", owned_by: id.split("/")[0], created: 0 }))
    res.json({ object: "list", data: fallback })
  }
})

// ─── Streaming via OpenCode's /event SSE + /prompt_async ─────────────────────
//
// This replaces the old approach of blocking on the synchronous
// POST /session/{id}/message call and only writing SSE heartbeats while
// waiting. Instead:
//   1. Open a subscription to GET /event BEFORE sending the prompt, so we
//      don't miss any early deltas.
//   2. Fire POST /session/{id}/prompt_async, which returns immediately —
//      the actual generation is observed entirely through the event stream.
//   3. Forward each message.part.updated text/reasoning delta to the
//      client as an SSE chunk, live, as it arrives.
//   4. Stop when session.idle (finished) or session.error fires, or when
//      the overall TIMEOUT_MS is hit.
//
// NOTE: exact event field names (message.part.updated / delta / sessionID)
// are based on OpenCode's own SDK usage pattern. If your OpenCode version
// emits slightly different field names, watch the raw stream with:
//   curl -N http://localhost:4096/event
// while sending a prompt from another terminal, and adjust the parsing
// below to match what you see.

// Runs ONE attempt against ONE model. Never writes to `res` directly —
// instead reports every event via callbacks so the caller decides whether
// to forward it live (already committed to this model) or buffer/drop it
// (still probing, may abandon this model for a fallback).
//
//   onFirstToken()      — called once, the first time real text/reasoning
//                          arrives. The caller uses this to "commit".
//   onDelta({type,text})— called for every text/reasoning delta.
//   onToolCall(part)    — called for every tool-call part.
//
// Throws if OpenCode itself errors. Resolves {textAcc, reasoningAcc,
// toolCalls} when the session goes idle. probeTimeoutMs, if given, aborts
// the attempt (throwing a recognizable error) if no real delta arrives in
// that window — the caller can then try the next fallback model.
async function runModelAttempt({ sessionId, providerID, modelID, msgParts, probeTimeoutMs, onFirstToken, onDelta, onToolCall }) {
  const { signal, clear } = withTimeout(TIMEOUT_MS)
  const eventRes = await fetch(`${OPENCODE_URL}/event`, { headers: baseHeaders(), signal })
  if (!eventRes.ok || !eventRes.body) {
    clear()
    throw new Error(`OpenCode GET /event → ${eventRes.status}`)
  }

  const reader  = eventRes.body.getReader()
  const decoder = new TextDecoder()
  let   buffer  = ""

  const promptRes = await fetch(`${OPENCODE_URL}/session/${sessionId}/prompt_async`, {
    method:  "POST",
    headers: baseHeaders(),
    body:    JSON.stringify({ model: { providerID, modelID }, parts: msgParts }),
  })
  if (!promptRes.ok) {
    clear()
    reader.cancel().catch(() => {})
    const text = await promptRes.text().catch(() => "")
    throw new Error(`OpenCode POST /session/${sessionId}/prompt_async → ${promptRes.status}: ${text.slice(0, 300)}`)
  }

  let textAcc      = ""
  let reasoningAcc = ""
  const toolCalls  = []
  let   done       = false
  let   gotFirst   = false

  let probeTimer = null
  let probeExpired = false
  if (probeTimeoutMs) {
    probeTimer = setTimeout(() => { probeExpired = true; reader.cancel().catch(() => {}) }, probeTimeoutMs)
  }

  try {
    while (!done && !probeExpired) {
      const { value, done: streamDone } = await reader.read()
      if (streamDone) break
      buffer += decoder.decode(value, { stream: true })

      let idx
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        const dataLine = frame.split("\n").find(l => l.startsWith("data:"))
        if (!dataLine) continue

        let evt
        try { evt = JSON.parse(dataLine.slice(5).trim()) } catch { continue }

        const props = evt.properties ?? {}
        if (props.sessionID && props.sessionID !== sessionId) continue

        if (evt.type === "message.part.updated") {
          const part  = props.part ?? {}
          const delta = props.delta

          if ((part.type === "text" || part.type === "reasoning") && delta) {
            if (!gotFirst) { gotFirst = true; if (probeTimer) clearTimeout(probeTimer); onFirstToken?.() }
            if (part.type === "text")      { textAcc      += delta; onDelta?.({ type: "text",      text: delta }) }
            else                            { reasoningAcc += delta; onDelta?.({ type: "reasoning", text: delta }) }
          } else if (part.type === "tool-call") {
            const tc = { toolCallId: part.toolCallId, toolName: part.toolName, toolArgs: part.toolArgs }
            toolCalls.push(tc)
            onToolCall?.(tc)
          }
        }

        if (evt.type === "session.idle" || evt.type === "session.error") {
          done = true
        }
      }
    }
  } finally {
    if (probeTimer) clearTimeout(probeTimer)
    clear()
    reader.cancel().catch(() => {})
  }

  if (probeExpired && !gotFirst) {
    const err = new Error(`No response from model ${modelID} within ${probeTimeoutMs}ms`)
    err.isProbeTimeout = true
    throw err
  }

  return { textAcc, reasoningAcc, toolCalls }
}

// Tries the requested model, then falls back through FALLBACK_MODELS (each
// on its own fresh session) until one produces real output, or attempts
// are exhausted. A shared heartbeat keeps the SSE connection alive across
// the whole process so VS Code never sees a gap, no matter how many models
// get tried internally.
async function streamWithFallback({ res, cmplId, created, requestedModelID, providerID, msgParts, makeSession }) {
  const candidates = [requestedModelID, ...FALLBACK_MODELS.filter(m => m !== requestedModelID)].slice(0, MAX_ATTEMPTS)

  const heartbeat = setInterval(() => { try { res.write(": heartbeat\n\n") } catch {} }, HEARTBEAT_MS)
  let committedModel = null
  let lastErr = null

  try {
    for (const candidateModel of candidates) {
      const sessionId = await makeSession()
      let firstTokenSeen = false

      const onFirstToken = () => {
        firstTokenSeen = true
        committedModel = candidateModel
        logger.debug(`committed to model ${candidateModel} (session ${sessionId})`)
      }
      const onDelta = ({ type, text }) => {
        const field = type === "text" ? "content" : "reasoning_content"
        res.write(`data: ${JSON.stringify({
          id: cmplId, object: "chat.completion.chunk", created, model: candidateModel,
          choices: [{ index: 0, delta: { [field]: text }, finish_reason: null }],
        })}\n\n`)
      }
      const onToolCall = () => {} // forwarded after completion, see caller

      try {
        const isLastCandidate = candidateModel === candidates[candidates.length - 1]
        const probeTimeoutMs  = isLastCandidate ? undefined : PROBE_TIMEOUT_MS
        const result = await runModelAttempt({
          sessionId, providerID, modelID: candidateModel, msgParts,
          probeTimeoutMs, onFirstToken, onDelta, onToolCall,
        })
        return { ...result, modelID: candidateModel }
      } catch (err) {
        lastErr = err
        if (firstTokenSeen) throw err // already committed and streaming live — a failure now is real, don't silently retry mid-stream
        logger.error(`model ${candidateModel} produced nothing (${err.message}) — trying next fallback`)
        continue
      }
    }
    throw lastErr ?? new Error("All fallback models failed")
  } finally {
    clearInterval(heartbeat)
  }
}

// ─── Chat completions ────────────────────────────────────────────────────────

app.post("/v1/chat/completions", authMiddleware, async (req, res) => {
  const reqId = `req_${Date.now()}`
  const { messages, model, stream, tools } = req.body

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({
      error: { message: "`messages` must be a non-empty array", type: "invalid_request_error" }
    })
  }

  let providerID = PROVIDER_ID
  let modelID    = model || DEFAULT_MODEL

  if (modelID.includes("/")) {
    const [p, ...m] = modelID.split("/")
    providerID = p
    modelID    = m.join("/")
  }

  logger.info(`[${reqId}] → provider=${providerID} model=${modelID} messages=${messages.length} stream=${!!stream} tools=${tools?.length ?? 0}`)

  const startMs = Date.now()

  const convId = req.headers["x-conversation-id"]
    ?? req.headers["x-session-id"]
    ?? (() => {
         const first = messages.find(m => m.role === "user")
         const text  = typeof first?.content === "string" ? first.content : JSON.stringify(first?.content)
         let h = 0
         for (const c of (text ?? "")) { h = (Math.imul(31, h) + c.charCodeAt(0)) | 0 }
         return `hash_${Math.abs(h)}`
       })()

  try {
    pruneSessionMap()
    let sessionId
    const existing = sessionMap.get(convId)
    const workDir = req.headers["x-working-directory"] ?? null

    if (existing) {
      sessionId = existing.sessionId
      existing.lastUsed = Date.now()
      logger.debug(`[${reqId}] reusing session ${sessionId} for conv ${convId}`)
    } else {
      const sessionBody = { title: `bridge-${reqId}` }
      if (workDir) sessionBody.directory = workDir
      const session = await withRetry(() => ocPost("/session", sessionBody))
      sessionId = session.id
      sessionMap.set(convId, { sessionId, lastUsed: Date.now() })
      logger.debug(`[${reqId}] new session ${sessionId} for conv ${convId}${workDir ? ` dir=${workDir}` : ""}`)
    }

    const msgsToSend = existing ? [messages[messages.length - 1]] : messages
    const { parts: msgParts, hasImg } = buildParts(msgsToSend, tools)
    if (hasImg) logger.info(`[${reqId}] multimodal — images detected`)

    const cmplId  = `chatcmpl-${sessionId}`
    const created = Math.floor(Date.now() / 1000)

    // ── Streaming path: real incremental deltas via /event + /prompt_async,
    //    with automatic silent fallback to a backup model if the requested
    //    one produces nothing within PROBE_TIMEOUT_MS. ─────────────────────
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream")
      res.setHeader("Cache-Control", "no-cache")
      res.setHeader("Connection", "keep-alive")
      res.flushHeaders()

      res.write(`data: ${JSON.stringify({
        id: cmplId, object: "chat.completion.chunk", created, model: modelID,
        choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
      })}\n\n`)

      // First attempt reuses the session already created above; any
      // fallback attempt gets its own fresh session so retries don't
      // confuse OpenCode's history for the original one.
      let usedFirstSession = false
      const makeSession = async () => {
        if (!usedFirstSession) { usedFirstSession = true; return sessionId }
        const s = await withRetry(() => ocPost("/session", { title: `bridge-${reqId}-fallback` }))
        return s.id
      }

      try {
        const { textAcc, reasoningAcc, toolCalls, modelID: usedModel } = await streamWithFallback({
          res, cmplId, created, requestedModelID: modelID, providerID, msgParts, makeSession,
        })

        const finishReason = toolCalls.length ? "tool_calls" : "stop"

        toolCalls.forEach((tc, i) => {
          res.write(`data: ${JSON.stringify({
            id: cmplId, object: "chat.completion.chunk", created, model: usedModel,
            choices: [{
              index: 0,
              delta: { tool_calls: [{ index: i, id: tc.toolCallId ?? `call_${i}`, type: "function", function: { name: tc.toolName, arguments: JSON.stringify(tc.toolArgs ?? {}) } }] },
              finish_reason: null,
            }],
          })}\n\n`)
        })

        res.write(`data: ${JSON.stringify({
          id: cmplId, object: "chat.completion.chunk", created, model: usedModel,
          choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
        })}\n\n`)
        res.write("data: [DONE]\n\n")

        const fallbackNote = usedModel !== modelID ? ` (fell back from ${modelID})` : ""
        logger.info(`[${reqId}] ✓ ${Date.now() - startMs}ms stream model=${usedModel}${fallbackNote} chars=${textAcc.length} reasoning_chars=${reasoningAcc.length} finish=${finishReason}`)
        return res.end()

      } catch (err) {
        logger.error(`[${reqId}] ✗ ${Date.now() - startMs}ms stream error (all models exhausted): ${err.message}`)
        sessionMap.delete(convId)
        res.write(`data: ${JSON.stringify({ error: { message: err.message, type: "bridge_error" } })}\n\n`)
        res.write("data: [DONE]\n\n")
        return res.end()
      }
    }

    // ── Non-streaming path: blocking call, with automatic fallback to a
    //    backup model if the requested one errors out entirely. ────────────
    const nonStreamCandidates = [modelID, ...FALLBACK_MODELS.filter(m => m !== modelID)].slice(0, MAX_ATTEMPTS)
    let result, usedModel = modelID, nsFirstSessionUsed = false
    let nsLastErr = null
    for (const candidateModel of nonStreamCandidates) {
      const candidateSessionId = nsFirstSessionUsed
        ? (await withRetry(() => ocPost("/session", { title: `bridge-${reqId}-fallback` }))).id
        : sessionId
      nsFirstSessionUsed = true
      try {
        result    = await withRetry(() => ocPost(`/session/${candidateSessionId}/message`, {
          model: { providerID, modelID: candidateModel },
          parts: msgParts,
        }))
        usedModel = candidateModel
        break
      } catch (err) {
        nsLastErr = err
        logger.error(`model ${candidateModel} failed (${err.message}) — trying next fallback`)
      }
    }
    if (!result) throw nsLastErr ?? new Error("All fallback models failed")
    modelID = usedModel

    const resParts = result.parts ?? []

    const textSegments = resParts
      .filter(p => p.type === "text" && p.text)
      .map(p => p.text.trim())
      .filter(Boolean)
    const responseText = textSegments.join("\n\n")

    const reasoningSegments = resParts
      .filter(p => p.type === "reasoning" && p.text)
      .map(p => p.text.trim())
      .filter(Boolean)
    const reasoningText = reasoningSegments.join("\n\n")

    const ocToolResults = resParts.filter(p => p.type === "tool-result")
    const toolResultBlock = ocToolResults.length
      ? "\n\n---\n**Tool outputs:**\n" + ocToolResults
          .map(r => `**${r.toolName ?? "tool"}:** ${typeof r.result === "string" ? r.result : JSON.stringify(r.result)}`)
          .join("\n")
      : ""
    const fullResponseText = responseText + toolResultBlock

    const toolCallParts = resParts.filter(p => p.type === "tool-call")
    const toolCalls     = toolCallParts.length
      ? toolCallParts.map((tc, i) => ({
          id:       tc.toolCallId ?? `call_${i}`,
          type:     "function",
          function: { name: tc.toolName, arguments: JSON.stringify(tc.toolArgs ?? {}) },
        }))
      : undefined

    const usage = {
      prompt_tokens:     result.info?.tokens?.input  ?? 0,
      completion_tokens: result.info?.tokens?.output ?? 0,
      total_tokens:      result.info?.tokens?.total  ?? 0,
    }

    const finishReason = toolCalls?.length ? "tool_calls" : "stop"
    logger.info(`[${reqId}] ✓ ${Date.now() - startMs}ms tokens=${usage.total_tokens} chars=${fullResponseText.length} reasoning_chars=${reasoningText.length} steps=${textSegments.length} finish=${finishReason}`)

    // Fall back to reasoning_content when the model left `content` empty —
    // some Zen models put their entire answer in the reasoning channel.
    const message = {
      role:    "assistant",
      content: toolCalls ? (fullResponseText || null) : (fullResponseText || reasoningText || ""),
      ...(reasoningText ? { reasoning_content: reasoningText } : {}),
      ...(toolCalls ? { tool_calls: toolCalls } : {}),
    }

    return res.json({
      id: cmplId, object: "chat.completion", created, model: modelID,
      choices: [{ index: 0, message, finish_reason: finishReason }],
      usage,
    })

  } catch (err) {
    logger.error(`[${reqId}] ✗ ${Date.now() - startMs}ms ${err.message}`)
    sessionMap.delete(convId)

    if (stream) {
      if (!res.headersSent) {
        res.setHeader("Content-Type", "text/event-stream")
        res.setHeader("Cache-Control", "no-cache")
        res.setHeader("Connection", "keep-alive")
        res.flushHeaders()
      }
      res.write(`data: ${JSON.stringify({ error: { message: err.message, type: "bridge_error" } })}\n\n`)
      res.write("data: [DONE]\n\n")
      return res.end()
    }

    if (!res.headersSent) {
      return res.status(502).json({ error: { message: err.message, type: "bridge_error" } })
    }
  }
})

// ─── 404 ─────────────────────────────────────────────────────────────────────

app.use((req, res) => {
  res.status(404).json({ error: { message: `Route ${req.method} ${req.path} not found` } })
})

// ─── Start ───────────────────────────────────────────────────────────────────

const server = app.listen(PORT, "0.0.0.0", async () => {
  logger.info(`opencode-bridge v1.3.0 started`)
  logger.info(`  Listening  : http://0.0.0.0:${PORT}`)
  logger.info(`  OpenCode   : ${OPENCODE_URL}`)
  logger.info(`  Provider   : ${PROVIDER_ID}`)
  logger.info(`  Auth       : ${BRIDGE_KEY    ? "enabled" : "disabled"}`)
  logger.info(`  OC Auth    : ${OPENCODE_PASS ? `enabled (user=${OPENCODE_USER})` : "disabled"}`)
  logger.info(`  Timeout    : ${TIMEOUT_MS}ms`)
  logger.info(`  Heartbeat  : ${HEARTBEAT_MS}ms`)
  logger.info(`  Retries    : ${RETRY_COUNT} × ${RETRY_DELAY}ms delay`)
  logger.info(`  Sessions   : cleanup every ${CLEANUP_EVERY / 60000}min, TTL ${SESSION_TTL_H}h`)
  logger.info(`  Log file   : ${LOG_FILE || "stdout only"}`)
  logger.info(`  Streaming  : live via /event + /prompt_async`)

  try {
    const h = await ocGet("/global/health", 10000)
    logger.info(`  OpenCode health: ✓ v${h.version ?? "unknown"}`)
  } catch {
    logger.error(`  OpenCode health: ✗ not reachable — check OPENCODE_URL`)
  }

  setInterval(cleanupOldSessions, CLEANUP_EVERY)
  setTimeout(cleanupOldSessions, 30000)
})

// ─── Graceful shutdown ───────────────────────────────────────────────────────

const shutdown = (signal) => {
  logger.info(`${signal} received, shutting down…`)
  if (logStream) logStream.end()
  server.close(() => { logger.info("Server closed"); process.exit(0) })
  setTimeout(() => process.exit(1), 5000)
}

process.on("SIGTERM", () => shutdown("SIGTERM"))
process.on("SIGINT",  () => shutdown("SIGINT"))