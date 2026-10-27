// Optional LLM report writer. Receives ONLY the numeric evidence JSON (never
// audio) and proxies it to any OpenAI-compatible chat endpoint, keeping the
// key server-side. Configure LLM_BASE_URL, LLM_API_KEY and LLM_MODEL.
export const config = { maxDuration: 30 }

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' })
  const { LLM_BASE_URL, LLM_API_KEY, LLM_MODEL } = process.env
  if (!LLM_BASE_URL || !LLM_API_KEY || !LLM_MODEL) return res.status(503).json({ error: 'LLM not configured' })
  const { messages } = req.body ?? {}
  if (!Array.isArray(messages) || messages.length > 4 || JSON.stringify(messages).length > 6000)
    return res.status(400).json({ error: 'bad request' })
  try {
    const r = await fetch(`${LLM_BASE_URL.replace(/\/$/, '')}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${LLM_API_KEY}` },
      body: JSON.stringify({ model: LLM_MODEL, messages, max_tokens: 220, temperature: 0 }),
      signal: AbortSignal.timeout(25000),
    })
    res.status(r.status).setHeader('Content-Type', 'application/json').send(await r.text())
  } catch {
    res.status(502).json({ error: 'upstream failed' })
  }
}
