import React, { useEffect, useState } from 'react'
import { api } from '../api/client'

export default function AnalystPanel({ windowId }) {
  const [result, setResult] = useState(null)

  useEffect(() => {
    let active = true
    setResult(null)
    api.getAnalyst(windowId).then((data) => {
      if (active) setResult(data)
    }).catch(() => {
      if (active) setResult({ error: 'Analyst summary is unavailable.' })
    })
    return () => { active = false }
  }, [windowId])

  if (!result) return <p>Generating analyst summary…</p>
  if (result.error) return <p>{result.error}</p>

  return (
    <div>
      <p><strong>Template fallback</strong> — SGLang is not configured.</p>
      <p>{result.summary}</p>
    </div>
  )
}
