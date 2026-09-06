import React, { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { api } from '../api/client'

export default function CounterfactualControls({ windowId, hosts, currentGraph }) {
  const [actionType, setActionType] = useState('isolate_host')
  const [host, setHost] = useState(hosts[0] || '')
  const [edgeSrc, setEdgeSrc] = useState('')
  const [edgeDst, setEdgeDst] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const activeHostsInWindow = currentGraph
    ? Array.from(new Set(currentGraph.edges.flatMap((e) => [e.src, e.dst])))
    : hosts

  async function runSimulation() {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const payload = { window_id: windowId, type: actionType, k_steps: 6 }
      if (actionType === 'isolate_host') payload.host = host
      if (actionType === 'block_edge') { payload.src = edgeSrc; payload.dst = edgeDst }
      const res = await api.runCounterfactual(payload)
      setResult(res)
    } catch (e) {
      setError(windowId === 0
        ? 'Pick a later window -- a counterfactual needs at least one prior window of history.'
        : e.message)
    } finally {
      setLoading(false)
    }
  }

  const chartData = result
    ? result.baseline_risk_curve.map((v, i) => ({
        step: i + 1,
        baseline: v,
        counterfactual: result.counterfactual_risk_curve[i],
      }))
    : []

  return (
    <div>
      <div className="cf-controls">
        <select value={actionType} onChange={(e) => setActionType(e.target.value)}>
          <option value="isolate_host">Isolate host</option>
          <option value="block_edge">Block communication path</option>
        </select>

        {actionType === 'isolate_host' && (
          <select value={host} onChange={(e) => setHost(e.target.value)}>
            {activeHostsInWindow.map((h) => <option key={h} value={h}>{h}</option>)}
          </select>
        )}

        {actionType === 'block_edge' && (
          <>
            <select value={edgeSrc} onChange={(e) => setEdgeSrc(e.target.value)}>
              <option value="">from…</option>
              {activeHostsInWindow.map((h) => <option key={h} value={h}>{h}</option>)}
            </select>
            <select value={edgeDst} onChange={(e) => setEdgeDst(e.target.value)}>
              <option value="">to…</option>
              {activeHostsInWindow.map((h) => <option key={h} value={h}>{h}</option>)}
            </select>
          </>
        )}

        <button onClick={runSimulation} disabled={loading || windowId === 0}>
          {loading ? 'Simulating…' : 'Simulate defence'}
        </button>
      </div>

      {error && <div className="cf-result" style={{ color: 'var(--risk-exfil)' }}>{error}</div>}

      {result && (
        <>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData} margin={{ top: 4, right: 18, left: -12, bottom: 0 }}>
              <CartesianGrid stroke="#263242" strokeDasharray="3 5" vertical={false} />
              <XAxis dataKey="step" stroke="#55647a" fontSize={12} tickLine={false} axisLine={false} label={{ value: 'steps ahead', position: 'insideBottom', offset: -2, fill: '#55647a', fontSize: 11 }} />
              <YAxis domain={[0, 1]} stroke="#55647a" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <Tooltip contentStyle={{ background: '#1a2432', border: '1px solid #263242', fontSize: 12.5 }} formatter={(v) => v.toFixed(3)} />
              <Legend wrapperStyle={{ fontSize: 12.5 }} />
              <Line type="monotone" dataKey="baseline" name="No action" stroke="#e56a65" strokeWidth={2.6} dot={false} animationDuration={450} />
              <Line type="monotone" dataKey="counterfactual" name="With intervention" stroke="#52c3ae" strokeWidth={2.6} dot={false} animationDuration={450} />
            </LineChart>
          </ResponsiveContainer>
          <div className="cf-result">
            Removed {result.edges_removed} flow(s). Predicted risk after 6 windows drops by{' '}
            <strong>{(result.risk_reduction * 100).toFixed(0)} points</strong>{' '}
            ({(result.baseline_risk_curve.at(-1) * 100).toFixed(0)}% → {(result.counterfactual_risk_curve.at(-1) * 100).toFixed(0)}%).
          </div>
        </>
      )}
    </div>
  )
}
