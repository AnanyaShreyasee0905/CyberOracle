import React, { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts'

/**
 * Shows the observed risk trajectory up to the currently-selected window
 * as a solid line, and the K-step forecast (from the rollout endpoint)
 * as a dashed continuation past "now" -- this is what makes the
 * "predictive" claim visible rather than just a detector's current score.
 */
export default function TrajectoryTimeline({ windows, currentWindowId, forecast }) {
  const data = useMemo(() => {
    return windows.map((w) => ({
      window: w.window_id,
      observed: w.window_id <= currentWindowId ? w.predicted_risk : null,
    }))
  }, [windows, currentWindowId])

  if (forecast && forecast.length) {
    forecast.forEach((risk, i) => {
      const wid = currentWindowId + i + 1
      let row = data.find((d) => d.window === wid)
      if (!row) {
        row = { window: wid, observed: null }
        data.push(row)
      }
      row.forecast = risk
    })
    // bridge the gap so the dashed line connects to the solid line
    const bridge = data.find((d) => d.window === currentWindowId)
    if (bridge) bridge.forecast = bridge.observed
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart className="trajectory-chart" data={data} margin={{ top: 8, right: 18, left: -12, bottom: 0 }}>
        <CartesianGrid stroke="#263242" strokeDasharray="3 5" vertical={false} />
        <XAxis dataKey="window" stroke="#55647a" fontSize={12} tickLine={false} axisLine={false} />
        <YAxis domain={[0, 1]} stroke="#55647a" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `${Math.round(value * 100)}%`} />
        <Tooltip
          contentStyle={{ background: '#1a2432', border: '1px solid #263242', fontSize: 12.5 }}
          labelFormatter={(w) => `Window ${w}`}
          formatter={(v) => (v == null ? '—' : v.toFixed(3))}
        />
        <ReferenceLine x={currentWindowId} stroke="#5b8def" strokeDasharray="2 3" />
        <Line type="monotone" dataKey="observed" stroke="#6f9dff" strokeWidth={2.8} dot={false} activeDot={{ r: 4, strokeWidth: 0 }} animationDuration={450} />
        <Line type="monotone" dataKey="forecast" stroke="#e0a458" strokeWidth={2.5} strokeDasharray="6 5" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} animationDuration={450} />
      </LineChart>
    </ResponsiveContainer>
  )
}
