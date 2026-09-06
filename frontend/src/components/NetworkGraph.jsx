import React, { useMemo } from 'react'

const STAGE_COLOR_VAR = {
  benign: '--risk-benign',
  reconnaissance: '--risk-recon',
  credential_access: '--risk-cred',
  lateral_movement: '--risk-lateral',
  exfiltration: '--risk-exfil',
}

/**
 * Renders hosts on a circle (deterministic layout by host id, so nodes
 * don't jump around between windows) with edges drawn as lines whose
 * thickness reflects byte volume. Deliberately not using a physics-based
 * force-graph library here -- for a fixed 12-host universe a stable
 * circular layout is easier to read at a glance than a force simulation
 * that resettles every time the window changes.
 */
export default function NetworkGraph({ nodes, edges, stage }) {
  const width = 560
  const height = 340
  const cx = width / 2
  const cy = height / 2 + 6
  const radius = 128

  const positions = useMemo(() => {
    const map = {}
    const sorted = [...nodes].sort((a, b) => a.id.localeCompare(b.id))
    sorted.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / sorted.length - Math.PI / 2
      map[n.id] = {
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
      }
    })
    return map
  }, [nodes])

  const maxBytes = Math.max(1, ...edges.map((e) => e.bytes))
  const stageColorVar = STAGE_COLOR_VAR[stage] || '--risk-benign'

  return (
    <svg className="network-graph-svg" viewBox={`0 0 ${width} ${height}`}>
      {edges.map((e, i) => {
        const a = positions[e.src]
        const b = positions[e.dst]
        if (!a || !b) return null
        const strokeWidth = 1 + 3 * (e.bytes / maxBytes)
        const isHot = e.bytes > 1_000_000 || e.syn_ratio > 0.6
        return (
          <line
            key={i}
            className={`edge-line${isHot ? ' edge-hot' : ''}`}
            x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            strokeWidth={strokeWidth}
            style={isHot ? { stroke: `var(${stageColorVar})`, opacity: 0.85 } : undefined}
          />
        )
      })}
      {nodes.map((n) => {
        const pos = positions[n.id]
        if (!pos) return null
        const roleClass = n.risk_role === 'attacker' ? 'role-attacker'
          : n.risk_role === 'victim' ? 'role-victim' : 'role-normal'
        return (
          <g key={n.id} className={`host-node ${roleClass}`} style={{ transform: `translate(${pos.x}px, ${pos.y}px)` }}>
            <circle r={n.risk_role ? 9 : 6.5} stroke="#0e1520" />
            <text textAnchor="middle" y={n.risk_role ? 24 : 20}>{n.id}</text>
          </g>
        )
      })}
    </svg>
  )
}
