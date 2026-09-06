import React, { useEffect, useState, useCallback } from 'react'
import { api } from '../api/client'
import NetworkGraph from '../components/NetworkGraph'
import MitreStageBadge from '../components/MitreStageBadge'
import ExplanationPanel from '../components/ExplanationPanel'
import AnalystPanel from '../components/AnalystPanel'
import TrajectoryTimeline from '../components/TrajectoryTimeline'
import CounterfactualControls from '../components/CounterfactualControls'

export default function Dashboard() {
  const [episodeInfo, setEpisodeInfo] = useState(null)
  const [windows, setWindows] = useState([])
  const [windowId, setWindowId] = useState(0)
  const [graph, setGraph] = useState(null)
  const [forecast, setForecast] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.episodeInfo(), api.listWindows()])
      .then(([info, wins]) => {
        setEpisodeInfo(info)
        setWindows(wins)
        setWindowId(info.attack_start ?? 0)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const loadWindow = useCallback((wid) => {
    api.getGraph(wid).then(setGraph).catch((e) => setError(e.message))
    api.getRollout(wid, 6).then((r) => setForecast(r.risk_curve)).catch(() => setForecast([]))
  }, [])

  useEffect(() => {
    if (windows.length) loadWindow(windowId)
  }, [windowId, windows, loadWindow])

  if (loading) return <div className="loading">Loading CYBER-ORACLE dashboard…</div>
  if (error) return <div className="error-banner">Failed to reach the backend: {error}<br/>Is `uvicorn app.main:app` running on port 8000?</div>
  if (!graph) return <div className="loading">Loading window {windowId}…</div>

  return (
    <div className="app">
      <div className="topbar">
        <h1>CYBER-ORACLE</h1>
        <span className="subtitle">Predictive cyber defence — network world model</span>
        <span className="episode-tag">episode {episodeInfo.episode_id} · {episodeInfo.num_windows} windows</span>
      </div>

      <div className="main-grid">
        <div className="panel graph-panel">
          <h2>Network graph — window {windowId}</h2>
          <NetworkGraph nodes={graph.nodes} edges={graph.edges} stage={graph.heuristic_stage} />
          <div className="legend">
            <span><span className="legend-swatch" style={{ background: 'var(--accent)' }} />normal host</span>
            <span><span className="legend-swatch" style={{ background: 'var(--risk-lateral)' }} />likely victim</span>
            <span><span className="legend-swatch" style={{ background: 'var(--risk-exfil)' }} />external attacker</span>
          </div>
        </div>

        <div className="side-panel">
          <div className="panel">
            <h2>Predicted stage &amp; risk</h2>
            <MitreStageBadge stage={graph.heuristic_stage} />
            <div className="risk-readout">{(graph.predicted_risk * 100).toFixed(1)}%</div>
            <div className="risk-meter" aria-label={`Predicted risk ${(graph.predicted_risk * 100).toFixed(1)}%`}>
              <div className="risk-meter-fill" style={{ width: `${graph.predicted_risk * 100}%` }} />
            </div>
            <div className="risk-readout-label">infiltration probability, this window</div>
          </div>

          <div className="panel">
            <h2>Why this prediction</h2>
            <ExplanationPanel explanations={graph.explanations} />
          </div>

          <div className="panel analyst-panel">
            <h2>Analyst summary</h2>
            <AnalystPanel windowId={windowId} />
          </div>
        </div>

        <div className="panel timeline-panel">
          <h2>Attack trajectory &amp; forecast</h2>
          <div className="window-scrubber">
            <input
              type="range"
              min={0}
              max={windows.length - 1}
              value={windowId}
              onChange={(e) => setWindowId(Number(e.target.value))}
            />
            <span className="window-label">window {windowId} / {windows.length - 1}</span>
          </div>
          <TrajectoryTimeline windows={windows} currentWindowId={windowId} forecast={forecast} />
          <div className="legend">
            <span><span className="legend-swatch" style={{ background: '#5b8def' }} />observed risk</span>
            <span><span className="legend-swatch" style={{ background: '#e0a458' }} />K-step forecast</span>
          </div>

          <h2 style={{ marginTop: 26 }}>Counterfactual defence simulator</h2>
          <CounterfactualControls windowId={windowId} hosts={episodeInfo.hosts} currentGraph={graph} />
        </div>
      </div>
    </div>
  )
}
