import React, { useEffect, useMemo, useState, useCallback } from 'react'
import { api } from '../api/client'
import NetworkGraph from '../components/NetworkGraph'
import MitreStageBadge from '../components/MitreStageBadge'
import ExplanationPanel from '../components/ExplanationPanel'
import AnalystPanel from '../components/AnalystPanel'
import TrajectoryTimeline from '../components/TrajectoryTimeline'
import CounterfactualControls from '../components/CounterfactualControls'

const LogoWordmark = () => (
  <>
    <span className="logo-text">CYBER</span>
    <span className="oracle-eye" aria-hidden="true">
      <span className="oracle-iris"><span className="oracle-glint" /></span>
      <span className="oracle-eyelid" />
    </span>
    <span className="logo-text">RACLE</span>
  </>
)

const Mascot = ({ activeCard }) => (
  <div className={`virus-mascot ${activeCard ? 'is-hunting' : ''}`} aria-hidden="true">
    <div className="virus-clone clone-one" />
    <div className="virus-clone clone-two" />
    <div className="virus-body">
      <span className="virus-spike spike-a" />
      <span className="virus-spike spike-b" />
      <span className="virus-spike spike-c" />
      <span className="virus-spike spike-d" />
      <span className="virus-eye eye-left" />
      <span className="virus-eye eye-right" />
      <span className="virus-mouth" />
    </div>
    <div className="virus-shadow" />
  </div>
)

export default function Dashboard() {
  const [episodeInfo, setEpisodeInfo] = useState(null)
  const [windows, setWindows] = useState([])
  const [windowId, setWindowId] = useState(0)
  const [graph, setGraph] = useState(null)
  const [forecast, setForecast] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeCard, setActiveCard] = useState(null)

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

  const currentWindow = useMemo(
    () => windows.find((win) => win.window_id === windowId),
    [windows, windowId],
  )

  const peakRisk = useMemo(
    () => windows.reduce((max, win) => Math.max(max, win.predicted_risk || 0), 0),
    [windows],
  )

  const dominantStage = useMemo(() => {
    const counts = {}
    windows.forEach((win) => {
      counts[win.heuristic_stage] = (counts[win.heuristic_stage] || 0) + 1
    })
    const entries = Object.entries(counts)
    if (!entries.length) return 'initializing'
    return entries.sort((a, b) => b[1] - a[1])[0][0]
  }, [windows])

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="circuit-stage" />
        <div className="oracle-logo is-loading">
          <LogoWordmark />
        </div>
        <p>Synchronizing threat telemetry...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="error-screen">
        <div className="oracle-logo">
          <LogoWordmark />
        </div>
        <p>Failed to reach the backend: {error}</p>
        <span>Start the API with `uvicorn app.main:app` on port 8000.</span>
      </div>
    )
  }

  if (!graph) return <div className="loading-screen"><p>Loading window {windowId}...</p></div>

  return (
    <div className="app" style={{ '--mascot-shift': activeCard ? 'clamp(16px, 8vw, 110px)' : '0px' }}>
      <div className="circuit-backdrop" aria-hidden="true">
        <iframe
          className="wallpaper-frame"
          src="/wallpaper-real-traces.html"
          title="CyberOracle animated circuit background"
          sandbox="allow-scripts"
        />
        <div className="motherboard-layer" />
        <svg className="circuit-current" viewBox="0 0 1200 900" preserveAspectRatio="none">
          <path d="M30 720 H260 V590 H430 V470 H650 V350 H860 V240 H1160" />
          <path d="M120 180 H300 V250 H470 V180 H720 V280 H980" />
          <path d="M0 430 H190 V360 H380 V420 H560 V560 H830 V640 H1200" />
          <path d="M1020 40 V170 H930 V330 H760 V520 H640 V810" />
          <path d="M200 860 V720 H340 V660 H520 V730 H760 V680 H1030" />
          <circle cx="260" cy="590" r="9" />
          <circle cx="650" cy="350" r="12" />
          <circle cx="470" cy="180" r="8" />
          <circle cx="830" cy="640" r="10" />
          <circle cx="930" cy="330" r="9" />
        </svg>
        <div className="clear-sweep" />
      </div>

      <header className="topbar">
        <a className="oracle-logo compact" href="#intro" aria-label="CyberOracle home">
          <LogoWordmark />
        </a>
        <span className="subtitle">Predictive cyber defence world model</span>
        <span className="episode-tag">episode {episodeInfo.episode_id} / {episodeInfo.num_windows} windows</span>
      </header>

      <Mascot activeCard={activeCard} />

      <section className="intro-section" id="intro">
        <div className="intro-copy">
          <div className="oracle-logo hero-logo">
            <LogoWordmark />
          </div>
          <p className="intro-kicker">neural defence console</p>
          <h1>Watch the attack path light up before it lands.</h1>
        </div>
        <div className="scroll-cue">
          <span />
          Scroll to arm the oracle
        </div>
      </section>

      <main className="dashboard-shell">
        <section className="command-strip reveal-card">
          <div>
            <span className="eyebrow">Current window</span>
            <strong>{windowId}</strong>
          </div>
          <div>
            <span className="eyebrow">Heuristic stage</span>
            <strong>{graph.heuristic_stage.replaceAll('_', ' ')}</strong>
          </div>
          <div>
            <span className="eyebrow">Peak episode risk</span>
            <strong>{(peakRisk * 100).toFixed(0)}%</strong>
          </div>
          <div>
            <span className="eyebrow">Dominant signal</span>
            <strong>{dominantStage.replaceAll('_', ' ')}</strong>
          </div>
        </section>

      <div className="main-grid">
        <div
          className="panel graph-panel reveal-card"
          onMouseEnter={() => setActiveCard('graph')}
          onMouseLeave={() => setActiveCard(null)}
        >
          <div className="panel-heading">
            <span className="eyebrow">Live topology</span>
            <h2>Network graph / window {windowId}</h2>
          </div>
          <NetworkGraph nodes={graph.nodes} edges={graph.edges} stage={graph.heuristic_stage} />
          <div className="legend">
            <span><span className="legend-swatch" style={{ background: 'var(--accent)' }} />normal host</span>
            <span><span className="legend-swatch" style={{ background: 'var(--risk-lateral)' }} />likely victim</span>
            <span><span className="legend-swatch" style={{ background: 'var(--risk-exfil)' }} />external attacker</span>
          </div>
        </div>

        <div className="side-panel">
          <div
            className="panel reveal-card"
            onMouseEnter={() => setActiveCard('risk')}
            onMouseLeave={() => setActiveCard(null)}
          >
            <div className="panel-heading">
              <span className="eyebrow">Threat pulse</span>
              <h2>Predicted stage &amp; risk</h2>
            </div>
            <MitreStageBadge stage={graph.heuristic_stage} />
            <div className="risk-readout">{(graph.predicted_risk * 100).toFixed(1)}%</div>
            <div className="risk-meter" aria-label={`Predicted risk ${(graph.predicted_risk * 100).toFixed(1)}%`}>
              <div className="risk-meter-fill" style={{ width: `${graph.predicted_risk * 100}%` }} />
            </div>
            <div className="risk-readout-label">
              infiltration probability, this window
              {currentWindow && <span> / truth: {currentWindow.ground_truth_stage.replaceAll('_', ' ')}</span>}
            </div>
          </div>

          <div
            className="panel reveal-card"
            onMouseEnter={() => setActiveCard('why')}
            onMouseLeave={() => setActiveCard(null)}
          >
            <div className="panel-heading">
              <span className="eyebrow">Model traces</span>
              <h2>Why this prediction</h2>
            </div>
            <ExplanationPanel explanations={graph.explanations} />
          </div>

          <div
            className="panel analyst-panel reveal-card"
            onMouseEnter={() => setActiveCard('analyst')}
            onMouseLeave={() => setActiveCard(null)}
          >
            <div className="panel-heading">
              <span className="eyebrow">Human-readable brief</span>
              <h2>Analyst summary</h2>
            </div>
            <AnalystPanel windowId={windowId} />
          </div>
        </div>

        <div
          className="panel timeline-panel reveal-card"
          onMouseEnter={() => setActiveCard('timeline')}
          onMouseLeave={() => setActiveCard(null)}
        >
          <div className="panel-heading">
            <span className="eyebrow">Future windows</span>
            <h2>Attack trajectory &amp; forecast</h2>
          </div>
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

          <div className="panel-heading cf-heading">
            <span className="eyebrow">Intervention lab</span>
            <h2>Counterfactual defence simulator</h2>
          </div>
          <CounterfactualControls windowId={windowId} hosts={episodeInfo.hosts} currentGraph={graph} />
        </div>
      </div>
      </main>
    </div>
  )
}
