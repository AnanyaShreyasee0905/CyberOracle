const BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json()
}

export const api = {
  episodeInfo: () => request('/api/episode-info'),
  listWindows: () => request('/api/windows'),
  getGraph: (windowId) => request(`/api/graph/${windowId}`),
  getAnalyst: (windowId) => request(`/api/analyst/${windowId}`),
  getRollout: (windowId, kSteps = 6) => request(`/api/rollout/${windowId}?k_steps=${kSteps}`),
  runCounterfactual: (payload) =>
    request('/api/counterfactual', { method: 'POST', body: JSON.stringify(payload) }),
}
