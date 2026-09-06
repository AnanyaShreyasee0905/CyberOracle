# CYBER-ORACLE judge demo — 60–90 seconds

**Prep:** Start the backend and frontend before judges arrive. Open the dashboard
at the attack-start window (window 9 in the default episode).

## Spoken script (~75 seconds)

“CYBER-ORACLE is a predictive cyber-defence console. Instead of only alerting
after compromise, it models each 10-second slice of network traffic as a graph,
forecasts the next few slices, and lets an analyst test a response first.

This replay is a synthetic, multi-stage attack. At window 9, risk is climbing
and the ATT&CK badge identifies credential access. The graph and explanation
panel show why: a connection burst against a single port. The trajectory is the
model’s six-step risk forecast, not just a current alert.

Now I’ll isolate the likely victim and simulate the action. Notice the before
and after curves: predicted risk falls from about 81% to 43%. This is the core
differentiator—test a containment decision before disrupting a real host.

Move to window 13, when exfiltration is underway, and run the same action. The
curves barely change. The useful finding is not just that an attack exists; it’s
that early intervention helps and late intervention may not.

The analyst panel turns the same risk, stage, and evidence into a concise
summary. For this demo it is deliberately template-based, not an LLM claim.
The replay is synthetic for a reliable offline demo, but the same ingestion path
also accepts CICIDS-style CSVs or PCAPs and emits the identical window graph
format. Predict, simulate, defend—before compromise spreads.”

## Speaker notes / likely questions

- **Data:** The default demo uses synthetic flow aggregates so it is small,
  deterministic, and works offline. `ingest_real_data.py` supports CICIDS-style
  CSV and PCAP/PCAPNG input (PCAP requires Scapy) and emits the same
  `{bytes, conns, ports, syn_ratio}` edge contract.
- **Model:** The risk model is a hand-written NumPy one-layer GCN plus RNN;
  its backpropagation was checked numerically. PyTorch was intentionally not
  used for the hackathon’s disk constraints.
- **MITRE and analyst text:** ATT&CK stages are auditable threshold rules. The
  analyst panel is a clearly labeled template fallback; SGLang is not included.
- **Counterfactual:** The before/after result is a real forward pass over the
  edited graph, not a hard-coded response. It forecasts impact; it does not
  execute containment on a live network.
- **If asked about smoothing:** Rollout uses persistence blending and rate
  limiting so a six-step self-feeding forecast stays readable; this is disclosed,
  not hidden.
