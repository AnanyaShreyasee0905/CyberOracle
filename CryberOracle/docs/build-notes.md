# Build notes — read this before your pitch

Being straight about what's real vs. simplified will make you sound
MORE credible to judges, not less. Here's exactly what was built, what
was tested, and the honest answer to the questions you're likely to get.

## What's genuinely real (not mocked/faked)

- The world model is a real GNN (one-layer graph convolution, mean
  aggregation) + RNN, with **hand-written backpropagation verified
  against numerical gradients** (`ml/_gradcheck.py`, max relative error
  ~3.6e-07). This is not a stub — it actually learns from data via
  gradient descent.
- Training genuinely reduces loss (0.62 → 0.24) and the resulting model
  produces a risk trajectory that tracks the real (synthetic) attack
  stages closely: 22 of 24 windows in the demo episode have the
  rule-based MITRE stage matching ground truth, and the *learned* risk
  score climbs from ~3% to ~95% across the attack progression.
- The counterfactual simulator does a real forward pass on an edited
  graph through the same trained encoder — it is not a scripted
  before/after number.
- Every API endpoint was tested against a running server with curl,
  including error paths (400s for bad requests, 404 for out-of-range
  windows). The frontend was built for production (`npm run build`)
  with zero errors and its data contracts were cross-checked against
  the actual (not assumed) backend response shapes.

## Deliberate simplifications — and how to talk about them

**"Why synthetic data instead of CICIDS2017/a real PCAP?"**
Real intrusion datasets require a manual, gigabytes-sized download from
sites with click-through agreements — not something you can script in
a locked-down environment or often even venue wifi. The generator
produces statistically realistic flow aggregates (byte volume,
connection bursts, port diversity, SYN ratios) with a genuine multi-stage
attack embedded, so the full pipeline is demoable offline. The data
contract (see README's "Swapping in real data later") is designed so a
real-CSV adapter is a drop-in replacement, not a rewrite. Say this
plainly if asked — it's a reasonable engineering call under time
pressure, not something to hide.

**"Why NumPy instead of PyTorch/PyTorch-Geometric?"**
The hackathon sandbox this was built in had under 3GB of free disk —
nowhere near enough for a PyTorch install (which pulls multi-GB CUDA
dependencies even for CPU-only use in recent versions). A hand-written
NumPy implementation of a one-layer GCN + RNN is small enough to verify
correctness by hand (see the gradient check) and trains in seconds. If
you have PyTorch available on your own machine, swapping to
`torch_geometric.nn.SAGEConv` + `nn.GRU` is a natural upgrade path — the
data contract doesn't change.

**"Why a rule-based MITRE stage classifier instead of a learned one?"**
Kill-chain-labeled training data (flows tagged with a specific ATT&CK
stage) is scarce in the wild. A hybrid approach — learned risk score,
rule-based stage label — is a legitimate, auditable design choice: a
SOC analyst can read the exact rule that fired ("59 connection attempts
against a single port") rather than trust an opaque softmax. This is
presented as a scoping decision, not a limitation to hide.

**"Why does the rollout curve get smoothed?"**
A small RNN feeding its own predictions back into itself for 6+ steps
can produce a technically-valid but visually chaotic, oscillating
curve — a well-known compounding-error problem in self-feeding forecast
models. Two disclosed stabilizers are applied: blending each predicted
embedding with the previous one (persistence), and rate-limiting the
step-to-step change in the output risk score. Both are standard
short-horizon-forecasting techniques (similar in spirit to exponential
smoothing), not a way of hiding a broken model — the direction and
magnitude the model actually learned are preserved, just without
step-to-step noise a SOC analyst would never trust.

## The best demo moment (found during testing)

Run the counterfactual simulator at **window 9** (credential access
stage) with "isolate host" on the victim (`ws-03` in the default
episode) — predicted risk over the next 6 windows drops from ~81% to
~43%, a genuinely large and clean effect.

Then run it again at **window 13** (exfiltration already underway) —
the two curves barely diverge. This is a real, valuable finding, not a
cherry-picked result: **early intervention works, late intervention
doesn't**. This is a strong, honest closing point for the pitch — it's
exactly the kind of insight a "predict → simulate → defend" tool should
surface, and it came from the model's actual learned dynamics, not from
scripting the demo to look good.

## What was NOT built (cut for time, be upfront if asked)

- SGLang local AI analyst, n8n workflow automation, ElevenLabs voice
  alerts — these were explicitly scoped as stretch goals in the original
  plan and cut so the core predict/simulate/defend loop could be solid
  and fully tested instead of having five half-working integrations.
- Live PCAP ingestion — the pipeline consumes precomputed windows;
  wiring up `pyshark`/`scapy` for live capture is straightforward given
  the existing `flows_to_windows.py` contract but wasn't built.

## 2026-09-06 — CSE-CIC-IDS2018 Wednesday capture

Training now uses `data/02-14-2018.csv` (Benign / FTP-BruteForce /
SSH-Bruteforce) via `data/scripts/preprocess_cicids2018.py`. That script
cleans Inf/NaN rows, downsamples Benign 1:1 with attack flows, maps brute
force onto the existing `credential_access` stage (no new classes), and
projects the IP-less CIC-IDS2018 rows onto the existing host/edge
contract so `ml/train.py` and the FastAPI routes are unchanged. The
public API schema was not modified.
