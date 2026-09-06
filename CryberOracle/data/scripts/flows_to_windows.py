"""
Turns each window's edge list into:
  - A: (N,N) weighted adjacency (log-scaled byte volume)
  - X: (N,F) per-host node features
  - A_norm: normalized adjacency ready for the GCN (see ml/model.py)

Node feature vector per host, per window (F=6):
  [out_degree, in_degree, max_single_edge_bytes(log), total_conns, distinct_ports_touched, max_syn_ratio]

Note: max_single_edge_bytes (not summed bytes) is deliberate -- an
exfiltration event is characteristically ONE outsized transfer. Summing
bytes across many small benign edges would dilute that spike; taking the
max preserves it as a clear, learnable signal.

This is the single data contract every other module depends on. If you
swap in real flow data (CICIDS2017 etc.), just produce this same
edge-list-per-window structure from your CSV and everything downstream
is unchanged.
"""
import numpy as np

F_DIM = 6


def build_host_index(hosts):
    return {h: i for i, h in enumerate(hosts)}


def window_to_tensors(window, host_index):
    N = len(host_index)
    A = np.zeros((N, N))
    out_deg = np.zeros(N)
    in_deg = np.zeros(N)
    max_bytes = np.zeros(N)
    total_conns = np.zeros(N)
    distinct_ports = np.zeros(N)
    # CIC-derived edges use host-p<port> destinations. Sets preserve the
    # actual distinct-port cardinality: a port node remains one even when many
    # sources contact it, while a source accumulates its destination ports.
    distinct_port_sets = [set() for _ in range(N)]
    max_syn = np.zeros(N)

    for src, dst, feats in window["edges"]:
        i, j = host_index[src], host_index[dst]
        weight = np.log1p(feats["bytes"])
        A[i, j] += weight
        A[j, i] += weight  # undirected for adjacency/GCN purposes
        out_deg[i] += 1
        in_deg[j] += 1
        max_bytes[i] = max(max_bytes[i], np.log1p(feats["bytes"]))
        max_bytes[j] = max(max_bytes[j], np.log1p(feats["bytes"]))
        total_conns[i] += feats["conns"]
        total_conns[j] += feats["conns"]
        if dst.startswith("host-p"):
            distinct_port_sets[i].add(dst)
            distinct_port_sets[j].add(dst)
        else:
            # Retain the original synthetic-data behavior.
            distinct_ports[i] = max(distinct_ports[i], feats["ports"])
            distinct_ports[j] = max(distinct_ports[j], feats["ports"])
        max_syn[i] = max(max_syn[i], feats["syn_ratio"])
        max_syn[j] = max(max_syn[j], feats["syn_ratio"])

    for index, ports in enumerate(distinct_port_sets):
        if ports:
            distinct_ports[index] = len(ports)
    X = np.stack([out_deg, in_deg, max_bytes, total_conns, distinct_ports, max_syn], axis=1)
    # Fixed (dataset-wide, not per-window) scaling constants so that
    # absolute magnitude differences BETWEEN windows are preserved --
    # this is exactly the signal that separates an exfiltration window
    # (huge total_bytes) from a benign one. Per-window max-normalization
    # would erase that signal by rescaling every window to its own peak
    # regardless of true magnitude -- do not "fix" this back to per-window
    # normalization.
    SCALE = np.array([10.0, 10.0, 12.0, 60.0, 20.0, 1.0])
    X = np.clip(X / SCALE, 0.0, 3.0)
    return A, X


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    A_hat = A + np.eye(A.shape[0])
    deg = A_hat.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return A_hat / deg


def episode_to_sequence(episode):
    """Returns list of (X_t, A_norm_t) tuples plus parallel metadata lists."""
    host_index = build_host_index(episode["hosts"])
    sequence = []
    stage_labels = []
    risk_labels = []
    raw_adjacency = []
    for window in episode["windows"]:
        A, X = window_to_tensors(window, host_index)
        A_norm = normalize_adjacency(A)
        sequence.append((X, A_norm))
        stage_labels.append(window["stage_label"])
        risk_labels.append(window["risk_label"])
        raw_adjacency.append(A)
    return sequence, stage_labels, risk_labels, raw_adjacency, host_index
