"""
Generates synthetic time-windowed network episodes for CYBER-ORACLE.

Why synthetic data instead of a downloaded PCAP/CICIDS2017 CSV?
Real intrusion datasets (CICIDS2017, UNSW-NB15) require manual download
from sites that need a browser/agreement click-through and are gigabytes
in size -- not something that can be fetched from a locked-down sandbox
or, often, a hackathon venue's wifi. This generator produces
statistically realistic flow aggregates for a fixed host universe, with
an embedded multi-stage attack (recon -> credential access -> lateral
movement -> exfiltration), so the whole pipeline is demoable offline
with zero downloads.

DROP-IN REPLACEMENT: if you have real flow-level CSVs (CICIDS2017 etc.),
write an adapter that produces the same per-window record schema below
(see build_windows_from_dataframe in flows_to_windows.py) and everything
downstream -- training, rollout, counterfactual, dashboard -- works
unchanged. The node/edge feature contract is the important part, not
where the numbers came from.

Output: data/processed/episodes.json
    A list of episodes. Each episode is a dict:
      {
        "episode_id": str,
        "hosts": [host_id, ...]                      # fixed node universe
        "attack": bool,
        "attack_start": int | null,
        "windows": [
          {
            "window_id": int,
            "edges": [[src, dst, {"bytes":.., "conns":.., "ports":.., "syn_ratio":..}], ...],
            "stage_label": "benign"|"reconnaissance"|"credential_access"|"lateral_movement"|"exfiltration",
            "risk_label": float in [0,1]     # supervised training target
          }, ...
        ]
      }
"""
import json
import os
import numpy as np

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "processed", "episodes.json")

# Fixed host universe used across every episode (keeps graph size constant,
# which keeps the GCN's adjacency dimensions constant -- a deliberate
# hackathon-speed simplification; see docs/architecture.md).
INTERNAL_HOSTS = [f"ws-{i:02d}" for i in range(1, 9)] + ["srv-db", "srv-web", "srv-file"]
ATTACKER_HOST = "ext-attacker"
ALL_HOSTS = INTERNAL_HOSTS + [ATTACKER_HOST]

T_WINDOWS = 24
STAGE_RISK = {
    "benign": 0.03,
    "reconnaissance": 0.30,
    "credential_access": 0.55,
    "lateral_movement": 0.75,
    "exfiltration": 0.95,
}


def _make_topology(rng, n_pairs=10):
    """A FIXED set of host pairs that represent an episode's 'normal'
    communication patterns (e.g. workstations always talking to the same
    file/db/web servers). This is what makes the lateral-movement
    heuristic ("new host relationship") meaningful -- without a stable
    baseline topology, every window would contain fresh random pairs and
    'new relationship' would be meaningless noise."""
    pairs = set()
    while len(pairs) < n_pairs:
        a, b = rng.choice(INTERNAL_HOSTS, size=2, replace=False)
        pairs.add(tuple(sorted((a, b))))
    return list(pairs)


def _benign_edges(rng, topology, n_edges=8):
    edges = []
    chosen = rng.choice(len(topology), size=min(n_edges, len(topology)), replace=False)
    for idx in chosen:
        a, b = topology[idx]
        edges.append([a, b, {
            "bytes": float(rng.uniform(2_000, 50_000)),
            "conns": int(rng.integers(1, 4)),
            "ports": int(rng.integers(1, 2)),
            "syn_ratio": float(rng.uniform(0.05, 0.2)),
        }])
    return edges


def _recon_edges(rng, attacker):
    # attacker touches many hosts on many ports, low bytes, high syn_ratio
    targets = rng.choice(INTERNAL_HOSTS, size=6, replace=False)
    edges = []
    for tgt in targets:
        edges.append([attacker, tgt, {
            "bytes": float(rng.uniform(50, 500)),
            "conns": int(rng.integers(3, 8)),
            "ports": int(rng.integers(8, 20)),
            "syn_ratio": float(rng.uniform(0.7, 0.95)),
        }])
    return edges


def _cred_access_edges(rng, attacker, victim):
    # repeated auth attempts against one host: high conns, single port, low bytes
    return [[attacker, victim, {
        "bytes": float(rng.uniform(500, 3000)),
        "conns": int(rng.integers(20, 60)),
        "ports": int(1),
        "syn_ratio": float(rng.uniform(0.4, 0.6)),
    }]]


def _lateral_edges(rng, victim, exclude=frozenset()):
    # compromised host starts talking to peers it never talked to before
    candidates = [h for h in INTERNAL_HOSTS if h != victim and h not in exclude]
    new_peers = rng.choice(candidates, size=min(3, len(candidates)), replace=False)
    edges = []
    for peer in new_peers:
        edges.append([victim, peer, {
            "bytes": float(rng.uniform(5_000, 20_000)),
            "conns": int(rng.integers(2, 6)),
            "ports": int(rng.integers(1, 3)),
            "syn_ratio": float(rng.uniform(0.2, 0.4)),
        }])
    return edges


def _exfil_edges(rng, victim, attacker):
    return [[victim, attacker, {
        "bytes": float(rng.uniform(2_000_000, 8_000_000)),
        "conns": int(rng.integers(1, 3)),
        "ports": int(1),
        "syn_ratio": float(rng.uniform(0.02, 0.1)),
    }]]


def make_episode(rng, episode_id: str, attack: bool):
    windows = []
    attack_start = int(rng.integers(6, 12)) if attack else None
    victim = str(rng.choice(INTERNAL_HOSTS)) if attack else None
    topology = _make_topology(rng)
    topology_hosts_of_victim = {p[0] if p[1] == victim else p[1] for p in topology if victim in p} if attack else set()

    for w in range(T_WINDOWS):
        edges = _benign_edges(rng, topology)
        stage = "benign"

        if attack:
            rel = w - attack_start
            if 0 <= rel <= 1:
                edges += _recon_edges(rng, ATTACKER_HOST)
                stage = "reconnaissance"
            elif 2 <= rel <= 3:
                edges += _cred_access_edges(rng, ATTACKER_HOST, victim)
                stage = "credential_access"
            elif 4 <= rel <= 5:
                edges += _lateral_edges(rng, victim, exclude=topology_hosts_of_victim)
                stage = "lateral_movement"
            elif 6 <= rel <= 8:
                edges += _exfil_edges(rng, victim, ATTACKER_HOST)
                stage = "exfiltration"
            elif rel > 8:
                # smoldering elevated risk after compromise, slowly cooling
                stage = "exfiltration" if rel <= 10 else "benign"

        windows.append({
            "window_id": w,
            "edges": edges,
            "stage_label": stage,
            "risk_label": STAGE_RISK[stage],
        })

    return {
        "episode_id": episode_id,
        "hosts": ALL_HOSTS,
        "attack": attack,
        "attack_start": attack_start,
        "victim": victim,
        "windows": windows,
    }


def generate(n_attack_episodes=24, n_benign_episodes=10, seed=42):
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(n_attack_episodes):
        episodes.append(make_episode(rng, f"attack-{i:03d}", attack=True))
    for i in range(n_benign_episodes):
        episodes.append(make_episode(rng, f"benign-{i:03d}", attack=False))
    rng.shuffle(episodes)
    return episodes


if __name__ == "__main__":
    episodes = generate()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(episodes, f)
    n_attack = sum(1 for e in episodes if e["attack"])
    print(f"Wrote {len(episodes)} episodes ({n_attack} attack, {len(episodes)-n_attack} benign) -> {OUT_PATH}")
