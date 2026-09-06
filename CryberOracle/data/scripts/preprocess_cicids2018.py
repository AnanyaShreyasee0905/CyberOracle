"""Create CIC-IDS2018-only world-model episodes.

The source CSV does not expose endpoint IP addresses. This module projects
each retained destination port to ``host-p<port>``. Attack flows originate at
``ext-attacker``; benign flows are assigned reproducibly to a small workstation
pool with a stable content hash. Raw-flow reduction is vectorized; the only
Python iteration is over already aggregated window/edge records.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = DATA_DIR / "02-14-2018.csv" / "02-14-2018.csv"
DEFAULT_OUTPUT = DATA_DIR / "processed" / "episodes.json"
TOP_K, WINDOW_SECONDS, WINDOWS_PER_EPISODE = 16, 60, 24
WORKSTATIONS = [f"ws-{index:02d}" for index in range(1, 9)]
ATTACKER = "ext-attacker"
RISK = {"benign": 0.03, "credential_access": 0.55}
USED_COLUMNS = ["Dst Port", "Timestamp", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "SYN Flag Cnt", "Label"]
NUMERIC_COLUMNS = ["Dst Port", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "SYN Flag Cnt"]
ATTACK_LABELS = {"FTP-BruteForce", "SSH-Bruteforce"}


def read_flows(path: Path, seed: int) -> tuple[pd.DataFrame, list[int], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    frame = pd.read_csv(path, usecols=lambda col: str(col).strip() in USED_COLUMNS,
                        encoding="utf-8-sig", dtype={"Timestamp": "string", "Label": "string"},
                        na_values=["", "NaN", "nan", "Infinity", "-Infinity", "Inf", "-Inf"], low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce", downcast="float")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    valid = frame["Label"].isin(["Benign", *ATTACK_LABELS]) & frame["Timestamp"].notna()
    valid &= frame[NUMERIC_COLUMNS].notna().all(axis=1)
    valid &= np.isfinite(frame[NUMERIC_COLUMNS].to_numpy(dtype=np.float64)).all(axis=1)
    frame = frame.loc[valid].copy()
    frame["Dst Port"] = frame["Dst Port"].astype(np.int32)
    top_ports = frame["Dst Port"].value_counts().head(TOP_K).index.astype(int).tolist()
    original_attack = int(frame["Label"].isin(ATTACK_LABELS).sum())
    frame = frame.loc[frame["Dst Port"].isin(top_ports)].copy()
    surviving_attack = int(frame["Label"].isin(ATTACK_LABELS).sum())
    if surviving_attack != original_attack:
        raise RuntimeError(f"Top-{TOP_K} removed {original_attack - surviving_attack} attacks; refusing to violate retention.")
    attack = frame.loc[frame["Label"].isin(ATTACK_LABELS)].copy()
    benign = frame.loc[frame["Label"].eq("Benign")]
    if len(benign) < len(attack):
        raise RuntimeError("Not enough top-K benign flows for a 1:1 downsample.")
    frame = pd.concat([attack, benign.sample(n=len(attack), random_state=seed)], ignore_index=True)
    frame["stage"] = np.where(frame["Label"].eq("Benign"), "benign", "credential_access")
    # Content hashing avoids row-order modulo artifacts at time/window boundaries.
    hash_columns = ["Timestamp", "Dst Port", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "SYN Flag Cnt"]
    hashes = pd.util.hash_pandas_object(frame[hash_columns], index=False).to_numpy(dtype=np.uint64)
    ws = np.asarray(WORKSTATIONS, dtype=object)[hashes % len(WORKSTATIONS)]
    frame["src"] = np.where(frame["stage"].eq("credential_access"), ATTACKER, ws)
    frame["dst"] = "host-p" + frame["Dst Port"].astype(str)
    # Use pandas' datetime-aware floor instead of assuming an internal
    # nanosecond representation (pandas 3 may use microseconds).
    frame["bucket"] = frame["Timestamp"].dt.floor(f"{WINDOW_SECONDS}s")
    frame["bytes"] = (frame["TotLen Fwd Pkts"] + frame["TotLen Bwd Pkts"]).clip(lower=0)
    frame["packets"] = (frame["Tot Fwd Pkts"] + frame["Tot Bwd Pkts"]).clip(lower=0)
    frame["syn"] = frame["SYN Flag Cnt"].clip(lower=0)
    counts = {"benign": int((frame["stage"] == "benign").sum()), "credential_access": int((frame["stage"] == "credential_access").sum())}
    return frame, top_ports, counts


def aggregate_windows(frame: pd.DataFrame) -> tuple[dict[int, list], dict[int, str]]:
    grouped = frame.groupby(["bucket", "src", "dst", "stage"], sort=True, observed=True).agg(
        bytes=("bytes", "sum"), conns=("bytes", "size"), syn=("syn", "sum"), packets=("packets", "sum")).reset_index()
    grouped["syn_ratio"] = np.divide(grouped["syn"], grouped["packets"], out=np.zeros(len(grouped), dtype=float), where=grouped["packets"].gt(0)).clip(0, 1)
    edges_by_bucket = {}
    for bucket, rows in grouped.groupby("bucket", sort=True, observed=True):
        edges_by_bucket[int(bucket.value)] = [[row.src, row.dst, {"bytes": round(float(row.bytes), 3), "conns": int(row.conns), "ports": 1, "syn_ratio": round(float(row.syn_ratio), 6)}] for row in rows.itertuples(index=False)]
    stages = frame.groupby("bucket", sort=True, observed=True)["stage"].agg(lambda values: "credential_access" if (values == "credential_access").any() else "benign").to_dict()
    return edges_by_bucket, {int(key.value): value for key, value in stages.items()}


def build_episodes(edges_by_bucket: dict[int, list], stages: dict[int, str], top_ports: list[int]):
    timeline = [{"window_id": i, "edges": edges_by_bucket[bucket], "stage_label": stages[bucket], "risk_label": RISK[stages[bucket]]} for i, bucket in enumerate(sorted(edges_by_bucket))]
    hosts = [f"host-p{port}" for port in top_ports] + WORKSTATIONS + [ATTACKER]
    episodes = []
    for start in range(0, len(timeline) - WINDOWS_PER_EPISODE + 1, WINDOWS_PER_EPISODE):
        windows = [{**window, "window_id": offset} for offset, window in enumerate(timeline[start:start + WINDOWS_PER_EPISODE])]
        attack_start = next((w["window_id"] for w in windows if w["stage_label"] == "credential_access"), None)
        targets = Counter(dst for w in windows if w["stage_label"] == "credential_access" for src, dst, _ in w["edges"] if src == ATTACKER)
        episodes.append({"episode_id": f"cic2018-{len(episodes):03d}", "hosts": hosts, "attack": attack_start is not None, "attack_start": attack_start, "victim": targets.most_common(1)[0][0] if targets else None, "windows": windows})
    # train.py selects the first attack episode; choose a visible baseline-to-attack transition for the dashboard.
    episodes.sort(key=lambda ep: (0 if ep["attack"] and ep["attack_start"] not in (None, 0) else 1, abs((ep["attack_start"] or 0) - 6), -sum(w["stage_label"] == "credential_access" for w in ep["windows"])))
    for index, episode in enumerate(episodes):
        episode["episode_id"] = f"{'attack' if episode['attack'] else 'benign'}-{index:03d}"
    return episodes, len(timeline)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess CIC-IDS2018 into CIC-only episodes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame, top_ports, class_counts = read_flows(args.input, args.seed)
    edges_by_bucket, stages = aggregate_windows(frame)
    episodes, total_windows = build_episodes(edges_by_bucket, stages, top_ports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(episodes), encoding="utf-8")
    node_counts = [len({host for edge in window["edges"] for host in edge[:2]}) for ep in episodes for window in ep["windows"]]
    fixed_n = TOP_K + len(WORKSTATIONS) + 1
    print(f"top-{TOP_K} ports: {top_ports}; fixed host graph N={fixed_n}")
    print(f"total windows produced: {total_windows}; complete episodes: {len(episodes)}")
    print(f"flow class distribution after downsampling: {class_counts}")
    print(f"active node-count range across emitted windows: {min(node_counts)}-{max(node_counts)} (fixed N={fixed_n})")
    print(f"wrote CIC-only episodes -> {args.output}")


if __name__ == "__main__":
    main()
