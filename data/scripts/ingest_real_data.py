"""Convert CICIDS-style CSVs or PCAPs into CYBER-ORACLE's episode contract.

Examples:
    python ingest_real_data.py path\\to\\Friday-WorkingHours.pcap_ISCX.csv
    python ingest_real_data.py path\\to\\capture.pcap --output ../processed/real.json

CSV input uses only the standard library. PCAP input additionally requires
``scapy`` (``python -m pip install scapy``). The output is a one-episode JSON
list compatible with ``flows_to_windows.py`` and ``ml/train.py``.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


STAGE_RISK = {
    "benign": 0.03,
    "reconnaissance": 0.30,
    "credential_access": 0.55,
    "lateral_movement": 0.75,
    "exfiltration": 0.95,
}
STAGE_ORDER = {stage: index for index, stage in enumerate(STAGE_RISK)}


def _key(name):
    return " ".join(name.strip().lower().replace("_", " ").split())


def _field(row, *names, default=""):
    normalized = {_key(name): value for name, value in row.items()}
    for name in names:
        value = normalized.get(_key(name))
        if value not in (None, ""):
            return value
    return default


def _number(value, default=0.0):
    try:
        parsed = float(str(value).strip())
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _timestamp(value):
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        pass
    for pattern in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(text, pattern).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Unable to parse timestamp: {text!r}")


def _stage_for_label(label):
    text = str(label).strip().lower().replace("-", " ")
    if not text or text == "benign":
        return "benign"
    if "portscan" in text or "port scan" in text:
        return "reconnaissance"
    if any(term in text for term in ("patator", "brute force", "web attack")):
        return "credential_access"
    if "infiltration" in text:
        return "lateral_movement"
    if any(term in text for term in ("exfil", "heartbleed")):
        return "exfiltration"
    return "benign"


def _new_aggregate():
    return {"bytes": 0.0, "conns": 0, "ports": set(), "syn": 0.0, "tcp_packets": 0.0}


def _add(aggregates, bucket, src, dst, port, byte_count, connections, syn_count, tcp_packets):
    edge = aggregates[bucket][(str(src), str(dst))]
    edge["bytes"] += max(0.0, byte_count)
    edge["conns"] += max(0, int(connections))
    if port is not None:
        edge["ports"].add(int(port))
    edge["syn"] += max(0.0, syn_count)
    edge["tcp_packets"] += max(0.0, tcp_packets)


def _read_csv(path, window_seconds):
    aggregates = defaultdict(lambda: defaultdict(_new_aggregate))
    stages = defaultdict(Counter)
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                when = _timestamp(_field(row, "Timestamp"))
                src = _field(row, "Source IP", "Src IP")
                dst = _field(row, "Destination IP", "Dst IP")
                if not src or not dst:
                    raise ValueError("missing source or destination IP")
            except ValueError as error:
                raise ValueError(f"CSV row {row_number}: {error}") from error
            bucket = int(when // window_seconds)
            fwd_packets = _number(_field(row, "Total Fwd Packets", "Fwd Packets"))
            back_packets = _number(_field(row, "Total Backward Packets", "Bwd Packets"))
            _add(
                aggregates, bucket, src, dst,
                _number(_field(row, "Destination Port", "Dst Port"), 0),
                _number(_field(row, "Total Length of Fwd Packets", "Total Length of Fwd Packet"))
                + _number(_field(row, "Total Length of Bwd Packets", "Total Length of Bwd Packet")),
                1,
                _number(_field(row, "SYN Flag Count")),
                fwd_packets + back_packets,
            )
            stages[bucket][_stage_for_label(_field(row, "Label"))] += 1
    return aggregates, stages


def _read_pcap(path, window_seconds):
    try:
        from scapy.all import IP, IPv6, PcapReader, TCP, UDP
    except ImportError as error:
        raise RuntimeError("PCAP input requires scapy; install it with: python -m pip install scapy") from error

    aggregates = defaultdict(lambda: defaultdict(_new_aggregate))
    stages = defaultdict(Counter)
    flow_keys = defaultdict(lambda: defaultdict(set))
    with PcapReader(str(path)) as packets:
        for packet in packets:
            if IP in packet:
                network = packet[IP]
            elif IPv6 in packet:
                network = packet[IPv6]
            else:
                continue
            when = float(packet.time)
            bucket = int(when // window_seconds)
            protocol, sport, dport, syn, tcp_packet = "ip", 0, 0, 0, 0
            if TCP in packet:
                protocol, sport, dport = "tcp", int(packet[TCP].sport), int(packet[TCP].dport)
                syn, tcp_packet = int(bool(int(packet[TCP].flags) & 0x02)), 1
            elif UDP in packet:
                protocol, sport, dport = "udp", int(packet[UDP].sport), int(packet[UDP].dport)
            key = (str(network.src), str(network.dst))
            flow_key = (protocol, sport, dport)
            is_new_flow = flow_key not in flow_keys[bucket][key]
            flow_keys[bucket][key].add(flow_key)
            _add(aggregates, bucket, key[0], key[1], dport, len(packet), int(is_new_flow), syn, tcp_packet)
            stages[bucket]["benign"] += 1
    return aggregates, stages


def _episode(episode_id, aggregates, stages):
    if not aggregates:
        raise ValueError("No IP flow records found in the input.")
    hosts = sorted({host for edges in aggregates.values() for pair in edges for host in pair})
    windows = []
    first_bucket = min(aggregates)
    for window_id, source_bucket in enumerate(range(first_bucket, max(aggregates) + 1)):
        edges = []
        for (src, dst), values in sorted(aggregates[source_bucket].items()):
            tcp_packets = values["tcp_packets"]
            edges.append([src, dst, {
                "bytes": round(values["bytes"], 3),
                "conns": values["conns"],
                "ports": len(values["ports"]),
                "syn_ratio": round(values["syn"] / tcp_packets, 6) if tcp_packets else 0.0,
            }])
        stage = max(stages[source_bucket] or {"benign": 1}, key=lambda name: STAGE_ORDER[name])
        windows.append({"window_id": window_id, "edges": edges, "stage_label": stage, "risk_label": STAGE_RISK[stage]})
    attack_start = next((window["window_id"] for window in windows if window["stage_label"] != "benign"), None)
    return {
        "episode_id": episode_id,
        "hosts": hosts,
        "attack": attack_start is not None,
        "attack_start": attack_start,
        "victim": None,
        "windows": windows,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert CICIDS CSV or PCAP data into CYBER-ORACLE episode JSON.")
    parser.add_argument("input", type=Path, help="CICIDS-style .csv, .pcap, or .pcapng file")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / ".." / "processed" / "real_episode.json")
    parser.add_argument("--window-seconds", type=int, default=10)
    args = parser.parse_args()
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")
    suffix = args.input.suffix.lower()
    if suffix == ".csv":
        aggregates, stages = _read_csv(args.input, args.window_seconds)
    elif suffix in {".pcap", ".pcapng"}:
        aggregates, stages = _read_pcap(args.input, args.window_seconds)
    else:
        parser.error("input must end in .csv, .pcap, or .pcapng")
    episode = _episode(args.input.stem, aggregates, stages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([episode], indent=2), encoding="utf-8")
    print(f"Wrote {len(episode['windows'])} windows for {len(episode['hosts'])} hosts -> {args.output}")


if __name__ == "__main__":
    main()
