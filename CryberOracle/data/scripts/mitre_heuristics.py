"""
Rule-based MITRE ATT&CK stage classifier + interpretability layer.

CYBER-ORACLE uses a HYBRID approach deliberately: the learned world model
predicts *risk* (a continuous infiltration probability + forecast), while
stage labeling and human-readable explanations come from transparent,
auditable rules over the same aggregate features. Kill-chain-labeled
training data is scarce in the wild, and a SOC analyst needs to be able
to see *why* a stage was assigned -- a rule they can read beats a
softmax score they have to trust blindly. This is a scoping decision,
not a limitation to hide: say so plainly if asked.

Each rule below maps directly onto network behaviors called out in the
project brief: unusual ports, TCP flag patterns (approximated here by
syn_ratio), connection bursts, timing anomalies, and new host
relationships.
"""

STAGES = ["benign", "reconnaissance", "credential_access", "lateral_movement", "exfiltration"]


def classify_window(window, seen_pairs_before: set, is_baseline_window: bool = False):
    """
    window: raw window dict (with 'edges') as produced by generate_synthetic_data.
    seen_pairs_before: set of (src,dst) host pairs observed in prior windows
                        of this episode -- used to flag "new host relationships".
    is_baseline_window: True for the very first window(s) used to seed
                         "known" host relationships. Every pair is trivially
                         "new" before any history exists, so the caller
                         should pass True here and skip the lateral-movement
                         rule until a baseline has been established --
                         otherwise window 0 always false-triggers.
    Returns (stage, explanations: list[str], new_pairs: set)
    """
    edges = window["edges"]
    explanations = []
    new_pairs = set()

    if not edges:
        return "benign", ["No active flows in this window."], new_pairs

    max_ports = max(f["ports"] for _, _, f in edges)
    max_conns = max(f["conns"] for _, _, f in edges)
    max_bytes = max(f["bytes"] for _, _, f in edges)
    max_syn = max(f["syn_ratio"] for _, _, f in edges)
    distinct_targets = len({dst for _, dst, _ in edges})

    def is_external(host):
        return host.startswith("ext-")

    for src, dst, _ in edges:
        pair = tuple(sorted((src, dst)))
        if pair not in seen_pairs_before:
            new_pairs.add(pair)
    # Recon from an external scanner naturally creates "new pairs" too --
    # that's a distinct pattern (see the reconnaissance rule below), so it
    # shouldn't also count as lateral movement between internal hosts.
    internal_new_pairs = {p for p in new_pairs if not any(is_external(h) for h in p)}

    # --- Exfiltration: very large one-way byte transfer, few connections ---
    if max_bytes > 1_000_000:
        explanations.append(f"Unusually large data transfer detected ({max_bytes/1e6:.1f} MB in one window).")
        if max_ports == 1:
            explanations.append("Transfer concentrated on a single destination port.")
        return "exfiltration", explanations, new_pairs

    # --- Reconnaissance: external source hitting many ports/hosts, high SYN ratio ---
    if distinct_targets >= 4 and max_ports >= 6 and max_syn > 0.6:
        explanations.append(f"Single source touched {distinct_targets} hosts across {int(max_ports)}+ distinct ports.")
        explanations.append(f"High SYN-only ratio ({max_syn:.0%}) typical of port scanning.")
        return "reconnaissance", explanations, new_pairs

    # --- Credential access: high connection count to one target, single port ---
    if max_conns >= 15 and max_ports <= 2:
        explanations.append(f"High connection burst ({int(max_conns)} attempts) against a single service port.")
        explanations.append("Pattern consistent with brute-force / credential-guessing behavior.")
        return "credential_access", explanations, new_pairs

    # --- Lateral movement: new INTERNAL host relationships appearing ---
    # Skipped on baseline windows -- with no history yet, every pair looks
    # "new" by definition, which would false-trigger on window 0.
    if not is_baseline_window and len(internal_new_pairs) >= 2:
        explanations.append(f"{len(internal_new_pairs)} new internal host-to-host relationships not seen earlier in this session.")
        explanations.append("Suggests a compromised host reaching out to peers it hasn't talked to before.")
        return "lateral_movement", explanations, new_pairs

    explanations.append("Traffic volume, port diversity, and connection patterns within normal range.")
    return "benign", explanations, new_pairs
