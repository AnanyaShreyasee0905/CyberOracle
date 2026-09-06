"""
Counterfactual Defence Simulator.

The novelty of CYBER-ORACLE: given the network's state at a chosen
window, edit the graph as if a defensive action had just been taken
(isolate a host, block a specific communication path), re-embed that
edited graph with the SAME trained encoder, and re-run the K-step
rollout from there. Comparing the two rollouts (do nothing vs.
intervene) answers "does this action actually reduce predicted risk
before I commit to it in real life?"

Supported actions:
    {"type": "isolate_host", "host": "<host_id>"}
        Removes every edge touching that host for this window.
    {"type": "block_edge", "src": "<host_id>", "dst": "<host_id>"}
        Removes only that specific communication path.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "scripts"))
from flows_to_windows import build_host_index, window_to_tensors, normalize_adjacency  # noqa: E402
from rollout import k_step_rollout  # noqa: E402


class InvalidAction(ValueError):
    pass


def _apply_action(edges, action):
    if action["type"] == "isolate_host":
        host = action["host"]
        return [e for e in edges if e[0] != host and e[1] != host]
    if action["type"] == "block_edge":
        src, dst = action["src"], action["dst"]
        pair = {src, dst}
        return [e for e in edges if {e[0], e[1]} != pair]
    raise InvalidAction(f"Unknown action type: {action.get('type')}")


def simulate_counterfactual(model, hosts, window, hidden_state_prev, action, k_steps=6):
    """
    hosts: list of host ids (fixed universe for this episode)
    window: the raw window dict (with 'edges') the action is applied at
    hidden_state_prev: the model's hidden state h_(t-1), i.e. BEFORE this
                  window -- both the baseline and counterfactual hidden
                  state at t are recomputed from here using the (edited
                  or original) graph embedding, so the intervention
                  actually changes h_t itself, not just the rollout's
                  starting point.
    action: dict, see module docstring
    Returns: {"baseline_risk_curve": [...], "counterfactual_risk_curve": [...],
              "edges_removed": int}
    """
    host_index = build_host_index(hosts)
    h_prev = np.array(hidden_state_prev)
    p = model.params

    def encode_and_advance(edges):
        A, X = window_to_tensors({**window, "edges": edges}, host_index)
        g = model.encode_window(X, normalize_adjacency(A))
        h = np.tanh(p["Wx"] @ g + p["Wh"] @ h_prev + p["bh"])
        return g, h

    g_base, h_base = encode_and_advance(window["edges"])
    baseline_curve = k_step_rollout(model, h_base, g_base, k_steps)

    edited_edges = _apply_action(window["edges"], action)
    g_cf, h_cf = encode_and_advance(edited_edges)
    counterfactual_curve = k_step_rollout(model, h_cf, g_cf, k_steps)

    return {
        "baseline_risk_curve": baseline_curve,
        "counterfactual_risk_curve": counterfactual_curve,
        "edges_removed": len(window["edges"]) - len(edited_edges),
    }
