"""
Trains the WorldModel on every synthetic episode, then precomputes a
fully enriched "demo episode" (predictions + MITRE stages + explanations
baked in) that the FastAPI backend loads at startup for an instant,
reliable demo -- no live training or flaky first-request latency during
judging.

Run from the ml/ directory:
    python3 train.py
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "scripts"))

from model import WorldModel  # noqa: E402
from flows_to_windows import episode_to_sequence, F_DIM  # noqa: E402
from mitre_heuristics import classify_window  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
# Produced by data/scripts/preprocess_cicids2018.py (same JSON contract as the
# synthetic generator). Retrain after re-running that script.
EPISODES_PATH = os.path.join(ROOT, "data", "processed", "episodes.json")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "world_model.npz")
DEMO_EPISODE_OUT = os.path.join(ROOT, "data", "processed", "demo_episode.json")

GCN_HIDDEN = 10
RNN_HIDDEN = 8
EPOCHS = 250
LR = 0.02
GRAD_CLIP = 2.0


def load_episodes():
    with open(EPISODES_PATH) as f:
        return json.load(f)


def clip_grads(grads, max_norm):
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for k in grads:
            grads[k] *= scale
    return grads


def train():
    episodes = load_episodes()
    prepared = []
    for ep in episodes:
        seq, stages, risks, raw_adj, host_index = episode_to_sequence(ep)
        prepared.append((seq, risks))

    model = WorldModel(F_DIM, GCN_HIDDEN, RNN_HIDDEN, seed=7)
    # Decay Wh so the recurrence can't dominate the graph-input pathway --
    # see adam_step() docstring. Gradient clipping keeps the small RNN
    # from blowing up into saturated, input-insensitive territory, which
    # is what was making the counterfactual simulator a no-op earlier.
    weight_decay = {"Wh": 0.04, "Wg": 0.01}

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        order = np.random.default_rng(epoch).permutation(len(prepared))
        for idx in order:
            seq, risks = prepared[idx]
            outputs, caches = model.forward_sequence(seq)
            grads, loss = model.backward(outputs, caches, risks)
            clip_grads(grads, GRAD_CLIP)
            model.adam_step(grads, lr=LR, weight_decay=weight_decay)
            epoch_loss += loss
        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  avg_loss={epoch_loss/len(prepared):.4f}")

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    model.save(CHECKPOINT_PATH)
    print(f"Saved checkpoint -> {CHECKPOINT_PATH}")
    return model, episodes


def pick_demo_episode(episodes):
    for ep in episodes:
        if ep["attack"] and ep["attack_start"] is not None:
            return ep
    raise RuntimeError("No attack episode found in dataset")


def precompute_demo_episode(model: WorldModel, episode: dict):
    seq, stages, risks, raw_adj, host_index = episode_to_sequence(episode)
    outputs, caches = model.forward_sequence(seq)

    seen_pairs = set()
    BASELINE_WINDOWS = 3  # first few windows just establish "known" traffic patterns
    enriched_windows = []
    for t, window in enumerate(episode["windows"]):
        is_baseline = t < BASELINE_WINDOWS
        heuristic_stage, explanations, new_pairs = classify_window(window, seen_pairs, is_baseline)
        seen_pairs |= {tuple(sorted((s, d))) for s, d, _ in window["edges"]}

        enriched_windows.append({
            "window_id": t,
            "edges": window["edges"],
            "ground_truth_stage": window["stage_label"],
            "heuristic_stage": heuristic_stage,
            "explanations": explanations,
            "predicted_risk": float(outputs["risk"][t]),
            "graph_embedding": outputs["g"][t].tolist(),
            "hidden_state": outputs["h"][t].tolist(),
            "hidden_state_prev": caches[t]["h_prev"].tolist(),
        })

    demo = {
        "episode_id": episode["episode_id"],
        "hosts": episode["hosts"],
        "attack_start": episode["attack_start"],
        "victim": episode["victim"],
        "windows": enriched_windows,
        "model_config": {"gcn_hidden": GCN_HIDDEN, "rnn_hidden": RNN_HIDDEN, "feat_dim": F_DIM},
    }
    os.makedirs(os.path.dirname(DEMO_EPISODE_OUT), exist_ok=True)
    with open(DEMO_EPISODE_OUT, "w") as f:
        json.dump(demo, f)
    print(f"Saved enriched demo episode ({episode['episode_id']}) -> {DEMO_EPISODE_OUT}")


if __name__ == "__main__":
    model, episodes = train()
    demo_ep = pick_demo_episode(episodes)
    precompute_demo_episode(model, demo_ep)
