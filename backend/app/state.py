"""
Loads the trained WorldModel and the precomputed demo episode once, at
process startup, and exposes them to the routers. Keeping this in one
place means the (cheap, but non-zero) model load happens exactly once,
not per-request.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ML_DIR = os.path.join(ROOT, "ml")
DATA_SCRIPTS_DIR = os.path.join(ROOT, "data", "scripts")
sys.path.insert(0, ML_DIR)
sys.path.insert(0, DATA_SCRIPTS_DIR)

from model import WorldModel  # noqa: E402
from flows_to_windows import build_host_index  # noqa: E402

CHECKPOINT_PATH = os.path.join(ML_DIR, "checkpoints", "world_model.npz")
DEMO_EPISODE_PATH = os.path.join(ROOT, "data", "processed", "demo_episode.json")


class AppState:
    def __init__(self):
        if not os.path.exists(CHECKPOINT_PATH):
            raise RuntimeError(
                f"No trained model found at {CHECKPOINT_PATH}. "
                f"Run `python3 ml/train.py` from the project root first."
            )
        if not os.path.exists(DEMO_EPISODE_PATH):
            raise RuntimeError(
                f"No precomputed demo episode found at {DEMO_EPISODE_PATH}. "
                f"Run `python3 ml/train.py` from the project root first."
            )
        self.model = WorldModel.load(CHECKPOINT_PATH)
        with open(DEMO_EPISODE_PATH) as f:
            self.demo_episode = json.load(f)
        self.host_index = build_host_index(self.demo_episode["hosts"])
        self.windows_by_id = {w["window_id"]: w for w in self.demo_episode["windows"]}

    def get_window(self, window_id: int):
        if window_id not in self.windows_by_id:
            raise KeyError(f"window_id {window_id} out of range (0-{len(self.windows_by_id)-1})")
        return self.windows_by_id[window_id]


state = AppState()
