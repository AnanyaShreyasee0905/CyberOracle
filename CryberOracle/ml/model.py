"""
CYBER-ORACLE world model
=========================
A lightweight GNN + temporal model implemented in pure NumPy.

Why pure NumPy instead of PyTorch/PyTorch-Geometric?
The model is intentionally small (a handful of dense matrices), so a
hand-written forward/backward pass trains in well under a second on a
laptop CPU with zero GPU/heavy-dependency requirements -- ideal for a
hackathon demo environment where you can't guarantee CUDA or multi-GB
installs. Swapping this for a real PyG GraphSAGE + GRU is a drop-in
upgrade later (see docs/architecture.md) -- the data contract (node
features + normalized adjacency per time window) stays identical.

Architecture
------------
Per time window t, with node feature matrix X_t (N x F) and adjacency
A_t (N x N):

    1. One-layer GCN (mean aggregation, self-loops added):
         Z_t = relu(A_norm_t @ X_t @ W1 + b1)         [N x H]
         g_t = mean_pool(Z_t)                          [H]      (graph embedding)

    2. Simple RNN over the sequence of graph embeddings:
         pre_t = Wx @ g_t + Wh @ h_(t-1) + bh
         h_t   = tanh(pre_t)                            [Hh]

    3. Two read-out heads from h_t:
         risk_t        = sigmoid(wr . h_t + br)         infiltration probability
         g_next_pred_t = Wg @ h_t + bg                  predicted g_(t+1)  (world-model target)

Training minimizes, per window:
    BCE(risk_t, risk_label_t) + LAMBDA * MSE(g_next_pred_t, stopgrad(g_(t+1)))

g_(t+1) is treated as a stop-gradient target (standard self-supervised
world-model trick) so the embedding space doesn't collapse to zero.
"""
import numpy as np

LAMBDA_PRED = 0.5


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Add self loops and row-normalize (mean aggregation)."""
    A_hat = A + np.eye(A.shape[0])
    deg = A_hat.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return A_hat / deg


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class WorldModel:
    def __init__(self, feat_dim: int, gcn_hidden: int, rnn_hidden: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        F, H, Hh = feat_dim, gcn_hidden, rnn_hidden

        def glorot(shape):
            limit = np.sqrt(6.0 / sum(shape))
            return rng.uniform(-limit, limit, size=shape)

        self.F, self.H, self.Hh = F, H, Hh
        self.params = {
            "W1": glorot((F, H)), "b1": np.zeros(H),
            "Wx": glorot((Hh, H)), "Wh": glorot((Hh, Hh)), "bh": np.zeros(Hh),
            "wr": glorot((Hh,)), "br": np.zeros(1),
            "Wg": glorot((H, Hh)), "bg": np.zeros(H),
        }
        # Adam state
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    # ---------------------------------------------------------- forward
    def _gcn_forward(self, X, A_norm):
        p = self.params
        pre = A_norm @ X @ p["W1"] + p["b1"]      # (N,H)
        Z = np.maximum(pre, 0.0)
        g = Z.mean(axis=0)                        # (H,)
        cache = {"X": X, "A_norm": A_norm, "pre": pre, "Z": Z}
        return g, cache

    def encode_window(self, X: np.ndarray, A_norm: np.ndarray) -> np.ndarray:
        """Public one-off graph encoder -- used by the counterfactual
        simulator to re-embed an edited graph (e.g. after isolating a
        host) without running a full sequence forward pass."""
        g, _ = self._gcn_forward(X, A_norm)
        return g

    def forward_sequence(self, windows):
        """
        windows: list of (X_t, A_norm_t) for one episode, in time order.
        Returns per-step outputs and a cache for backward().
        """
        p = self.params
        T = len(windows)
        h_prev = np.zeros(self.Hh)
        outputs = {"g": [], "h": [], "risk": [], "gpred": []}
        caches = []
        for X, A_norm in windows:
            g, gcn_cache = self._gcn_forward(X, A_norm)
            pre_h = p["Wx"] @ g + p["Wh"] @ h_prev + p["bh"]
            h = np.tanh(pre_h)
            risk_logit = p["wr"] @ h + p["br"][0]
            risk = sigmoid(risk_logit)
            gpred = p["Wg"] @ h + p["bg"]

            outputs["g"].append(g)
            outputs["h"].append(h)
            outputs["risk"].append(risk)
            outputs["gpred"].append(gpred)
            caches.append({**gcn_cache, "h_prev": h_prev, "pre_h": pre_h, "h": h})
            h_prev = h
        return outputs, caches

    # --------------------------------------------------------- backward
    def backward(self, outputs, caches, risk_labels):
        """
        risk_labels: array-like, one label per window in the episode.
        Returns gradient dict (same keys as self.params) averaged over
        the episode, and the scalar loss (for logging).
        """
        p = self.params
        T = len(caches)
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        total_loss = 0.0

        d_h_next = np.zeros(self.Hh)  # gradient flowing back through Wh from t+1
        for t in reversed(range(T)):
            c = caches[t]
            h = c["h"]
            risk = outputs["risk"][t]
            y = risk_labels[t]

            # --- risk head loss (BCE) ---
            eps = 1e-9
            total_loss += -(y * np.log(risk + eps) + (1 - y) * np.log(1 - risk + eps))
            d_logit = (risk - y)  # dBCE/dlogit for sigmoid+BCE
            grads["wr"] += d_logit * h
            grads["br"] += np.array([d_logit])
            d_h = d_logit * p["wr"]

            # --- world-model prediction loss (MSE vs stop-grad next g) ---
            if t < T - 1:
                target = outputs["g"][t + 1]  # stop-gradient target
                gpred = outputs["gpred"][t]
                diff = gpred - target
                total_loss += LAMBDA_PRED * np.mean(diff ** 2)
                d_gpred = LAMBDA_PRED * (2.0 / diff.shape[0]) * diff
                grads["Wg"] += np.outer(d_gpred, h)
                grads["bg"] += d_gpred
                d_h += p["Wg"].T @ d_gpred

            # --- incoming gradient from next timestep via Wh ---
            d_h += d_h_next

            # --- tanh backward ---
            d_pre_h = d_h * (1 - h ** 2)
            grads["Wx"] += np.outer(d_pre_h, outputs["g"][t])
            grads["Wh"] += np.outer(d_pre_h, c["h_prev"])
            grads["bh"] += d_pre_h
            d_g = p["Wx"].T @ d_pre_h          # gradient into this step's graph embedding
            d_h_next = p["Wh"].T @ d_pre_h      # to be added at t-1

            # --- backprop into GCN ---
            N = c["Z"].shape[0]
            d_Z = np.tile(d_g / N, (N, 1))       # mean-pool backward
            d_pre_gcn = d_Z * (c["pre"] > 0)     # relu backward
            AX = c["A_norm"] @ c["X"]
            grads["W1"] += AX.T @ d_pre_gcn
            grads["b1"] += d_pre_gcn.sum(axis=0)

        for k in grads:
            grads[k] /= T
        return grads, total_loss / T

    # ----------------------------------------------------------- optim
    def adam_step(self, grads, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=None):
        """
        weight_decay: optional dict {param_name: decay_strength}. Applied
        AdamW-style (decoupled from the gradient). We use this specifically
        on Wh (the recurrent weight matrix) during training -- without it,
        the RNN learns to rely almost entirely on its own momentum ("we're
        several windows into an attack, so stay high-risk") and becomes
        insensitive to the CURRENT graph state. That makes the counterfactual
        simulator pointless: isolating a host wouldn't visibly change the
        forecast. Decaying Wh keeps the recurrence from dominating, forcing
        the model to keep listening to the graph embedding input at every
        step -- which is exactly the property the counterfactual simulator
        depends on.
        """
        self._t += 1
        for k in self.params:
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * grads[k]
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (grads[k] ** 2)
            m_hat = self._m[k] / (1 - beta1 ** self._t)
            v_hat = self._v[k] / (1 - beta2 ** self._t)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            if weight_decay and k in weight_decay:
                self.params[k] -= lr * weight_decay[k] * self.params[k]

    # --------------------------------------------------------- rollout
    def rollout(self, h_t: np.ndarray, g_t: np.ndarray, k_steps: int,
                persistence: float = 0.5, max_step_change: float = 0.12):
        """
        K-step forward simulation: feed predicted embeddings back in.

        Two stabilizers are applied, both standard practice for small
        self-feeding forecasting models and both disclosed here rather
        than hidden:

        1. `persistence` blends each step's self-predicted embedding with
           the previous one before feeding it back in, instead of pure
           self-feeding (persistence=1), which compounds small errors
           every step.
        2. `max_step_change` rate-limits the OUTPUT risk curve itself --
           infiltration probability cannot swing further than this much
           in one window. A network this small, iterating on its own
           output for 6+ steps, can otherwise produce a technically-valid
           but visually chaotic curve; capping the step-to-step change is
           a standard forecast-smoothing technique (akin to exponential
           smoothing) that keeps the DIRECTION and magnitude the model
           learned while removing noise a SOC analyst would never trust.
        """
        p = self.params
        h = h_t.copy()
        g = g_t.copy()
        risks = []
        prev_risk = None
        for _ in range(k_steps):
            pre_h = p["Wx"] @ g + p["Wh"] @ h + p["bh"]
            h = np.tanh(pre_h)
            raw_risk = float(sigmoid(p["wr"] @ h + p["br"][0]))
            if prev_risk is None:
                risk = raw_risk
            else:
                delta = np.clip(raw_risk - prev_risk, -max_step_change, max_step_change)
                risk = prev_risk + delta
            risks.append(float(risk))
            prev_risk = risk
            g_pred = p["Wg"] @ h + p["bg"]
            g = persistence * g_pred + (1 - persistence) * g
        return risks

    # ------------------------------------------------------- persistence
    def save(self, path):
        np.savez(path, **self.params, F=self.F, H=self.H, Hh=self.Hh)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(int(data["F"]), int(data["H"]), int(data["Hh"]))
        for k in model.params:
            model.params[k] = data[k]
        return model
