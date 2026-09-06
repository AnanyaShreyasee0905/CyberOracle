"""Quick numerical gradient check for WorldModel.backward(). Run directly:
    python3 ml/_gradcheck.py
Exits nonzero if any parameter gradient deviates from the numerical
estimate by more than the tolerance.
"""
import numpy as np
from model import WorldModel, normalize_adjacency

rng = np.random.default_rng(0)
N, F, H, Hh, T = 5, 4, 6, 5, 4

model = WorldModel(F, H, Hh, seed=1)

windows = []
for _ in range(T):
    X = rng.normal(size=(N, F))
    A = (rng.random((N, N)) > 0.5).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    windows.append((X, normalize_adjacency(A)))
labels = rng.integers(0, 2, size=T).astype(float)


# backward() treats g_(t+1) as a STOP-GRADIENT target (standard for
# self-supervised world models, prevents embedding collapse). That means
# the *intended* gradient is w.r.t. a loss where those targets are frozen
# constants -- so for the numerical check we must freeze them too,
# otherwise finite-differencing would also pick up the (intentionally
# excluded) gradient path through the target itself.
base_outputs, base_caches = model.forward_sequence(windows)
frozen_targets = [g.copy() for g in base_outputs["g"]]


def loss_fn():
    outputs, _ = model.forward_sequence(windows)
    from model import sigmoid
    T_ = len(outputs["risk"])
    loss = 0.0
    eps_ = 1e-9
    for t in range(T_):
        y = labels[t]
        risk = outputs["risk"][t]
        loss += -(y * np.log(risk + eps_) + (1 - y) * np.log(1 - risk + eps_))
        if t < T_ - 1:
            diff = outputs["gpred"][t] - frozen_targets[t + 1]
            loss += 0.5 * np.mean(diff ** 2)
    return loss / T_


outputs, caches = model.forward_sequence(windows)
grads, analytic_loss = model.backward(outputs, caches, labels)

eps = 1e-5
max_rel_err = 0.0
worst = None
for name, param in model.params.items():
    it = np.nditer(param, flags=["multi_index"])
    # sample a handful of entries per param (full check would be slow but fine here; params are tiny)
    for _ in it:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + eps
        loss_plus = loss_fn()
        param[idx] = orig - eps
        loss_minus = loss_fn()
        param[idx] = orig
        numeric_grad = (loss_plus - loss_minus) / (2 * eps)
        analytic_grad = grads[name][idx]
        denom = max(abs(numeric_grad), abs(analytic_grad), 1e-8)
        rel_err = abs(numeric_grad - analytic_grad) / denom
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            worst = (name, idx, numeric_grad, analytic_grad)

print(f"max relative error across all params: {max_rel_err:.6e}")
print(f"worst case: {worst}")
assert max_rel_err < 1e-3, "Gradient check FAILED"
print("Gradient check PASSED")
