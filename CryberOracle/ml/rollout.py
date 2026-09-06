"""
K-step forward simulation ("what happens next if nothing changes").

Given the model's hidden state h_t and graph embedding g_t at the current
window, repeatedly feed the model's own predicted next-embedding back in
as the input for the following step. This is the P(S_(t+1) | S_t) rollout
described in the project brief -- a genuine forward simulation using the
model's own learned dynamics, not a scripted/faked curve.
"""
import numpy as np


def k_step_rollout(model, h_t, g_t, k_steps: int = 6):
    return model.rollout(np.array(h_t), np.array(g_t), k_steps)
