from fastapi import APIRouter, HTTPException
from app.state import state
from app.schemas import RolloutResponse
import numpy as np

router = APIRouter()


@router.get("/rollout/{window_id}", response_model=RolloutResponse)
def rollout(window_id: int, k_steps: int = 6):
    try:
        w = state.get_window(window_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    h_t = np.array(w["hidden_state"])
    g_t = np.array(w["graph_embedding"])
    risk_curve = state.model.rollout(h_t, g_t, k_steps)
    return RolloutResponse(window_id=window_id, k_steps=k_steps, risk_curve=risk_curve)
