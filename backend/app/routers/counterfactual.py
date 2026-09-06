from fastapi import APIRouter, HTTPException
from app.state import state
from app.schemas import CounterfactualAction, CounterfactualResponse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml"))
from counterfactual import simulate_counterfactual, InvalidAction  # noqa: E402

router = APIRouter()


@router.post("/counterfactual", response_model=CounterfactualResponse)
def counterfactual(action: CounterfactualAction):
    try:
        window = state.get_window(action.window_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if action.window_id == 0:
        raise HTTPException(
            status_code=400,
            detail="Counterfactuals need a prior window's hidden state -- pick window_id >= 1.",
        )
    prev_window = state.get_window(action.window_id - 1)

    action_dict = {"type": action.type}
    if action.type == "isolate_host":
        if not action.host:
            raise HTTPException(status_code=400, detail="isolate_host requires 'host'")
        action_dict["host"] = action.host
    elif action.type == "block_edge":
        if not (action.src and action.dst):
            raise HTTPException(status_code=400, detail="block_edge requires 'src' and 'dst'")
        action_dict["src"] = action.src
        action_dict["dst"] = action.dst
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action type '{action.type}'")

    try:
        result = simulate_counterfactual(
            state.model,
            state.demo_episode["hosts"],
            window,
            prev_window["hidden_state"],
            action_dict,
            k_steps=action.k_steps,
        )
    except InvalidAction as e:
        raise HTTPException(status_code=400, detail=str(e))

    risk_reduction = result["baseline_risk_curve"][-1] - result["counterfactual_risk_curve"][-1]

    return CounterfactualResponse(
        window_id=action.window_id,
        action=action,
        baseline_risk_curve=result["baseline_risk_curve"],
        counterfactual_risk_curve=result["counterfactual_risk_curve"],
        edges_removed=result["edges_removed"],
        risk_reduction=risk_reduction,
    )
