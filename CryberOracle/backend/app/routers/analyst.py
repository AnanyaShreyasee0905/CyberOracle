"""Template fallback for the optional SGLang analyst experience."""
from fastapi import APIRouter, HTTPException

from app.state import state

router = APIRouter()


def _summary(window):
    risk = window["predicted_risk"]
    level = "high" if risk >= 0.70 else "elevated" if risk >= 0.40 else "low"
    reasons = "; ".join(window["explanations"]) or "no unusual graph signals were recorded"
    return (
        f"{level.title()} predicted risk ({risk:.0%}) at the {window['heuristic_stage'].replace('_', ' ')} stage. "
        f"Observed drivers: {reasons}."
    )


@router.get("/analyst/{window_id}")
def get_analyst_summary(window_id: int):
    """A clearly labelled non-LLM fallback until SGLang is configured."""
    try:
        window = state.get_window(window_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        "window_id": window_id,
        "generator": "template_fallback",
        "summary": _summary(window),
    }
