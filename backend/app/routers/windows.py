from fastapi import APIRouter
from app.state import state
from app.schemas import WindowSummary

router = APIRouter()


@router.get("/windows", response_model=list[WindowSummary])
def list_windows():
    """All windows in the demo episode, in order -- powers the confidence
    timeline and MITRE stage strip on the dashboard."""
    return [
        WindowSummary(
            window_id=w["window_id"],
            ground_truth_stage=w["ground_truth_stage"],
            heuristic_stage=w["heuristic_stage"],
            predicted_risk=w["predicted_risk"],
        )
        for w in state.demo_episode["windows"]
    ]


@router.get("/episode-info")
def episode_info():
    return {
        "episode_id": state.demo_episode["episode_id"],
        "hosts": state.demo_episode["hosts"],
        "attack_start": state.demo_episode["attack_start"],
        "victim": state.demo_episode["victim"],
        "num_windows": len(state.demo_episode["windows"]),
    }
