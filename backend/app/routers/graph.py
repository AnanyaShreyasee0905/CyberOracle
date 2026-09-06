from fastapi import APIRouter, HTTPException
from app.state import state
from app.schemas import WindowDetail, NodeOut, EdgeOut

router = APIRouter()


@router.get("/graph/{window_id}", response_model=WindowDetail)
def get_graph(window_id: int):
    try:
        w = state.get_window(window_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    victim = state.demo_episode["victim"]
    active_hosts = {h for edge in w["edges"] for h in (edge[0], edge[1])}
    nodes = []
    for h in state.demo_episode["hosts"]:
        role = None
        if h == victim and w["window_id"] >= (state.demo_episode["attack_start"] or 0):
            role = "victim"
        elif h.startswith("ext-"):
            role = "attacker"
        if h in active_hosts or role:
            nodes.append(NodeOut(id=h, risk_role=role))

    edges = [
        EdgeOut(src=e[0], dst=e[1], bytes=e[2]["bytes"], conns=e[2]["conns"],
                ports=e[2]["ports"], syn_ratio=e[2]["syn_ratio"])
        for e in w["edges"]
    ]

    return WindowDetail(
        window_id=w["window_id"],
        nodes=nodes,
        edges=edges,
        heuristic_stage=w["heuristic_stage"],
        ground_truth_stage=w["ground_truth_stage"],
        predicted_risk=w["predicted_risk"],
        explanations=w["explanations"],
    )
