from typing import List, Optional
from pydantic import BaseModel


class NodeOut(BaseModel):
    id: str
    risk_role: Optional[str] = None  # "victim" | "attacker" | None


class EdgeOut(BaseModel):
    src: str
    dst: str
    bytes: float
    conns: int
    ports: int
    syn_ratio: float


class WindowSummary(BaseModel):
    window_id: int
    ground_truth_stage: str
    heuristic_stage: str
    predicted_risk: float


class WindowDetail(BaseModel):
    window_id: int
    nodes: List[NodeOut]
    edges: List[EdgeOut]
    heuristic_stage: str
    ground_truth_stage: str
    predicted_risk: float
    explanations: List[str]


class RolloutResponse(BaseModel):
    window_id: int
    k_steps: int
    risk_curve: List[float]


class CounterfactualAction(BaseModel):
    window_id: int
    type: str  # "isolate_host" | "block_edge"
    host: Optional[str] = None
    src: Optional[str] = None
    dst: Optional[str] = None
    k_steps: int = 6


class CounterfactualResponse(BaseModel):
    window_id: int
    action: CounterfactualAction
    baseline_risk_curve: List[float]
    counterfactual_risk_curve: List[float]
    edges_removed: int
    risk_reduction: float  # baseline final - counterfactual final
