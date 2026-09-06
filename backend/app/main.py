from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import windows, graph, rollout, counterfactual, analyst

app = FastAPI(
    title="CYBER-ORACLE API",
    description="Predictive cyber defence using a network world model.",
    version="0.1.0",
)

# Wide-open CORS for hackathon demo purposes (frontend runs on a different
# port during local dev). Tighten this before deploying anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(windows.router, prefix="/api", tags=["windows"])
app.include_router(graph.router, prefix="/api", tags=["graph"])
app.include_router(rollout.router, prefix="/api", tags=["rollout"])
app.include_router(counterfactual.router, prefix="/api", tags=["counterfactual"])
app.include_router(analyst.router, prefix="/api", tags=["analyst"])


@app.get("/")
def root():
    return {"status": "ok", "service": "cyber-oracle-api"}
