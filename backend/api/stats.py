import logging
from fastapi import APIRouter, HTTPException

from backend.api import deps

logger = logging.getLogger("backend.api.stats")
router = APIRouter()


@router.get("/api/stats")
def get_stats():
    if deps.data_manager is None or deps.scoring_engine is None:
        raise HTTPException(status_code=503, detail="Application dependencies are not initialized.")
    dist = deps.data_manager.get_level_distribution()
    return {
        "total_sentences": len(deps.data_manager.sentences),
        "level_distribution": dist,
        "current_user": deps.scoring_engine.profile,
    }
