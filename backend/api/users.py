import logging
from fastapi import APIRouter, HTTPException

from backend.api import deps
from backend.api.schemas import (
    CreateUserRequest,
    UpdateUserRequest,
    UpdateUserSettingsRequest,
    ProfileResetRequest,
)

logger = logging.getLogger("backend.api.users")
router = APIRouter()


@router.get("/api/users")
def get_all_users():
    return deps.data_manager.list_users()


@router.post("/api/users")
def create_new_user(req: CreateUserRequest):
    clean_name = req.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    return deps.data_manager.create_user(name=clean_name)


@router.put("/api/users/{user_id}")
def update_user_endpoint(user_id: str, req: UpdateUserRequest):
    updated = deps.data_manager.update_user(user_id=user_id, name=req.name)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")
    return updated


@router.delete("/api/users/{user_id}")
def delete_user_endpoint(user_id: str):
    success = deps.data_manager.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete user (at least one user must remain).")
    return {"status": "deleted", "user_id": user_id}


@router.get("/api/user/settings")
def get_user_settings_endpoint(user_id: str = "default_user"):
    return deps.data_manager.get_user_settings(user_id=user_id)


@router.put("/api/user/settings")
def update_user_settings_endpoint(req: UpdateUserSettingsRequest, user_id: str = "default_user"):
    update_data = req.model_dump(exclude_unset=True) if hasattr(req, "model_dump") else req.dict(exclude_unset=True)
    target_user_id = update_data.pop("user_id", None) or user_id
    return deps.data_manager.update_user_settings(user_id=target_user_id, **update_data)


@router.get("/api/user/profile")
def get_user_profile(user_id: str = "default_user", mode: str = "translation"):
    prof = deps.scoring_engine.get_profile(user_id=user_id, mode=mode)
    try:
        vocab_stats = deps.data_manager.get_user_vocabulary_stats(user_id=user_id)
        prof["vocabulary"] = {
            "total_known": vocab_stats["total_known"],
            "total_learning": vocab_stats["total_learning"],
            "by_level": vocab_stats["by_level"]
        }
    except Exception as exc:
        logger.warning("Could not fetch vocabulary stats for profile: %s", exc)
    return prof


@router.get("/api/user/vocabulary")
@router.get("/api/user/vocab")
def get_user_vocabulary(user_id: str = "default_user"):
    return deps.data_manager.get_user_vocabulary_stats(user_id=user_id)


@router.post("/api/user/reset")
def reset_user_profile(req: ProfileResetRequest):
    return deps.scoring_engine.reset_profile(start_score=req.start_score, mode=req.mode, user_id=req.user_id)
