import os
import re
import logging
import threading
import unicodedata
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.data_manager import DataManager
from backend.scoring_engine import ScoringEngine, score_to_level
from backend.tts_engine import get_tts_engine
from backend.eager_processor import EagerPrecomputeManager
from backend.text_utils import levenshtein, remove_accents, autocorrect_text, get_aligned_targets
from backend.schemas import (
    CreateUserRequest,
    UpdateUserRequest,
    UpdateUserSettingsRequest,
    ProfileResetRequest,
    HintRequest,
    EvaluateSingleRequest,
    EvaluateTranscriptionRequest,
    CompleteBatchRequest,
    ExerciseBatchResponse,
    EvaluateSingleResponse,
    EvaluateTranscriptionResponse,
    CompleteBatchResponse,
)
from backend.config import (
    HOST,
    PORT,
    RELOAD,
    DEFAULT_CEFR_LEVEL,
    STARTING_SCORE,
    DEFAULT_CARDS_PER_EXERCISE,
    MAX_UNKNOWN_WORDS,
    DEFAULT_CORRECTION_THRESHOLD,
)

logger = logging.getLogger("backend.server")

# Global singleton handles — initialised inside lifespan, never at import time
data_manager = None
scoring_engine = None
tts_engine = None
eager_processor = None

# Lazy load MLEngine with thread-safe lock
ml_engine = None
ml_engine_lock = threading.Lock()

def get_ml_engine():
    global ml_engine
    if ml_engine is None:
        with ml_engine_lock:
            if ml_engine is None:
                logger.info("Loading MLEngine models...")
                from backend.models import MLEngine
                ml_engine = MLEngine()
                logger.info("MLEngine ready.")
    return ml_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_manager, scoring_engine, tts_engine, eager_processor
    data_manager = DataManager()
    scoring_engine = ScoringEngine(data_manager=data_manager)
    tts_engine = get_tts_engine(data_manager=data_manager)
    eager_processor = EagerPrecomputeManager(get_engine_fn=get_ml_engine, data_manager=data_manager)
    logger.info("FrenchTutor API backend started.")
    yield
    logger.info("FrenchTutor API backend shutting down.")

# Initialize FastAPI app
app = FastAPI(title="FrenchTutor API", version="1.0.0", lifespan=lifespan)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default

# --- API Endpoints ---

@app.get("/api/users")
def get_all_users():
    return data_manager.list_users()

@app.post("/api/users")
def create_new_user(req: CreateUserRequest):
    clean_name = req.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    return data_manager.create_user(name=clean_name, avatar=req.avatar or "owl")

@app.put("/api/users/{user_id}")
def update_user_endpoint(user_id: str, req: UpdateUserRequest):
    updated = data_manager.update_user(user_id=user_id, name=req.name, avatar=req.avatar)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")
    return updated

@app.delete("/api/users/{user_id}")
def delete_user_endpoint(user_id: str):
    success = data_manager.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete user (at least one user must remain).")
    return {"status": "deleted", "user_id": user_id}

@app.get("/api/user/settings")
def get_user_settings_endpoint(user_id: str = "default_user"):
    return data_manager.get_user_settings(user_id=user_id)

@app.put("/api/user/settings")
def update_user_settings_endpoint(req: UpdateUserSettingsRequest, user_id: str = "default_user"):
    update_data = req.model_dump(exclude_unset=True) if hasattr(req, "model_dump") else req.dict(exclude_unset=True)
    return data_manager.update_user_settings(user_id=user_id, **update_data)

@app.get("/api/user/profile")
def get_user_profile(user_id: str = "default_user", mode: str = "translation"):
    prof = scoring_engine.get_profile(user_id=user_id, mode=mode)
    try:
        vocab_stats = data_manager.get_user_vocabulary_stats(user_id=user_id)
        prof["vocabulary"] = {
            "total_known": vocab_stats["total_known"],
            "total_learning": vocab_stats["total_learning"],
            "by_level": vocab_stats["by_level"]
        }
    except Exception as exc:
        logger.warning("Could not fetch vocabulary stats for profile: %s", exc)
    return prof

@app.get("/api/user/vocabulary")
@app.get("/api/user/vocab")
def get_user_vocabulary(user_id: str = "default_user"):
    return data_manager.get_user_vocabulary_stats(user_id=user_id)

@app.post("/api/user/reset")
def reset_user_profile(req: ProfileResetRequest):
    return scoring_engine.reset_profile(start_score=req.start_score, mode=req.mode, user_id=req.user_id)

@app.get("/api/exercises/batch", response_model=ExerciseBatchResponse)
def get_exercises_batch(
    mode: str = "translation",
    score: Optional[float] = None,
    count: Optional[int] = Query(default=None, ge=1, le=10),
    exclude: Optional[str] = None,
    user_id: str = "default_user",
    max_unknown: int = MAX_UNKNOWN_WORDS
):
    effective_count = count
    if effective_count is None:
        try:
            settings = data_manager.get_user_settings(user_id=user_id)
            effective_count = settings.get("cards_per_exercise", DEFAULT_CARDS_PER_EXERCISE)
        except Exception as exc:
            logger.warning("Failed to fetch user settings for '%s', using default: %s", user_id, exc)
            effective_count = DEFAULT_CARDS_PER_EXERCISE
    mode_prof = scoring_engine.get_profile(user_id=user_id, mode=mode)
    target_score = score if score is not None else mode_prof["score"]
    exclude_ids = []
    if exclude:
        try:
            exclude_ids = [int(x.strip()) for x in exclude.split(",") if x.strip().isdigit()]
        except Exception as exc:
            logger.debug("Failed to parse exclude param '%s': %s", exclude, exc)

    # For transcription exercises, only include familiar words (no new words)
    effective_max_unknown = 0 if mode == "transcription" else max_unknown

    batch = scoring_engine.get_batch_sentences(
        all_sentences=None,
        target_score=target_score,
        count=effective_count,
        exclude_ids=exclude_ids,
        data_manager=data_manager,
        user_id=user_id,
        max_unknown=effective_max_unknown,
        mode=mode
    )

    # If transcription mode, prewarm audio synthesis in background
    if mode == "transcription" and batch and tts_engine is not None:
        tts_engine.prewarm_batch(batch)

    # If translation mode, eagerly prewarm reference passes in background (Thread 1)
    if mode == "translation" and batch and eager_processor is not None:
        eager_processor.enqueue_batch(batch)

    return {
        "target_score": target_score,
        "target_level": score_to_level(target_score),
        "mode": mode,
        "count": len(batch),
        "items": batch
    }

@app.post("/api/exercises/hint")
def get_sentence_hint(req: HintRequest):
    return eager_processor.get_hint(
        source_text=req.source_text,
        reference_text=req.reference_text,
        selected_indices=req.selected_indices,
        sentence_id=req.sentence_id,
        first_time_lemmas=req.first_time_lexemas
    )

@app.post("/api/exercises/evaluate-single", response_model=EvaluateSingleResponse)
def evaluate_single_sentence(req: EvaluateSingleRequest):
    if req.is_skipped or not req.student_text.strip():
        vocab_progress = []
        try:
            vocab_progress = data_manager.record_lexema_appearances(
                user_id=req.user_id or "default_user",
                sentence_id=req.sentence_id,
                is_success=False
            )
        except Exception as exc:
            logger.warning("Failed recording skipped lexema appearance: %s", exc)

        return {
            "sentence_id": req.sentence_id,
            "is_skipped": True,
            "is_correct": False,
            "threshold": req.threshold,
            "quality": 0.0,
            "composite_score": 0.0,
            "status_message": "Sentence skipped",
            "vocabulary_progress": vocab_progress,
            "reference_text": req.reference_text,
            "autocorrect_msgs": [],
            "syntactical_errors": [],
            "src_words": [],
            "tgt_words": [],
            "word_alignment_matrix": [],
            "aligned_targets_map": {},
            "nmt_word_breakdown": [],
            "fluency_word_breakdown": []
        }

    try:
        engine = get_ml_engine()
        lt = getattr(engine, "lang_tool", None)
        corrected_student_text, autocorrect_msgs = autocorrect_text(req.student_text, req.reference_text, lang_tool=lt)
        
        # Compute ML evaluation via Thread 2 evaluator (reusing precomputed reference bundle)
        result = eager_processor.evaluate_student(
            source_text=req.source_text,
            student_text=corrected_student_text,
            reference_text=req.reference_text,
            sentence_id=req.sentence_id
        )

        eval_data = scoring_engine.compute_sentence_quality(result, is_skipped=False, threshold=req.threshold)
        is_correct = eval_data.get("is_correct", False)
        quality = _safe_float(eval_data.get("quality", 0.0), 0.0)
        composite_score = round(quality * 100.0, 1)

        # Record lexema appearances and mastery for progressive vocabulary unlock
        vocab_progress = []
        try:
            is_success = is_correct
            vocab_progress = data_manager.record_lexema_appearances(
                user_id=req.user_id or "default_user",
                sentence_id=req.sentence_id,
                is_success=is_success
            )
        except Exception as exc:
            logger.warning("Failed recording lexema appearances: %s", exc)

        mapping_data_student = result.get("mapping_data")
        mapping_data_ref = result.get("mapping_data_ref")

        src_words = []
        student_words = []
        aligned_student_map = {}
        if mapping_data_student is not None:
            src_words, student_words, attn_matrix_student = mapping_data_student
            aligned_student_map = {
                i: get_aligned_targets([i], attn_matrix_student, len(student_words))
                for i in range(len(src_words))
            }

        ref_words = []
        aligned_ref_map = {}
        if mapping_data_ref is not None:
            src_words_ref, ref_words, attn_matrix_ref = mapping_data_ref
            if not src_words:
                src_words = src_words_ref
            aligned_ref_map = {
                i: get_aligned_targets([i], attn_matrix_ref, len(ref_words))
                for i in range(len(src_words_ref))
            }

        lid_val = result.get("lid_prob")
        clean_lid = 1.0 if (lid_val is None or np.isnan(lid_val) or np.isinf(lid_val)) else float(lid_val)

        return {
            "sentence_id": req.sentence_id,
            "is_skipped": False,
            "is_correct": is_correct,
            "threshold": req.threshold,
            "quality": quality,
            "composite_score": composite_score,
            "vocabulary_progress": vocab_progress,
            "is_french": bool(result.get("is_french", True)),
            "lid_prob": clean_lid,
            "s_sem": round(_safe_float(result.get("s_sem", 0.0)), 1),
            "s_nmt": round(_safe_float(result.get("s_nmt", 0.0)), 1),
            "s_fluency": round(_safe_float(result.get("s_fluency", 0.0)), 1),
            "nmt_ppl": round(_safe_float(result.get("nmt_ppl", 0.0)), 2),
            "fluency_ppl": round(_safe_float(result.get("fluency_ppl", 0.0)), 2),
            "corrected_text": corrected_student_text,
            "autocorrect_msgs": autocorrect_msgs,
            "syntactical_errors": result.get("syntactical_errors", []),
            "src_words": src_words,
            "ref_words": ref_words,
            "aligned_ref_map": aligned_ref_map,
            "student_words": student_words,
            "aligned_student_map": aligned_student_map,
            "tgt_words": ref_words or student_words,
            "aligned_targets_map": aligned_ref_map or aligned_student_map,
            "nmt_word_breakdown": result.get("nmt_word_breakdown", []),
            "fluency_word_breakdown": result.get("fluency_word_breakdown", [])
        }
    except Exception as e:
        logger.error("Evaluation error: %s", e, exc_info=True)
        return {
            "sentence_id": req.sentence_id,
            "is_skipped": False,
            "is_correct": False,
            "threshold": req.threshold,
            "quality": 0.0,
            "composite_score": 0.0,
            "status_message": f"Evaluation error: {str(e)}",
            "vocabulary_progress": [],
            "reference_text": req.reference_text,
            "autocorrect_msgs": [],
            "syntactical_errors": [f"Evaluation error: {str(e)}"],
            "src_words": [],
            "tgt_words": [],
            "word_alignment_matrix": [],
            "aligned_targets_map": {},
            "nmt_word_breakdown": [],
            "fluency_word_breakdown": []
        }

@app.post("/api/exercises/evaluate-transcription", response_model=EvaluateTranscriptionResponse)
def evaluate_transcription_sentence(req: EvaluateTranscriptionRequest):
    result = scoring_engine.evaluate_transcription(
        student_text=req.student_text,
        reference_text=req.reference_text,
        is_skipped=req.is_skipped
    )
    result["sentence_id"] = req.sentence_id
    result["is_skipped"] = req.is_skipped

    # Record lexema appearances for familiar words practiced in transcription
    vocab_progress = []
    try:
        is_success = (not req.is_skipped) and (result.get("quality", 0.0) >= 0.60)
        vocab_progress = data_manager.record_lexema_appearances(
            user_id=req.user_id or "default_user",
            sentence_id=req.sentence_id,
            is_success=is_success
        )
    except Exception as exc:
        logger.warning("Failed recording transcription lexema appearances: %s", exc)
    result["vocabulary_progress"] = vocab_progress

    return result

@app.post("/api/exercises/complete-batch", response_model=CompleteBatchResponse)
def complete_batch(req: CompleteBatchRequest):
    mode_prof = scoring_engine.get_profile(user_id=req.user_id, mode=req.mode)
    curr_score = req.current_score if req.current_score is not None else mode_prof["score"]
    items = [it.model_dump() for it in req.evaluated_items]
    res = scoring_engine.calculate_score_adjustment(
        current_score=curr_score,
        evaluated_items=items,
        mode=req.mode,
        user_id=req.user_id,
        is_retry=req.is_retry
    )
    return res

@app.get("/api/audio/synthesize")
def synthesize_audio(text: str, sentence_id: Optional[int] = None, voice: Optional[str] = None):
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty.")
    try:
        audio_path = tts_engine.synthesize(clean_text, sentence_id=sentence_id, voice=voice)
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Failed to generate audio file.")
        
        media_type = "audio/mpeg" if audio_path.endswith(".mp3") else "audio/wav"
        return FileResponse(
            audio_path,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=86400"}
        )
    except Exception as e:
        logger.error("Audio synthesis error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.get("/api/stats")
def get_stats():
    dist = data_manager.get_level_distribution()
    return {
        "total_sentences": len(data_manager.sentences),
        "level_distribution": dist,
        "current_user": scoring_engine.profile
    }

# Mount static files if frontend build exists
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dist = os.path.join(ROOT_DIR, "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="FrenchTutor API Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind to")
    parser.add_argument("--host", type=str, default=HOST, help="Host interface to bind to")
    parser.add_argument("--reload", action="store_true", default=RELOAD, help="Enable auto-reload")
    args = parser.parse_args()
    uvicorn.run("backend.server:app", host=args.host, port=args.port, reload=args.reload)
