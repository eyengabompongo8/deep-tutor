import logging
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.api import deps
from backend.config import (
    DEFAULT_CARDS_PER_EXERCISE,
    MAX_UNKNOWN_WORDS,
)
from backend.nlp.text_utils import autocorrect_text, get_aligned_targets
from backend.scoring.cefr import score_to_level
from backend.api.schemas import (
    HintRequest,
    EvaluateSingleRequest,
    EvaluateTranscriptionRequest,
    CompleteBatchRequest,
    ExerciseBatchResponse,
    EvaluateSingleResponse,
    EvaluateTranscriptionResponse,
    CompleteBatchResponse,
)

logger = logging.getLogger("backend.api.exercises")
router = APIRouter()


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


@router.get("/api/exercises/batch", response_model=ExerciseBatchResponse)
def get_exercises_batch(
    mode: str = "translation",
    score: Optional[float] = None,
    count: Optional[int] = Query(default=None, ge=1, le=10),
    exclude: Optional[str] = None,
    user_id: str = "default_user",
    max_unknown: int = MAX_UNKNOWN_WORDS,
):
    effective_count = count
    if effective_count is None:
        try:
            settings = deps.data_manager.get_user_settings(user_id=user_id)
            effective_count = settings.get("cards_per_exercise", DEFAULT_CARDS_PER_EXERCISE)
        except Exception as exc:
            logger.warning("Failed to fetch user settings for '%s', using default: %s", user_id, exc)
            effective_count = DEFAULT_CARDS_PER_EXERCISE
    mode_prof = deps.scoring_engine.get_profile(user_id=user_id, mode=mode)
    target_score = score if score is not None else mode_prof["score"]
    exclude_ids = []
    if exclude:
        try:
            exclude_ids = [int(x.strip()) for x in exclude.split(",") if x.strip().isdigit()]
        except Exception as exc:
            logger.debug("Failed to parse exclude param '%s': %s", exclude, exc)

    # For transcription exercises, only include familiar words (no new words)
    effective_max_unknown = 0 if mode == "transcription" else max_unknown

    batch = deps.scoring_engine.get_batch_sentences(
        all_sentences=None,
        target_score=target_score,
        count=effective_count,
        exclude_ids=exclude_ids,
        data_manager=deps.data_manager,
        user_id=user_id,
        max_unknown=effective_max_unknown,
        mode=mode,
    )

    # Prewarm audio synthesis in background for smooth instant playback
    if batch and deps.tts_engine is not None:
        deps.tts_engine.prewarm_batch(batch)

    # If translation mode, eagerly prewarm reference passes in background (Thread 1)
    if mode == "translation" and batch and deps.eager_processor is not None:
        deps.eager_processor.enqueue_batch(batch)

    return {
        "target_score": target_score,
        "target_level": score_to_level(target_score),
        "mode": mode,
        "count": len(batch),
        "items": batch,
    }


@router.post("/api/exercises/hint")
def get_sentence_hint(req: HintRequest):
    if deps.eager_processor is None:
        raise HTTPException(status_code=503, detail="Eager processor is not initialized.")
    return deps.eager_processor.get_hint(
        source_text=req.source_text,
        reference_text=req.reference_text,
        selected_indices=req.selected_indices,
        sentence_id=req.sentence_id,
        first_time_lemmas=req.first_time_lexemas,
    )


@router.post("/api/exercises/evaluate-single", response_model=EvaluateSingleResponse)
def evaluate_single_sentence(req: EvaluateSingleRequest):
    if req.is_skipped or not req.student_text.strip():
        vocab_progress = []
        try:
            vocab_progress = deps.data_manager.record_lexema_appearances(
                user_id=req.user_id or "default_user",
                sentence_id=req.sentence_id,
                is_success=False,
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
            "fluency_word_breakdown": [],
        }

    try:
        engine = deps.eager_processor.get_engine() if deps.eager_processor is not None else None
        if engine is None:
            from backend.server import get_ml_engine
            engine = get_ml_engine()
        lt = getattr(engine, "lang_tool", None)
        corrected_student_text, autocorrect_msgs = autocorrect_text(req.student_text, req.reference_text, lang_tool=lt)

        # Compute ML evaluation via Thread 2 evaluator (reusing precomputed reference bundle)
        result = deps.eager_processor.evaluate_student(
            source_text=req.source_text,
            student_text=corrected_student_text,
            reference_text=req.reference_text,
            sentence_id=req.sentence_id,
        )

        eval_data = deps.scoring_engine.compute_sentence_quality(result, is_skipped=False, threshold=req.threshold)
        is_correct = eval_data.get("is_correct", False)
        quality = _safe_float(eval_data.get("quality", 0.0), 0.0)
        composite_score = round(quality * 100.0, 1)

        # Record lexema appearances and mastery for progressive vocabulary unlock
        vocab_progress = []
        try:
            is_success = is_correct
            vocab_progress = deps.data_manager.record_lexema_appearances(
                user_id=req.user_id or "default_user",
                sentence_id=req.sentence_id,
                is_success=is_success,
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
            "fluency_word_breakdown": result.get("fluency_word_breakdown", []),
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
            "fluency_word_breakdown": [],
        }


@router.post("/api/exercises/evaluate-transcription", response_model=EvaluateTranscriptionResponse)
def evaluate_transcription_sentence(req: EvaluateTranscriptionRequest):
    corrected_student_text = req.student_text
    autocorrect_msgs = []
    if not req.is_skipped and req.student_text:
        engine = deps.eager_processor.get_engine() if deps.eager_processor is not None else None
        if engine is None:
            try:
                from backend.server import get_ml_engine
                engine = get_ml_engine()
            except Exception:
                engine = None
        lt = getattr(engine, "lang_tool", None)
        corrected_student_text, autocorrect_msgs = autocorrect_text(
            req.student_text,
            req.reference_text,
            lang_tool=lt,
        )

    result = deps.scoring_engine.evaluate_transcription(
        student_text=corrected_student_text,
        reference_text=req.reference_text,
        is_skipped=req.is_skipped,
    )
    result["sentence_id"] = req.sentence_id
    result["is_skipped"] = req.is_skipped
    result["student_text"] = req.student_text
    result["corrected_text"] = corrected_student_text
    result["autocorrect_msgs"] = autocorrect_msgs

    # Record lexema appearances for familiar words practiced in transcription
    vocab_progress = []
    try:
        is_success = (not req.is_skipped) and bool(result.get("is_correct", False))
        vocab_progress = deps.data_manager.record_lexema_appearances(
            user_id=req.user_id or "default_user",
            sentence_id=req.sentence_id,
            is_success=is_success,
        )
    except Exception as exc:
        logger.warning("Failed recording transcription lexema appearances: %s", exc)
    result["vocabulary_progress"] = vocab_progress

    return result


@router.post("/api/exercises/complete-batch", response_model=CompleteBatchResponse)
def complete_batch(req: CompleteBatchRequest):
    mode_prof = deps.scoring_engine.get_profile(user_id=req.user_id, mode=req.mode)
    curr_score = req.current_score if req.current_score is not None else mode_prof["score"]
    items = [it.model_dump() for it in req.evaluated_items]
    res = deps.scoring_engine.calculate_score_adjustment(
        current_score=curr_score,
        evaluated_items=items,
        mode=req.mode,
        user_id=req.user_id,
        is_retry=req.is_retry,
    )
    return res
