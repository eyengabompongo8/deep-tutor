import os
import re
import unicodedata
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from spellchecker import SpellChecker

try:
    from backend.data_manager import DataManager
    from backend.scoring_engine import ScoringEngine, score_to_level
    from backend.tts_engine import get_tts_engine
except ImportError:
    from data_manager import DataManager
    from scoring_engine import ScoringEngine, score_to_level
    from tts_engine import get_tts_engine

# Initialize FastAPI app
app = FastAPI(title="FrenchTutor API", version="1.0.0")

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import threading

# Initialize singletons
print("Initializing DataManager and ScoringEngine...")
data_manager = DataManager()
scoring_engine = ScoringEngine(data_manager=data_manager)
tts_engine = get_tts_engine(data_manager=data_manager)
spell = SpellChecker(language='fr')

# Lazy load MLEngine with thread-safe lock
ml_engine = None
ml_engine_lock = threading.Lock()

def get_ml_engine():
    global ml_engine
    if ml_engine is None:
        with ml_engine_lock:
            if ml_engine is None:
                print("Loading MLEngine models...")
                try:
                    from backend.models import MLEngine
                except ImportError:
                    from models import MLEngine
                ml_engine = MLEngine()
                print("MLEngine ready.")
    return ml_engine

try:
    from backend.eager_processor import EagerPrecomputeManager
except ImportError:
    from eager_processor import EagerPrecomputeManager

# Initialize eager background precomputer singleton
eager_processor = EagerPrecomputeManager(get_engine_fn=get_ml_engine, data_manager=data_manager)

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

# --- Text & Alignment Helpers ---

def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def remove_accents(input_str: str) -> str:
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def autocorrect_text(student_text: str, reference_text: str):
    if not student_text:
        return student_text, []
        
    ref_words_unaccented = {remove_accents(w.lower()): w for w in re.findall(r"\b[^\W\d_]+\b", reference_text, re.UNICODE)}
    corrections_made = []
    ignored_words = {'j', 'l', 'c', 'd', 'm', 't', 's', 'n', 'qu'}
    
    def replace_word(match):
        word = match.group(0)
        word_lower = word.lower()
        word_unaccented = remove_accents(word_lower)
        
        if word_lower in ignored_words:
            return word
            
        if word_unaccented in ref_words_unaccented and ref_words_unaccented[word_unaccented].lower() != word_lower:
            correct_word = ref_words_unaccented[word_unaccented]
            if word.istitle():
                correct_word = correct_word.capitalize()
            elif word.isupper():
                correct_word = correct_word.upper()
            corrections_made.append(f"{word} -> {correct_word}")
            return correct_word
            
        if not spell.known([word_lower]):
            candidate = spell.correction(word_lower)
            if candidate and candidate != word_lower:
                if levenshtein(word_lower, candidate) <= 2:
                    if word.istitle():
                        candidate = candidate.capitalize()
                    elif word.isupper():
                        candidate = candidate.upper()
                    corrections_made.append(f"{word} -> {candidate}")
                    return candidate
                    
        return word

    corrected_text = re.sub(r"\b[^\W\d_]+\b", replace_word, student_text, flags=re.UNICODE)
    seen = set()
    unique_corrections = []
    for c in corrections_made:
        if c not in seen:
            seen.add(c)
            unique_corrections.append(c)
            
    return corrected_text, unique_corrections

def get_aligned_targets(selected_source_indices: List[int], matrix: np.ndarray, num_tgt_words: int) -> List[int]:
    if not selected_source_indices or matrix is None or matrix.size == 0:
        return []

    num_tgt, num_src = matrix.shape
    aligned = set()

    for s_idx in selected_source_indices:
        if 0 <= s_idx < num_src:
            col = matrix[:, s_idx]
            best_t = int(np.argmax(col))
            aligned.add(best_t)

            max_col_val = col[best_t]
            for t_idx in range(num_tgt):
                row_best_s = int(np.argmax(matrix[t_idx, :]))
                if row_best_s == s_idx and col[t_idx] >= 0.65 * max_col_val:
                    aligned.add(t_idx)

    return sorted(list(aligned))

# --- API Models ---

class CreateUserRequest(BaseModel):
    name: str
    avatar: Optional[str] = "owl"

class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    avatar: Optional[str] = None

class ProfileResetRequest(BaseModel):
    start_score: float = 1.20
    mode: str = "translation"
    user_id: str = "default_user"

class HintRequest(BaseModel):
    sentence_id: Optional[int] = None
    source_text: str
    reference_text: str
    selected_indices: Optional[List[int]] = None
    first_time_lexemas: Optional[List[str]] = None
    user_id: Optional[str] = "default_user"

class EvaluateSingleRequest(BaseModel):
    sentence_id: int
    source_text: str
    student_text: str
    reference_text: str
    sentence_score: float = 1.0
    is_skipped: bool = False
    user_id: Optional[str] = "default_user"

class EvaluateTranscriptionRequest(BaseModel):
    sentence_id: int
    student_text: str
    reference_text: str
    sentence_score: float = 1.0
    is_skipped: bool = False
    user_id: Optional[str] = "default_user"

class BatchItemEvaluation(BaseModel):
    sentence_id: int
    sentence_score: float
    quality: float
    is_skipped: bool = False

class CompleteBatchRequest(BaseModel):
    current_score: Optional[float] = None
    evaluated_items: List[BatchItemEvaluation]
    mode: str = "translation"
    user_id: str = "default_user"
    is_retry: bool = False

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
    except Exception:
        pass
    return prof

@app.get("/api/user/vocabulary")
@app.get("/api/user/vocab")
def get_user_vocabulary(user_id: str = "default_user"):
    return data_manager.get_user_vocabulary_stats(user_id=user_id)

@app.post("/api/user/reset")
def reset_user_profile(req: ProfileResetRequest):
    return scoring_engine.reset_profile(start_score=req.start_score, mode=req.mode, user_id=req.user_id)

@app.get("/api/exercises/batch")
def get_exercises_batch(
    mode: str = "translation",
    score: Optional[float] = None,
    count: int = Query(default=3, ge=1, le=10),
    exclude: Optional[str] = None,
    user_id: str = "default_user",
    max_unknown: int = 2
):
    mode_prof = scoring_engine.get_profile(user_id=user_id, mode=mode)
    target_score = score if score is not None else mode_prof["score"]
    exclude_ids = []
    if exclude:
        try:
            exclude_ids = [int(x.strip()) for x in exclude.split(",") if x.strip().isdigit()]
        except Exception:
            pass

    # For transcription exercises, only include familiar words (no new words)
    effective_max_unknown = 0 if mode == "transcription" else max_unknown

    all_sentences = data_manager.get_all_sentences()
    batch = scoring_engine.get_batch_sentences(
        all_sentences,
        target_score=target_score,
        count=count,
        exclude_ids=exclude_ids,
        data_manager=data_manager,
        user_id=user_id,
        max_unknown=effective_max_unknown,
        mode=mode
    )

    # If transcription mode, prewarm audio synthesis in background
    if mode == "transcription" and batch:
        tts_engine.prewarm_batch(batch)

    # If translation mode, eagerly prewarm reference passes in background (Thread 1)
    if mode == "translation" and batch:
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

@app.post("/api/exercises/evaluate-single")
def evaluate_single_sentence(req: EvaluateSingleRequest):
    if req.is_skipped or not req.student_text.strip():
        vocab_progress = []
        try:
            vocab_progress = data_manager.record_lexema_appearances(
                user_id=req.user_id or "default_user",
                sentence_id=req.sentence_id,
                is_success=False
            )
        except Exception:
            pass

        return {
            "sentence_id": req.sentence_id,
            "is_skipped": True,
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
        corrected_student_text, autocorrect_msgs = autocorrect_text(req.student_text, req.reference_text)
        
        # Compute ML evaluation via Thread 2 evaluator (reusing precomputed reference bundle)
        result = eager_processor.evaluate_student(
            source_text=req.source_text,
            student_text=corrected_student_text,
            reference_text=req.reference_text,
            sentence_id=req.sentence_id
        )

        quality = _safe_float(scoring_engine.compute_sentence_quality(result, is_skipped=False), 0.0)
        composite_score = round(quality * 100.0, 1)

        # Record lexema appearances and mastery for progressive vocabulary unlock
        vocab_progress = []
        try:
            is_success = quality >= 0.60
            vocab_progress = data_manager.record_lexema_appearances(
                user_id=req.user_id or "default_user",
                sentence_id=req.sentence_id,
                is_success=is_success
            )
        except Exception:
            pass

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
        import traceback
        traceback.print_exc()
        return {
            "sentence_id": req.sentence_id,
            "is_skipped": False,
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

@app.post("/api/exercises/evaluate-transcription")
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
    except Exception:
        pass
    result["vocabulary_progress"] = vocab_progress

    return result

@app.post("/api/exercises/complete-batch")
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
        print(f"Audio synthesis error: {e}")
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
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
