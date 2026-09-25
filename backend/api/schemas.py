from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, model_validator
from backend.config import (
    DEFAULT_CEFR_LEVEL,
    STARTING_SCORE,
    DEFAULT_CARDS_PER_EXERCISE,
    DEFAULT_CORRECTION_THRESHOLD,
)

# --- User Schemas ---

class CreateUserRequest(BaseModel):
    name: str

class UpdateUserRequest(BaseModel):
    name: Optional[str] = None

class UpdateUserSettingsRequest(BaseModel):
    user_id: Optional[str] = None
    theme: Optional[str] = None
    preferred_voice: Optional[str] = None
    correction_threshold: Optional[float] = None
    cards_per_exercise: Optional[int] = None
    auto_jump: Optional[bool] = None
    auto_play_audio: Optional[bool] = None
    transcription_retry: Optional[bool] = None
    blind_retry: Optional[bool] = None
    active_mode: Optional[str] = None

class ProfileResetRequest(BaseModel):
    start_score: float = STARTING_SCORE
    mode: str = "translation"
    user_id: str = "default_user"

# --- Exercise Request Schemas ---

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
    threshold: float = DEFAULT_CORRECTION_THRESHOLD
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
    is_correct: Optional[bool] = None

class CompleteBatchRequest(BaseModel):
    current_score: Optional[float] = None
    evaluated_items: List[BatchItemEvaluation]
    mode: str = "translation"
    user_id: str = "default_user"
    is_retry: bool = False

# --- Exercise Response Schemas ---

class VocabProgressItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    lemma: Optional[str] = None
    pos: Optional[str] = None
    level: Optional[str] = None
    appearance_count: int = 0
    success_count: int = 0
    is_known: int = 0

    # Backwards-compatibility aliases
    lexema: Optional[str] = None
    appearances: Optional[int] = None
    mastered: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def populate_compat_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            lemma = data.get("lemma") or data.get("lexema") or ""
            data.setdefault("lemma", lemma)
            data.setdefault("lexema", lemma)

            appearances = data.get("appearance_count", data.get("appearances", 0))
            data.setdefault("appearance_count", appearances)
            data.setdefault("appearances", appearances)

            is_known = data.get("is_known", 1 if data.get("mastered") else 0)
            data.setdefault("is_known", is_known)
            data.setdefault("mastered", bool(is_known))
        return data

class TranscriptionDiffItem(BaseModel):
    word: str
    ref_word: str = ""
    type: str

class ExerciseBatchResponse(BaseModel):
    target_score: float
    target_level: str
    mode: str
    count: int
    items: List[Dict[str, Any]] = []

class EvaluateSingleResponse(BaseModel):
    sentence_id: Optional[int] = None
    is_skipped: bool = False
    is_correct: bool = False
    threshold: float = DEFAULT_CORRECTION_THRESHOLD
    quality: float = 0.0
    composite_score: float = 0.0
    vocabulary_progress: List[VocabProgressItem] = []
    is_french: bool = True
    lid_prob: float = 1.0
    s_sem: float = 0.0
    s_nmt: float = 0.0
    s_fluency: float = 0.0
    nmt_ppl: float = 0.0
    fluency_ppl: float = 0.0
    corrected_text: str = ""
    autocorrect_msgs: List[str] = []
    syntactical_errors: List[str] = []
    src_words: List[str] = []
    ref_words: List[str] = []
    student_words: List[str] = []
    tgt_words: List[str] = []
    nmt_word_breakdown: List[Any] = []
    fluency_word_breakdown: List[Any] = []
    aligned_ref_map: Dict[int, List[int]] = {}
    aligned_student_map: Dict[int, List[int]] = {}
    aligned_targets_map: Dict[int, List[int]] = {}
    status_message: str = ""
    reference_text: str = ""
    word_alignment_matrix: List[Any] = []

class EvaluateTranscriptionResponse(BaseModel):
    sentence_id: Optional[int] = None
    is_skipped: bool = False
    quality: float = 0.0
    composite_score: float = 0.0
    is_exact: bool = False
    status_message: str = ""
    diff: List[TranscriptionDiffItem] = []
    notes: List[str] = []
    reference_text: str = ""
    student_text: str = ""
    corrected_text: str = ""
    autocorrect_msgs: List[str] = []
    vocabulary_progress: List[VocabProgressItem] = []

class CompleteBatchResponse(BaseModel):
    old_score: float
    new_score: float
    delta: float
    old_level: str
    new_level: str
    level_changed: bool
    avg_quality: float
    correct_count: Optional[int] = None
    total_count: Optional[int] = None
    mode: Optional[str] = None
    profile: Dict[str, Any]
