import os
import json
import random
import math
from typing import List, Dict, Any, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEVEL_RANGES = [
    (1.0, 2.0, 'A1'),
    (2.0, 3.0, 'A2'),
    (3.0, 4.0, 'B1'),
    (4.0, 5.0, 'B2'),
    (5.0, 6.0, 'C1'),
    (6.0, 6.01, 'C2'),
]

def score_to_level(score: float) -> str:
    """Converts a continuous numerical score (1.00 - 6.00) to its CEFR level string."""
    score = max(1.0, min(6.0, score))
    for low, high, level in LEVEL_RANGES:
        if low <= score < high:
            return level
    return 'C2'


import unicodedata
import re
import difflib

from backend.text_utils import levenshtein, remove_accents


class ScoringEngine:
    def __init__(self, data_manager: Optional[Any] = None):
        self.data_manager = data_manager
        self.profile = self.load_profile()

    def get_default_profile(self) -> Dict[str, Any]:
        return {
            "user_id": "default_user",
            "score": 1.20,
            "level": "A1",
            "streak": 0,
            "total_completed": 0,
            "total_skipped": 0,
            "current_mode": "translation",
            "modes": {
                "translation": {
                    "score": 1.20,
                    "level": "A1",
                    "streak": 0,
                    "total_completed": 0,
                    "total_skipped": 0
                },
                "transcription": {
                    "score": 1.20,
                    "level": "A1",
                    "streak": 0,
                    "total_completed": 0,
                    "total_skipped": 0
                }
            },
            "history": []
        }

    def get_profile(self, user_id: str = "default_user", mode: str = "translation") -> Dict[str, Any]:
        """Returns the user profile and stats for the specified mode, backed by SQLite if available."""
        if self.data_manager is not None:
            try:
                modes_prof = self.data_manager.get_user_modes_profile(user_id)
                mode_data = modes_prof.get("modes", {}).get(mode, {})
                score = float(mode_data.get("score", 1.20))
                level = mode_data.get("level", score_to_level(score))
                streak = int(mode_data.get("streak", 0))
                completed = int(mode_data.get("total_completed", 0))
                skipped = int(mode_data.get("total_skipped", 0))
                history = self.data_manager.get_exercise_history(user_id, mode=mode, limit=50)

                res = {
                    "user_id": user_id,
                    "name": modes_prof.get("name", "Default User"),
                    "avatar": modes_prof.get("avatar", "owl"),
                    "score": score,
                    "level": level,
                    "streak": streak,
                    "total_completed": completed,
                    "total_skipped": skipped,
                    "current_mode": mode,
                    "modes": modes_prof.get("modes", {}),
                    "history": history
                }
                self.profile = res
                return res
            except Exception as e:
                print(f"Warning: Could not read user mode from SQLite: {e}")

        # Fallback to local profile dict
        self.profile.setdefault("current_mode", mode)
        self.profile["current_mode"] = mode
        if "modes" not in self.profile:
            self.profile["modes"] = {
                "translation": {
                    "score": self.profile.get("score", 1.20),
                    "level": self.profile.get("level", "A1"),
                    "streak": self.profile.get("streak", 0),
                    "total_completed": self.profile.get("total_completed", 0),
                    "total_skipped": self.profile.get("total_skipped", 0)
                },
                "transcription": {
                    "score": 1.20,
                    "level": "A1",
                    "streak": 0,
                    "total_completed": 0,
                    "total_skipped": 0
                }
            }
        mode_data = self.profile["modes"].get(mode, self.profile["modes"]["translation"])
        self.profile["score"] = float(mode_data.get("score", 1.20))
        self.profile["level"] = mode_data.get("level", score_to_level(self.profile["score"]))
        self.profile["streak"] = int(mode_data.get("streak", 0))
        self.profile["total_completed"] = int(mode_data.get("total_completed", 0))
        self.profile["total_skipped"] = int(mode_data.get("total_skipped", 0))
        return self.profile

    def load_profile(self) -> Dict[str, Any]:
        # If data_manager with SQLite is available, read from database
        if self.data_manager is not None:
            try:
                return self.get_profile("default_user", "translation")
            except Exception as e:
                print(f"Warning: Could not read user from SQLite: {e}")

        profile = self.get_default_profile()
        self.save_profile(profile)
        return profile

    def save_profile(self, profile: Optional[Dict[str, Any]] = None):
        if profile is not None:
            self.profile = profile

        # Sync to SQLite if data_manager available
        if self.data_manager is not None:
            try:
                user_id = self.profile.get("user_id", "default_user")
                current_mode = self.profile.get("current_mode", "translation")
                self.data_manager.update_user_mode(
                    user_id=user_id,
                    mode=current_mode,
                    score=self.profile.get("score"),
                    level=self.profile.get("level"),
                    streak=self.profile.get("streak"),
                    total_completed=self.profile.get("total_completed"),
                    total_skipped=self.profile.get("total_skipped")
                )
            except Exception:
                pass

    def reset_profile(
        self,
        start_score: float = 1.20,
        mode: str = "translation",
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        start_score = round(max(1.0, min(6.0, float(start_score))), 2)
        lvl = score_to_level(start_score)

        if self.data_manager is not None:
            try:
                self.data_manager.reset_user_mode(user_id=user_id, mode=mode, start_score=start_score, level=lvl)
                self.data_manager.unlock_user_vocabulary_for_level(user_id=user_id, target_level=lvl)
                return self.get_profile(user_id=user_id, mode=mode)
            except Exception as e:
                print(f"Warning: Reset user mode failed in DataManager: {e}")

        # In-memory fallback
        if "modes" not in self.profile:
            self.get_profile(user_id, mode)
        if mode in self.profile.get("modes", {}):
            self.profile["modes"][mode] = {
                "score": start_score,
                "level": lvl,
                "streak": 0,
                "total_completed": 0,
                "total_skipped": 0
            }
        self.profile["score"] = start_score
        self.profile["level"] = lvl
        self.profile["streak"] = 0
        self.profile["total_completed"] = 0
        self.profile["total_skipped"] = 0
        self.profile["history"] = []
        self.save_profile()
        return self.profile

    def get_batch_sentences(
        self,
        all_sentences: Optional[List[Dict[str, Any]]] = None,
        target_score: Optional[float] = None,
        count: int = 3,
        exclude_ids: Optional[List[int]] = None,
        data_manager: Optional[Any] = None,
        user_id: str = "default_user",
        max_unknown: int = 2,
        mode: str = "translation"
    ) -> List[Dict[str, Any]]:
        """
        Samples sentences for the user session.
        If data_manager is provided, uses progressive vocabulary selection
        (sentences with known words + controlled new word roots).
        For transcription mode, ensures exercises only include familiar words (0 new words).
        Otherwise, falls back to distance-based difficulty score matching.
        """
        if mode == "transcription":
            max_unknown = 0

        if target_score is None:
            target_score = self.profile["score"]

        dm = data_manager or self.data_manager
        if dm is not None:
            try:
                prog_batch = dm.get_progressive_sentences(
                    user_id=user_id,
                    target_score=target_score,
                    count=count,
                    max_unknown=max_unknown,
                    exclude_ids=exclude_ids,
                    mode=mode
                )
                if prog_batch:
                    if mode == "transcription":
                        # Double-check that no sentence has new words
                        familiar_prog = [s for s in prog_batch if s.get("unknown_count", 0) == 0 and not s.get("new_lexemas")]
                        if familiar_prog:
                            return familiar_prog
                    else:
                        return prog_batch
            except Exception as e:
                print(f"Warning: Progressive sentence selection fallback: {e}")

        # Lazy-load all_sentences only if progressive selection didn't return candidates
        if all_sentences is None and dm is not None:
            try:
                all_sentences = dm.get_all_sentences()
            except Exception:
                all_sentences = []

        if not all_sentences:
            return []

        exclude_set = set(exclude_ids or [])
        candidates = [s for s in all_sentences if s.get("id") not in exclude_set]
        if not candidates:
            candidates = all_sentences

        if mode == "transcription":
            familiar_cands = [s for s in candidates if s.get("unknown_count", 0) == 0 and not s.get("new_lexemas")]
            if familiar_cands:
                candidates = familiar_cands

        target_int_level = max(1, min(6, round(target_score)))

        # Primary candidates: exact integer score match
        primary = [s for s in candidates if s.get("score") == target_int_level]
        # Secondary candidates: +/- 1 score difference
        secondary = [s for s in candidates if abs(s.get("score", 1) - target_int_level) == 1]
        # Remaining
        others = [s for s in candidates if abs(s.get("score", 1) - target_int_level) > 1]

        selected = []
        random.shuffle(primary)
        random.shuffle(secondary)
        random.shuffle(others)

        # Allocate mostly primary (e.g. 2 of 3) and some secondary
        num_primary = min(len(primary), max(1, count - 1))
        selected.extend(primary[:num_primary])

        needed = count - len(selected)
        if needed > 0 and secondary:
            num_sec = min(len(secondary), needed)
            selected.extend(secondary[:num_sec])

        needed = count - len(selected)
        if needed > 0 and others:
            selected.extend(others[:needed])

        # If still needed, fill from candidates
        if len(selected) < count:
            remaining = [s for s in candidates if s not in selected]
            selected.extend(remaining[:count - len(selected)])

        random.shuffle(selected)
        return selected[:count]

    def compute_sentence_quality(self, eval_result: Optional[Dict[str, Any]], is_skipped: bool) -> float:
        """
        Computes composite translation quality Q in [0.0, 1.0].
        """
        if is_skipped or not eval_result:
            return 0.0

        if not eval_result.get("is_french", True):
            return 0.05

        def _clean_val(v, default=0.0):
            try:
                val = float(v)
                return default if (math.isnan(val) or math.isinf(val)) else val
            except (ValueError, TypeError):
                return default

        s_sem = _clean_val(eval_result.get("s_sem", 0.0))
        s_nmt = _clean_val(eval_result.get("s_nmt", 0.0))
        s_fluency = _clean_val(eval_result.get("s_fluency", 0.0))

        # Base composite score (0-100)
        raw_composite = (0.55 * s_sem) + (0.25 * s_nmt) + (0.20 * s_fluency)

        # Syntax error deductions
        syntax_errs = len(eval_result.get("syntactical_errors", []))
        penalty = min(20.0, syntax_errs * 4.0)
        final_composite = max(0.0, min(100.0, raw_composite - penalty))

        return round(final_composite / 100.0, 4)

    def calculate_score_adjustment(
        self,
        current_score: float,
        evaluated_items: List[Dict[str, Any]],
        mode: str = "translation",
        user_id: str = "default_user",
        is_retry: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates new user score based on the batch of evaluated items for the specified mode.
        Updates SQLite user_modes & exercise_history if data_manager is available,
        or falls back to local in-memory/JSON profile.
        """
        if not evaluated_items:
            prof = self.get_profile(user_id=user_id, mode=mode)
            return {
                "old_score": current_score,
                "new_score": current_score,
                "delta": 0.0,
                "old_level": score_to_level(current_score),
                "new_level": score_to_level(current_score),
                "level_changed": False,
                "avg_quality": 0.0,
                "profile": prof
            }

        total_delta = 0.0
        qualities = []

        for item in evaluated_items:
            s_difficulty = float(item.get("sentence_score", round(current_score)))
            quality = float(item.get("quality", 0.0))
            is_skipped = item.get("is_skipped", False)
            qualities.append(quality)

            if is_skipped:
                if s_difficulty <= current_score:
                    item_delta = -0.05 * (1.0 + (current_score - s_difficulty) * 0.5)
                else:
                    item_delta = -0.02
            else:
                diff = s_difficulty - current_score
                expected_q = 1.0 / (1.0 + math.exp(-diff))
                
                if quality >= 0.80:
                    base_gain = 0.08 + (0.05 * max(0.0, diff))
                    item_delta = base_gain * quality
                elif quality >= 0.60:
                    item_delta = 0.03 * (quality - 0.5)
                else:
                    item_delta = -0.06 * (1.0 - quality)

            total_delta += item_delta

        scaled_delta = max(-0.30, min(0.35, total_delta))
        new_score = round(max(1.00, min(6.00, current_score + scaled_delta)), 2)
        real_delta = round(new_score - current_score, 2)

        old_level = score_to_level(current_score)
        new_level = score_to_level(new_score)
        avg_q = round(float(sum(qualities) / max(1, len(qualities))), 2)

        completed_count = sum(1 for it in evaluated_items if not it.get("is_skipped"))
        skipped_count = sum(1 for it in evaluated_items if it.get("is_skipped"))

        # Update SQLite if data_manager present
        if self.data_manager is not None:
            try:
                prev_mode_data = self.data_manager.get_user_mode(user_id, mode)
                new_streak = int(prev_mode_data.get("streak", 0))
                if not is_retry:
                    if avg_q >= 0.70:
                        new_streak += 1
                    elif avg_q < 0.40:
                        new_streak = 0

                new_completed = int(prev_mode_data.get("total_completed", 0)) + (0 if is_retry else completed_count)
                new_skipped = int(prev_mode_data.get("total_skipped", 0)) + (0 if is_retry else skipped_count)

                self.data_manager.update_user_mode(
                    user_id=user_id,
                    mode=mode,
                    score=new_score,
                    level=new_level,
                    streak=new_streak,
                    total_completed=new_completed,
                    total_skipped=new_skipped
                )

                if new_level != old_level:
                    self.data_manager.unlock_user_vocabulary_for_level(user_id=user_id, target_level=new_level)

                if not is_retry:
                    for it in evaluated_items:
                        self.data_manager.record_exercise_history(
                            user_id=user_id,
                            mode=mode,
                            sentence_id=it.get("sentence_id"),
                            quality=float(it.get("quality", 0.0)),
                            score_before=current_score,
                            score_after=new_score,
                            delta=real_delta,
                            is_skipped=it.get("is_skipped", False)
                        )
            except Exception as e:
                print(f"Warning: Score adjustment SQLite persistence warning: {e}")

        # In-memory profile update
        if "modes" not in self.profile:
            self.get_profile(user_id, mode)
        if mode in self.profile.get("modes", {}):
            m = self.profile["modes"][mode]
            m["score"] = new_score
            m["level"] = new_level
            if not is_retry:
                m["total_completed"] = m.get("total_completed", 0) + completed_count
                m["total_skipped"] = m.get("total_skipped", 0) + skipped_count
                if avg_q >= 0.70:
                    m["streak"] = m.get("streak", 0) + 1
                elif avg_q < 0.40:
                    m["streak"] = 0

        self.profile["score"] = new_score
        self.profile["level"] = new_level
        if not is_retry:
            self.profile["total_completed"] = self.profile.get("total_completed", 0) + completed_count
            self.profile["total_skipped"] = self.profile.get("total_skipped", 0) + skipped_count
            if avg_q >= 0.70:
                self.profile["streak"] = self.profile.get("streak", 0) + 1
            elif avg_q < 0.40:
                self.profile["streak"] = 0

        self.profile.setdefault("history", [])
        self.profile["history"].append({
            "mode": mode,
            "old_score": current_score,
            "new_score": new_score,
            "delta": real_delta,
            "avg_quality": avg_q,
            "batch_size": len(evaluated_items)
        })
        if len(self.profile["history"]) > 50:
            self.profile["history"] = self.profile["history"][-50:]

        self.save_profile()
        full_profile = self.get_profile(user_id=user_id, mode=mode)

        return {
            "old_score": current_score,
            "new_score": new_score,
            "delta": real_delta,
            "old_level": old_level,
            "new_level": new_level,
            "level_changed": old_level != new_level,
            "avg_quality": avg_q,
            "mode": mode,
            "profile": full_profile
        }

    def evaluate_transcription(
        self,
        student_text: str,
        reference_text: str,
        is_skipped: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates student French transcription against reference text.
        Computes token-level and character-level alignment, identifying exact matches,
        accent errors, typos, missing words, and extra words.
        """
        if is_skipped or not student_text.strip():
            ref_words = re.findall(r"\b[^\W\d_]+(?:['’][^\W\d_]+)?\b", reference_text, flags=re.UNICODE)
            diff = [{"word": w, "ref_word": w, "type": "missing"} for w in ref_words]
            return {
                "quality": 0.0,
                "composite_score": 0.0,
                "is_exact": False,
                "status_message": "Sentence skipped" if is_skipped else "No transcription submitted",
                "diff": diff,
                "notes": ["Sentence skipped" if is_skipped else "No transcription submitted"],
                "reference_text": reference_text,
                "student_text": student_text
            }

        ref_words = re.findall(r"\b[^\W\d_]+(?:['’][^\W\d_]+)?\b", reference_text, flags=re.UNICODE)
        stu_words = re.findall(r"\b[^\W\d_]+(?:['’][^\W\d_]+)?\b", student_text, flags=re.UNICODE)

        ref_norm = [remove_accents(w.lower()) for w in ref_words]
        stu_norm = [remove_accents(w.lower()) for w in stu_words]

        matcher = difflib.SequenceMatcher(None, ref_norm, stu_norm)
        diff_items = []
        token_scores = []
        notes = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for r_idx, s_idx in zip(range(i1, i2), range(j1, j2)):
                    r_orig = ref_words[r_idx]
                    s_orig = stu_words[s_idx]
                    if r_orig.lower() == s_orig.lower():
                        diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "exact"})
                        token_scores.append(1.0)
                    else:
                        diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "accent"})
                        token_scores.append(0.85)
                        notes.append(f"Accent: '{s_orig}' → '{r_orig}'")
            elif tag == 'replace':
                r_slice = list(range(i1, i2))
                s_slice = list(range(j1, j2))
                min_len = min(len(r_slice), len(s_slice))
                for k in range(min_len):
                    r_orig = ref_words[r_slice[k]]
                    s_orig = stu_words[s_slice[k]]
                    dist = levenshtein(s_orig.lower(), r_orig.lower())
                    if remove_accents(s_orig.lower()) == remove_accents(r_orig.lower()):
                        diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "accent"})
                        token_scores.append(0.85)
                        notes.append(f"Accent: '{s_orig}' → '{r_orig}'")
                    elif dist <= 2:
                        diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "typo"})
                        token_scores.append(0.50)
                        notes.append(f"Typo: '{s_orig}' → '{r_orig}'")
                    else:
                        diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "mismatch"})
                        token_scores.append(0.0)

                for k in range(min_len, len(r_slice)):
                    r_orig = ref_words[r_slice[k]]
                    diff_items.append({"word": r_orig, "ref_word": r_orig, "type": "missing"})
                    token_scores.append(0.0)
                    notes.append(f"Missing: '{r_orig}'")

                for k in range(min_len, len(s_slice)):
                    s_orig = stu_words[s_slice[k]]
                    diff_items.append({"word": s_orig, "ref_word": "", "type": "extra"})
                    token_scores.append(0.0)
            elif tag == 'delete':
                for r_idx in range(i1, i2):
                    r_orig = ref_words[r_idx]
                    diff_items.append({"word": r_orig, "ref_word": r_orig, "type": "missing"})
                    token_scores.append(0.0)
                    notes.append(f"Missing: '{r_orig}'")
            elif tag == 'insert':
                for s_idx in range(j1, j2):
                    s_orig = stu_words[s_idx]
                    diff_items.append({"word": s_orig, "ref_word": "", "type": "extra"})
                    token_scores.append(0.0)

        # Character similarity fallback
        clean_ref = re.sub(r'[^\w\s]', '', reference_text.lower()).strip()
        clean_stu = re.sub(r'[^\w\s]', '', student_text.lower()).strip()
        max_c = max(1, max(len(clean_ref), len(clean_stu)))
        c_dist = levenshtein(clean_ref, clean_stu)
        char_sim = max(0.0, 1.0 - (c_dist / max_c))

        token_quality = sum(token_scores) / max(1, len(ref_words))
        raw_quality = 0.70 * token_quality + 0.30 * char_sim
        quality = round(max(0.0, min(1.0, raw_quality)), 4)
        composite_score = round(quality * 100.0, 1)

        is_exact = quality >= 0.98
        status_msg = "Exact match!" if is_exact else ("Minor corrections" if quality >= 0.70 else "Transcription reviewed")

        return {
            "quality": quality,
            "composite_score": composite_score,
            "is_exact": is_exact,
            "status_message": status_msg,
            "diff": diff_items,
            "notes": notes,
            "reference_text": reference_text,
            "student_text": student_text
        }
