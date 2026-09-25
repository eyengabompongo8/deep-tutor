import logging
import math
import os
import random
from typing import Any, Dict, List, Optional

from backend.config import (
    DEFAULT_CARDS_PER_EXERCISE,
    DEFAULT_CEFR_LEVEL,
    DEFAULT_CORRECTION_THRESHOLD,
    HISTORY_LIMIT,
    MAX_UNKNOWN_WORDS,
)
from backend.nlp.text_utils import levenshtein, remove_accents
from backend.scoring.cefr import LEVEL_RANGES, score_to_level
from backend.scoring.transcription import evaluate_transcription as _evaluate_transcription

logger = logging.getLogger("backend.scoring.engine")


class ScoringEngine:
    def __init__(self, data_manager: Optional[Any] = None, profile_file: Optional[str] = None):
        self.data_manager = data_manager
        self.profile = self.load_profile()

    def get_default_profile(self) -> Dict[str, Any]:
        return {
            "user_id": "default_user",
            "level": DEFAULT_CEFR_LEVEL,
            "streak": 0,
            "total_completed": 0,
            "total_skipped": 0,
            "current_mode": "translation",
            "modes": {
                "translation": {
                    "level": DEFAULT_CEFR_LEVEL,
                    "streak": 0,
                    "total_completed": 0,
                    "total_skipped": 0
                },
                "transcription": {
                    "level": DEFAULT_CEFR_LEVEL,
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
                # 'mixed' mode shares the translation baseline without modifying it
                query_mode = "translation" if mode == "mixed" else mode
                mode_data = modes_prof.get("modes", {}).get(query_mode, {})
                level = mode_data.get("level", DEFAULT_CEFR_LEVEL)
                streak = int(mode_data.get("streak", 0))
                completed = int(mode_data.get("total_completed", 0))
                skipped = int(mode_data.get("total_skipped", 0))
                history = self.data_manager.get_exercise_history(user_id, mode=query_mode, limit=HISTORY_LIMIT)

                res = {
                    "user_id": user_id,
                    "name": modes_prof.get("name", "Default User"),
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
                logger.warning("Could not read user mode from SQLite: %s", e)

        # Fallback to local profile dict
        self.profile.setdefault("current_mode", mode)
        self.profile["current_mode"] = mode
        if "modes" not in self.profile:
            self.profile["modes"] = {
                "translation": {
                    "level": self.profile.get("level", DEFAULT_CEFR_LEVEL),
                    "streak": self.profile.get("streak", 0),
                    "total_completed": self.profile.get("total_completed", 0),
                    "total_skipped": self.profile.get("total_skipped", 0)
                },
                "transcription": {
                    "level": DEFAULT_CEFR_LEVEL,
                    "streak": 0,
                    "total_completed": 0,
                    "total_skipped": 0
                }
            }
        mode_data = self.profile["modes"].get(mode, self.profile["modes"]["translation"])
        self.profile["level"] = mode_data.get("level", DEFAULT_CEFR_LEVEL)
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
                logger.warning("Could not read user from SQLite: %s", e)

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
                    level=self.profile.get("level"),
                    streak=self.profile.get("streak"),
                    total_completed=self.profile.get("total_completed"),
                    total_skipped=self.profile.get("total_skipped")
                )
            except Exception:
                pass

    def reset_profile(
        self,
        level: str = DEFAULT_CEFR_LEVEL,
        mode: str = "translation",
        user_id: str = "default_user",
        **kwargs
    ) -> Dict[str, Any]:
        if "start_score" in kwargs and not level:
            level = score_to_level(kwargs["start_score"])
        lvl = str(level).upper().strip() if level else DEFAULT_CEFR_LEVEL

        if self.data_manager is not None:
            try:
                self.data_manager.reset_user_mode(user_id=user_id, mode=mode, level=lvl)
                self.data_manager.unlock_user_vocabulary_for_level(user_id=user_id, target_level=lvl)
                return self.get_profile(user_id=user_id, mode=mode)
            except Exception as e:
                logger.warning("Reset user mode failed in DataManager: %s", e)

        # In-memory fallback
        if "modes" not in self.profile:
            self.get_profile(user_id, mode)
        if mode in self.profile.get("modes", {}):
            self.profile["modes"][mode] = {
                "level": lvl,
                "streak": 0,
                "total_completed": 0,
                "total_skipped": 0
            }
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
        target_level: Optional[str] = None,
        count: int = DEFAULT_CARDS_PER_EXERCISE,
        exclude_ids: Optional[List[int]] = None,
        data_manager: Optional[Any] = None,
        user_id: str = "default_user",
        max_unknown: int = MAX_UNKNOWN_WORDS,
        mode: str = "translation",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Samples sentences for the user session.
        If data_manager is provided, uses progressive vocabulary selection
        (sentences with known words + controlled new word roots).
        For transcription mode, ensures exercises only include familiar words (0 new words).
        """
        if target_level is None and "target_score" in kwargs and kwargs["target_score"] is not None:
            target_level = score_to_level(kwargs["target_score"])

        if mode == "transcription":
            max_unknown = 0

        dm = data_manager or self.data_manager
        if target_level is None:
            if dm is not None:
                try:
                    mode_info = dm.get_user_mode(user_id, mode)
                    target_level = mode_info.get("level", DEFAULT_CEFR_LEVEL)
                except Exception:
                    target_level = self.profile.get("level", DEFAULT_CEFR_LEVEL)
            else:
                target_level = self.profile.get("level", DEFAULT_CEFR_LEVEL)

        target_level = str(target_level).upper().strip() if target_level else DEFAULT_CEFR_LEVEL

        if dm is not None:
            try:
                prog_batch = dm.get_progressive_sentences(
                    user_id=user_id,
                    target_level=target_level,
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
                logger.warning("Progressive sentence selection fallback: %s", e)

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

        # Primary candidates: exact level match
        primary = [s for s in candidates if s.get("level", "").upper() == target_level]
        others = [s for s in candidates if s.get("level", "").upper() != target_level]

        selected = []
        random.shuffle(primary)
        random.shuffle(others)

        selected.extend(primary[:count])
        needed = count - len(selected)
        if needed > 0 and others:
            selected.extend(others[:needed])

        random.shuffle(selected)
        return selected[:count]

    def compute_sentence_quality(self, eval_result: Optional[Dict[str, Any]], is_skipped: bool, threshold: float = DEFAULT_CORRECTION_THRESHOLD) -> Dict[str, Any]:
        """
        Determines binary correctness of the translation based on the 3 core metrics.
        Returns a dictionary with 'is_correct' and 'quality' (1.0 or 0.0 for compatibility).
        """
        if is_skipped or not eval_result:
            return {"is_correct": False, "quality": 0.0}

        if not eval_result.get("is_french", True):
            return {"is_correct": False, "quality": 0.0}

        def _clean_val(v, default=0.0):
            try:
                val = float(v)
                return default if (math.isnan(val) or math.isinf(val)) else val
            except (ValueError, TypeError):
                return default

        s_sem = _clean_val(eval_result.get("s_sem", 0.0))
        s_nmt = _clean_val(eval_result.get("s_nmt", 0.0))
        s_fluency = _clean_val(eval_result.get("s_fluency", 0.0))

        # Binary correctness check
        is_correct = bool(s_sem >= threshold and s_nmt >= threshold and s_fluency >= threshold)

        return {
            "is_correct": is_correct,
            "quality": 1.0 if is_correct else 0.0
        }

    def calculate_score_adjustment(
        self,
        evaluated_items: List[Dict[str, Any]],
        mode: str = "translation",
        user_id: str = "default_user",
        is_retry: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculates batch summary (accuracy, streak, completion stats) for the specified mode.
        Updates SQLite user_modes & exercise_history if data_manager is available,
        or falls back to local in-memory/JSON profile.
        """
        qualities = []
        correct_count = 0
        total_count = len(evaluated_items) if evaluated_items else 0

        for item in (evaluated_items or []):
            if "is_correct" in item and item["is_correct"] is not None:
                is_correct = bool(item["is_correct"])
            else:
                is_correct = float(item.get("quality", 0.0)) >= 0.70
            is_skipped = item.get("is_skipped", False)
            if is_skipped:
                is_correct = False

            qualities.append(float(item.get("quality", 1.0 if is_correct else 0.0)))
            if is_correct:
                correct_count += 1

        accuracy_rate = (correct_count / total_count) if total_count > 0 else 0.0
        avg_q = round(float(sum(qualities) / max(1, len(qualities))), 2) if qualities else 0.0
        completed_count = sum(1 for it in (evaluated_items or []) if not it.get("is_skipped"))
        skipped_count = sum(1 for it in (evaluated_items or []) if it.get("is_skipped"))

        # Update SQLite if data_manager present (authoritative)
        if self.data_manager is not None:
            try:
                prev_mode_data = self.data_manager.get_user_mode(user_id, mode)
                level = prev_mode_data.get("level", DEFAULT_CEFR_LEVEL)
                new_streak = int(prev_mode_data.get("streak", 0))
                if not is_retry and total_count > 0 and mode != "mixed":
                    if accuracy_rate >= 0.66:
                        new_streak += 1
                    elif accuracy_rate < 0.33:
                        new_streak = 0

                new_completed = int(prev_mode_data.get("total_completed", 0)) + (0 if is_retry else completed_count)
                new_skipped = int(prev_mode_data.get("total_skipped", 0)) + (0 if is_retry else skipped_count)

                if mode != "mixed":
                    self.data_manager.update_user_mode(
                        user_id=user_id,
                        mode=mode,
                        level=level,
                        streak=new_streak,
                        total_completed=new_completed,
                        total_skipped=new_skipped
                    )

                if not is_retry and evaluated_items:
                    for it in evaluated_items:
                        self.data_manager.record_exercise_history(
                            user_id=user_id,
                            mode=mode,
                            sentence_id=it.get("sentence_id"),
                            quality=float(it.get("quality", 0.0)),
                            is_skipped=it.get("is_skipped", False)
                        )
            except Exception as e:
                logger.warning("Batch summary SQLite persistence failed: %s", e, exc_info=True)
        else:
            # Offline / test fallback
            if "modes" not in self.profile:
                self.get_profile(user_id, mode)
            level = self.profile.get("level", DEFAULT_CEFR_LEVEL)
            if mode in self.profile.get("modes", {}):
                m = self.profile["modes"][mode]
                level = m.get("level", DEFAULT_CEFR_LEVEL)
                if not is_retry and mode != "mixed":
                    m["total_completed"] = m.get("total_completed", 0) + completed_count
                    m["total_skipped"] = m.get("total_skipped", 0) + skipped_count
                    if total_count > 0:
                        if accuracy_rate >= 0.66:
                            m["streak"] = m.get("streak", 0) + 1
                        elif accuracy_rate < 0.33:
                            m["streak"] = 0

            if not is_retry and mode != "mixed":
                self.profile["total_completed"] = self.profile.get("total_completed", 0) + completed_count
                self.profile["total_skipped"] = self.profile.get("total_skipped", 0) + skipped_count
                if total_count > 0:
                    if accuracy_rate >= 0.66:
                        self.profile["streak"] = self.profile.get("streak", 0) + 1
                    elif accuracy_rate < 0.33:
                        self.profile["streak"] = 0

            self.profile.setdefault("history", [])
            if not is_retry and evaluated_items:
                self.profile["history"].append({
                    "mode": mode,
                    "avg_quality": avg_q,
                    "batch_size": len(evaluated_items),
                    "accuracy_rate": accuracy_rate
                })
                if len(self.profile["history"]) > HISTORY_LIMIT:
                    self.profile["history"] = self.profile["history"][-HISTORY_LIMIT:]

            self.save_profile()

        full_profile = self.get_profile(user_id=user_id, mode=mode)
        curr_mode_data = full_profile.get("modes", {}).get(mode, {})
        current_streak = curr_mode_data.get("streak", full_profile.get("streak", 0))
        current_level = curr_mode_data.get("level", full_profile.get("level", DEFAULT_CEFR_LEVEL))

        return {
            "level": current_level,
            "streak": current_streak,
            "accuracy_rate": accuracy_rate,
            "correct_count": correct_count,
            "total_count": total_count,
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
        return _evaluate_transcription(student_text, reference_text, is_skipped=is_skipped)
