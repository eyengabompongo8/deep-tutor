from typing import List, Dict, Any, Optional
from backend.config import HISTORY_LIMIT, STARTING_SCORE

class HistoryMixin:
    def record_exercise_history(
        self,
        user_id: str = 'default_user',
        mode: str = 'translation',
        sentence_id: Optional[int] = None,
        quality: float = 0.0,
        score_before: float = STARTING_SCORE,
        score_after: float = STARTING_SCORE,
        delta: float = 0.0,
        is_skipped: bool = False
    ):
        """Records an evaluation outcome into exercise_history."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO exercise_history (user_id, mode, sentence_id, quality, score_before, score_after, delta, is_skipped)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (user_id, mode, sentence_id, float(quality), float(score_before), float(score_after), float(delta), 1 if is_skipped else 0))

    def get_exercise_history(
        self,
        user_id: str = 'default_user',
        mode: Optional[str] = None,
        limit: int = HISTORY_LIMIT
    ) -> List[Dict[str, Any]]:
        """Retrieves recent exercise history entries."""
        with self._get_connection() as conn:
            if mode:
                cursor = conn.execute("""
                    SELECT * FROM exercise_history
                    WHERE user_id = ? AND mode = ?
                    ORDER BY id DESC LIMIT ?;
                """, (user_id, mode, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM exercise_history
                    WHERE user_id = ?
                    ORDER BY id DESC LIMIT ?;
                """, (user_id, limit))
            return [dict(r) for r in cursor.fetchall()]

