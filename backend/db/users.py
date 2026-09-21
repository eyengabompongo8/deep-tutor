import json
from typing import List, Dict, Any, Optional
from backend.config import (
    DEFAULT_CEFR_LEVEL,
    STARTING_SCORE,
    DEFAULT_CORRECTION_THRESHOLD,
    DEFAULT_CARDS_PER_EXERCISE,
)

class UsersMixin:
    def get_or_create_user(self, user_id: str = 'default_user', name: str = 'Default User', avatar: str = 'owl') -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE id = ?;", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            conn.execute(
                "INSERT INTO users (id, name, avatar, score, level) VALUES (?, ?, ?, ?, ?);",
                (user_id, name, avatar, STARTING_SCORE, DEFAULT_CEFR_LEVEL)
            )
            cursor = conn.execute("SELECT * FROM users WHERE id = ?;", (user_id,))
            return dict(cursor.fetchone())

    def get_user(self, user_id: str = 'default_user') -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE id = ?;", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_users(self) -> List[Dict[str, Any]]:
        """Returns all registered users with their statistics, mode progress, and avatars."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT u.id, u.name, COALESCE(u.avatar, 'owl') AS avatar,
                       u.score, u.level, u.streak, u.total_completed, u.total_skipped, u.created_at, u.updated_at,
                       um_trans.score AS trans_score, um_trans.level AS trans_level, um_trans.streak AS trans_streak, um_trans.total_completed AS trans_completed,
                       um_audio.score AS audio_score, um_audio.level AS audio_level, um_audio.streak AS audio_streak, um_audio.total_completed AS audio_completed
                 FROM users u
                 LEFT JOIN user_modes um_trans ON u.id = um_trans.user_id AND um_trans.mode = 'translation'
                 LEFT JOIN user_modes um_audio ON u.id = um_audio.user_id AND um_audio.mode = 'transcription'
                 ORDER BY u.updated_at DESC;
            """)
            users = []
            for row in cursor.fetchall():
                d = dict(row)
                users.append({
                    "id": d["id"],
                    "name": d["name"],
                    "avatar": d["avatar"],
                    "score": d["score"],
                    "level": d["level"],
                    "streak": d["streak"],
                    "total_completed": d["total_completed"],
                    "total_skipped": d["total_skipped"],
                    "created_at": d["created_at"],
                    "updated_at": d["updated_at"],
                    "modes": {
                        "translation": {
                            "score": d["trans_score"] if d["trans_score"] is not None else d["score"],
                            "level": d["trans_level"] or d["level"],
                            "streak": d["trans_streak"] or 0,
                            "total_completed": d["trans_completed"] or 0,
                        },
                        "transcription": {
                            "score": d["audio_score"] if d["audio_score"] is not None else STARTING_SCORE,
                            "level": d["audio_level"] or DEFAULT_CEFR_LEVEL,
                            "streak": d["audio_streak"] or 0,
                            "total_completed": d["audio_completed"] or 0,
                        }
                    }
                })
            return users

    def create_user(self, name: str, avatar: str = 'owl', user_id: Optional[str] = None) -> Dict[str, Any]:
        """Creates a new user profile with initialized modes."""
        clean_name = name.strip() if name else "Apprenant"
        if not clean_name:
            clean_name = "Apprenant"
            
        if not user_id:
            import re
            import uuid
            slug = re.sub(r'[^a-zA-Z0-9]', '_', clean_name.lower())[:15].strip('_')
            user_id = f"user_{slug}_{uuid.uuid4().hex[:6]}" if slug else f"user_{uuid.uuid4().hex[:8]}"

        clean_avatar = avatar.strip() if avatar else "owl"
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO users (id, name, avatar, score, level, streak, total_completed, total_skipped)
                VALUES (?, ?, ?, ?, ?, 0, 0, 0);
            """, (user_id, clean_name, clean_avatar, STARTING_SCORE, DEFAULT_CEFR_LEVEL))
            
            conn.execute("""
                INSERT OR IGNORE INTO user_modes (user_id, mode, score, level, streak, total_completed, total_skipped)
                VALUES (?, 'translation', ?, ?, 0, 0, 0);
            """, (user_id, STARTING_SCORE, DEFAULT_CEFR_LEVEL))
            conn.execute("""
                INSERT OR IGNORE INTO user_modes (user_id, mode, score, level, streak, total_completed, total_skipped)
                VALUES (?, 'transcription', ?, ?, 0, 0, 0);
            """, (user_id, STARTING_SCORE, DEFAULT_CEFR_LEVEL))

        return self.get_user_modes_profile(user_id)

    def update_user(self, user_id: str, name: Optional[str] = None, avatar: Optional[str] = None) -> Dict[str, Any]:
        """Updates user name and/or avatar."""
        return self.update_user_profile(user_id=user_id, name=name, avatar=avatar)

    def delete_user(self, user_id: str) -> bool:
        """Deletes a user and cascades. Will refuse if there is only 1 user left."""
        with self._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM users;").fetchone()[0]
            if count <= 1:
                return False
            conn.execute("DELETE FROM users WHERE id = ?;", (user_id,))
            return True

    def update_user_profile(
        self,
        user_id: str = 'default_user',
        name: Optional[str] = None,
        avatar: Optional[str] = None,
        score: Optional[float] = None,
        level: Optional[str] = None,
        streak: Optional[int] = None,
        total_completed: Optional[int] = None,
        total_skipped: Optional[int] = None
    ) -> Dict[str, Any]:
        updates = []
        params = []
        if name is not None and str(name).strip():
            updates.append("name = ?")
            params.append(str(name).strip())
        if avatar is not None and str(avatar).strip():
            updates.append("avatar = ?")
            params.append(str(avatar).strip())
        if score is not None:
            updates.append("score = ?")
            params.append(float(score))
        if level is not None:
            updates.append("level = ?")
            params.append(level)
        if streak is not None:
            updates.append("streak = ?")
            params.append(int(streak))
        if total_completed is not None:
            updates.append("total_completed = ?")
            params.append(int(total_completed))
        if total_skipped is not None:
            updates.append("total_skipped = ?")
            params.append(int(total_skipped))

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            set_str = ", ".join(updates)
            params.append(user_id)
            with self._get_connection() as conn:
                conn.execute(f"UPDATE users SET {set_str} WHERE id = ?;", params)

        return self.get_or_create_user(user_id)

    # ==========================================
    # User Modes, History & Audio Cache Methods
    # ==========================================

    def get_user_mode(self, user_id: str = 'default_user', mode: str = 'translation') -> Dict[str, Any]:
        """Gets user profile stats for a specific exercise mode ('translation' or 'transcription')."""
        self.get_or_create_user(user_id)
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM user_modes WHERE user_id = ? AND mode = ?;",
                (user_id, mode)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            # Default initialization if mode does not exist yet
            conn.execute(
                """
                INSERT OR IGNORE INTO user_modes (user_id, mode, score, level, streak, total_completed, total_skipped)
                VALUES (?, ?, ?, ?, 0, 0, 0);
                """,
                (user_id, mode, STARTING_SCORE, DEFAULT_CEFR_LEVEL)
            )
            cursor = conn.execute(
                "SELECT * FROM user_modes WHERE user_id = ? AND mode = ?;",
                (user_id, mode)
            )
            return dict(cursor.fetchone())

    def update_user_mode(
        self,
        user_id: str = 'default_user',
        mode: str = 'translation',
        score: Optional[float] = None,
        level: Optional[str] = None,
        streak: Optional[int] = None,
        total_completed: Optional[int] = None,
        total_skipped: Optional[int] = None
    ) -> Dict[str, Any]:
        """Updates score and stats for a specific mode and syncs to top-level users table if translation."""
        self.get_user_mode(user_id, mode)
        updates = []
        params = []
        if score is not None:
            updates.append("score = ?")
            params.append(float(score))
        if level is not None:
            updates.append("level = ?")
            params.append(level)
        if streak is not None:
            updates.append("streak = ?")
            params.append(int(streak))
        if total_completed is not None:
            updates.append("total_completed = ?")
            params.append(int(total_completed))
        if total_skipped is not None:
            updates.append("total_skipped = ?")
            params.append(int(total_skipped))

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            set_str = ", ".join(updates)
            params.extend([user_id, mode])
            with self._get_connection() as conn:
                conn.execute(f"UPDATE user_modes SET {set_str} WHERE user_id = ? AND mode = ?;", params)

        # Keep legacy users table in sync if updating translation mode
        if mode == 'translation':
            self.update_user_profile(
                user_id=user_id,
                score=score,
                level=level,
                streak=streak,
                total_completed=total_completed,
                total_skipped=total_skipped
            )

        return self.get_user_mode(user_id, mode)

    def reset_user_mode(
        self,
        user_id: str = 'default_user',
        mode: str = 'translation',
        start_score: float = STARTING_SCORE,
        level: str = DEFAULT_CEFR_LEVEL
    ) -> Dict[str, Any]:
        """Resets the score and statistics for a given mode."""
        self.get_or_create_user(user_id)
        start_score = max(1.0, min(6.0, float(start_score)))
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO user_modes (user_id, mode, score, level, streak, total_completed, total_skipped, updated_at)
                VALUES (?, ?, ?, ?, 0, 0, 0, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, mode) DO UPDATE SET
                    score = excluded.score,
                    level = excluded.level,
                    streak = 0,
                    total_completed = 0,
                    total_skipped = 0,
                    updated_at = CURRENT_TIMESTAMP;
            """, (user_id, mode, start_score, level))
        if mode == 'translation':
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE users
                    SET score = ?, level = ?, streak = 0, total_completed = 0, total_skipped = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?;
                """, (start_score, level, user_id))
        return self.get_user_mode(user_id, mode)

    def get_user_modes_profile(self, user_id: str = 'default_user') -> Dict[str, Any]:
        """Returns the full multi-mode profile dictionary containing both translation and transcription stats."""
        u = self.get_or_create_user(user_id)
        trans_mode = self.get_user_mode(user_id, 'translation')
        audio_mode = self.get_user_mode(user_id, 'transcription')
        return {
            "user_id": user_id,
            "name": u.get("name", "Default User"),
            "avatar": u.get("avatar", "owl"),
            "score": trans_mode.get("score", STARTING_SCORE),
            "level": trans_mode.get("level", DEFAULT_CEFR_LEVEL),
            "streak": trans_mode.get("streak", 0),
            "total_completed": trans_mode.get("total_completed", 0),
            "total_skipped": trans_mode.get("total_skipped", 0),
            "modes": {
                "translation": trans_mode,
                "transcription": audio_mode
            }
        }

    # ==========================================
    # User Settings Methods
    # ==========================================

    def get_user_settings(self, user_id: str = 'default_user') -> Dict[str, Any]:
        """Fetches user settings, creating default settings row if not exists."""
        self.get_or_create_user(user_id)
        default_daily_goal = 10
        default_theme = 'system'
        default_preferred_voice = 'default'
        default_correction_threshold = DEFAULT_CORRECTION_THRESHOLD
        default_cards_per_exercise = DEFAULT_CARDS_PER_EXERCISE
        default_auto_jump = 1

        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM user_settings WHERE user_id = ?;", (user_id,))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                if 'auto_jump' in res and res['auto_jump'] is not None:
                    res['auto_jump'] = bool(res['auto_jump'])
                return res
            conn.execute("""
                INSERT OR IGNORE INTO user_settings (user_id, daily_goal, theme, preferred_voice, correction_threshold, cards_per_exercise, auto_jump)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (user_id, default_daily_goal, default_theme, default_preferred_voice, default_correction_threshold, default_cards_per_exercise, default_auto_jump))
            cursor = conn.execute("SELECT * FROM user_settings WHERE user_id = ?;", (user_id,))
            res = dict(cursor.fetchone())
            if 'auto_jump' in res and res['auto_jump'] is not None:
                res['auto_jump'] = bool(res['auto_jump'])
            return res

    def update_user_settings(self, user_id: str = 'default_user', **kwargs) -> Dict[str, Any]:
        """Updates user settings fields (daily_goal, theme, preferred_voice, correction_threshold, cards_per_exercise, auto_jump)."""
        self.get_user_settings(user_id)
        allowed_fields = {
            'daily_goal': int,
            'theme': str,
            'preferred_voice': str,
            'correction_threshold': float,
            'cards_per_exercise': int,
            'auto_jump': lambda v: 1 if v in (True, 1, 'true', 'True', '1') else 0,
        }
        updates = []
        params = []
        for field, ftype in allowed_fields.items():
            if field in kwargs and kwargs[field] is not None:
                try:
                    val = ftype(kwargs[field])
                    if ftype is str:
                        val = val.strip()
                    updates.append(f"{field} = ?")
                    params.append(val)
                except (ValueError, TypeError):
                    continue

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            set_str = ", ".join(updates)
            params.append(user_id)
            with self._get_connection() as conn:
                conn.execute(f"UPDATE user_settings SET {set_str} WHERE user_id = ?;", params)

        return self.get_user_settings(user_id)

