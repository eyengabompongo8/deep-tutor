import os
import sqlite3
import re
import json
import threading
from typing import List, Dict, Any, Optional, Set, Tuple
from contextlib import contextmanager

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("FRENCHTUTOR_DB_PATH") or os.path.join(ROOT_DIR, 'data', 'app.db')
OLD_JSON_PATH = os.path.join(ROOT_DIR, 'data', 'sentences.json')

SYSTEM_COLUMNS = {
    'id', 'level', 'score_lexical', 'score_ml', 'source',
    'metadata', 'created_at', 'updated_at'
}


class DataManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DB_PATH
        self._lexemas_cache: Optional[Dict[int, Dict[str, Any]]] = None
        self._lexemas_cache_lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 10000;")
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()
        except sqlite3.OperationalError as e:
            if conn:
                try:
                    conn.rollback()
                    conn.close()
                except Exception:
                    pass
                conn = None
            if os.path.exists(self.db_path):
                uri_path = f"file:{os.path.abspath(self.db_path)}?immutable=1"
                conn = sqlite3.connect(uri_path, uri=True, timeout=15.0)
                conn.row_factory = sqlite3.Row
                yield conn
                conn.close()
                return
            raise e
        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute("PRAGMA foreign_keys = ON;")
        except sqlite3.OperationalError:
            pass

        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sentences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fr TEXT NOT NULL,
                        es TEXT NOT NULL,
                        level TEXT NOT NULL DEFAULT 'A1',
                        score_lexical REAL NOT NULL DEFAULT 1.0,
                        score_ml REAL DEFAULT NULL,
                        source TEXT NOT NULL DEFAULT 'custom',
                        metadata TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_level ON sentences(level);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_score_lexical ON sentences(score_lexical);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_score_ml ON sentences(score_ml);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_created_at ON sentences(created_at);")

                # Users table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL DEFAULT 'Default User',
                        avatar TEXT NOT NULL DEFAULT 'owl',
                        score REAL NOT NULL DEFAULT 1.20,
                        level TEXT NOT NULL DEFAULT 'A1',
                        streak INTEGER NOT NULL DEFAULT 0,
                        total_completed INTEGER NOT NULL DEFAULT 0,
                        total_skipped INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                # Dynamic migration check for avatar column
                try:
                    user_cols = [c[1] for c in conn.execute("PRAGMA table_info(users);").fetchall()]
                    if 'avatar' not in user_cols:
                        conn.execute("ALTER TABLE users ADD COLUMN avatar TEXT NOT NULL DEFAULT 'owl';")
                except Exception:
                    pass

                # Lexemas table (canonical French word roots)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS lexemas (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        lemma TEXT NOT NULL,
                        pos TEXT DEFAULT NULL,
                        level TEXT NOT NULL DEFAULT 'A1',
                        frequency REAL DEFAULT 0.0,
                        is_base BOOLEAN NOT NULL DEFAULT 0,
                        metadata TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(lemma, pos)
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_lexemas_lemma ON lexemas(lemma);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_lexemas_level ON lexemas(level);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_lexemas_is_base ON lexemas(is_base);")

                # Sentence-Lexemas junction table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sentence_lexemas (
                        sentence_id INTEGER NOT NULL,
                        lexema_id INTEGER NOT NULL,
                        PRIMARY KEY (sentence_id, lexema_id),
                        FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE,
                        FOREIGN KEY (lexema_id) REFERENCES lexemas(id) ON DELETE CASCADE
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sentence_lexemas_sentence ON sentence_lexemas(sentence_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sentence_lexemas_lexema ON sentence_lexemas(lexema_id);")

                # User lexema tracking & appearance counts
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_lexema_counts (
                        user_id TEXT NOT NULL,
                        lexema_id INTEGER NOT NULL,
                        appearance_count INTEGER NOT NULL DEFAULT 0,
                        success_count INTEGER NOT NULL DEFAULT 0,
                        is_known BOOLEAN NOT NULL DEFAULT 0,
                        last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, lexema_id),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        FOREIGN KEY (lexema_id) REFERENCES lexemas(id) ON DELETE CASCADE
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ulc_user_known ON user_lexema_counts(user_id, is_known);")

                # Seed default user if not exists
                user_row = conn.execute("SELECT id FROM users WHERE id = 'default_user';").fetchone()
                if not user_row:
                    initial_score = 1.20
                    initial_level = 'A1'
                    initial_streak = 0
                    initial_completed = 0
                    initial_skipped = 0
                    
                    # Check if legacy user_profile.json exists
                    profile_path = os.path.join(os.path.dirname(self.db_path), 'user_profile.json')
                    if os.path.exists(profile_path):
                        try:
                            with open(profile_path, 'r', encoding='utf-8') as f:
                                p_data = json.load(f)
                                initial_score = float(p_data.get('score', 1.20))
                                initial_level = p_data.get('level', 'A1')
                                initial_streak = int(p_data.get('streak', 0))
                                initial_completed = int(p_data.get('total_completed', 0))
                                initial_skipped = int(p_data.get('total_skipped', 0))
                        except Exception:
                            pass

                    conn.execute("""
                        INSERT OR IGNORE INTO users (id, name, avatar, score, level, streak, total_completed, total_skipped)
                        VALUES ('default_user', 'Default User', 'owl', ?, ?, ?, ?, ?);
                    """, (initial_score, initial_level, initial_streak, initial_completed, initial_skipped))

                # User modes table for separate exercise mode tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_modes (
                        user_id TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        score REAL NOT NULL DEFAULT 1.20,
                        level TEXT NOT NULL DEFAULT 'A1',
                        streak INTEGER NOT NULL DEFAULT 0,
                        total_completed INTEGER NOT NULL DEFAULT 0,
                        total_skipped INTEGER NOT NULL DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, mode),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user_modes_user ON user_modes(user_id);")

                # Exercise history table for tracking progress and batches
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS exercise_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        mode TEXT NOT NULL DEFAULT 'translation',
                        sentence_id INTEGER,
                        quality REAL,
                        score_before REAL,
                        score_after REAL,
                        delta REAL,
                        is_skipped BOOLEAN DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE SET NULL
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_history_user_mode ON exercise_history(user_id, mode);")

                # Audio cache table for synthesized speech files
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audio_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sentence_id INTEGER,
                        text_hash TEXT UNIQUE NOT NULL,
                        text TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audio_cache_hash ON audio_cache(text_hash);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audio_cache_sentence ON audio_cache(sentence_id);")

                # Seed user_modes for default_user if not exists
                trans_mode = conn.execute("SELECT mode FROM user_modes WHERE user_id = 'default_user' AND mode = 'translation';").fetchone()
                if not trans_mode:
                    u_row = conn.execute("SELECT score, level, streak, total_completed, total_skipped FROM users WHERE id = 'default_user';").fetchone()
                    u_score = u_row['score'] if u_row else 1.20
                    u_level = u_row['level'] if u_row else 'A1'
                    u_streak = u_row['streak'] if u_row else 0
                    u_completed = u_row['total_completed'] if u_row else 0
                    u_skipped = u_row['total_skipped'] if u_row else 0
                    conn.execute("""
                        INSERT OR IGNORE INTO user_modes (user_id, mode, score, level, streak, total_completed, total_skipped)
                        VALUES ('default_user', 'translation', ?, ?, ?, ?, ?);
                    """, (u_score, u_level, u_streak, u_completed, u_skipped))

                audio_mode = conn.execute("SELECT mode FROM user_modes WHERE user_id = 'default_user' AND mode = 'transcription';").fetchone()
                if not audio_mode:
                    conn.execute("""
                        INSERT OR IGNORE INTO user_modes (user_id, mode, score, level, streak, total_completed, total_skipped)
                        VALUES ('default_user', 'transcription', 1.20, 'A1', 0, 0, 0);
                    """)
        except sqlite3.OperationalError:
            if os.path.exists(self.db_path):
                return
            raise

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        lex_score = d.get('score_lexical')
        if lex_score is None:
            lex_score = d.get('score', 1.0)
        try:
            score_float = float(lex_score)
        except (ValueError, TypeError):
            score_float = 1.0
        d['score'] = score_float
        d['score_lexical'] = score_float
        return d

    def get_existing_columns(self) -> List[str]:
        with self._get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(sentences);")
            return [col['name'] for col in cursor.fetchall()]

    def get_languages(self) -> List[str]:
        """Returns list of supported language columns, e.g. ['fr', 'es', 'en']."""
        cols = self.get_existing_columns()
        return [c for c in cols if c not in SYSTEM_COLUMNS]

    def add_language_column(self, lang_code: str):
        """Dynamically adds a new language column to the sentences table."""
        lang_code = lang_code.lower().strip()
        if not re.match(r'^[a-z0-9_]{2,10}$', lang_code):
            raise ValueError(f"Invalid language code: {lang_code}")

        existing = self.get_existing_columns()
        if lang_code in existing:
            return

        with self._get_connection() as conn:
            conn.execute(f"ALTER TABLE sentences ADD COLUMN {lang_code} TEXT DEFAULT NULL;")

    @property
    def sentences(self) -> List[Dict[str, Any]]:
        """Backward compatible property returning all sentences."""
        return self.get_all_sentences()

    @sentences.setter
    def sentences(self, new_sentences: List[Dict[str, Any]]):
        """
        Allows assigning a new list of sentences (used in batch reload/saving).
        If an empty list is assigned, clears sentences.
        """
        if not new_sentences:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM sentences;")
            return
        self.bulk_insert_sentences(new_sentences, clear_existing=True)

    def load_data(self) -> List[Dict[str, Any]]:
        return self.get_all_sentences()

    def save_data(self):
        """No-op kept for backward compatibility since changes persist immediately in SQLite."""
        pass

    def get_all_sentences(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if level and level.upper() != 'ALL':
                cursor = conn.execute(
                    "SELECT * FROM sentences WHERE UPPER(level) = ? ORDER BY id ASC;",
                    (level.upper().strip(),)
                )
            else:
                cursor = conn.execute("SELECT * FROM sentences ORDER BY id ASC;")
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_sentence_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM sentences WHERE id = ?;", (item_id,))
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None

    def get_level_distribution(self) -> Dict[str, int]:
        dist = {'A1': 0, 'A2': 0, 'B1': 0, 'B2': 0, 'C1': 0, 'C2': 0}
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT UPPER(level) as lvl, COUNT(*) as cnt FROM sentences GROUP BY UPPER(level);")
            for row in cursor.fetchall():
                lvl = row['lvl']
                if lvl in dist:
                    dist[lvl] = row['cnt']
                else:
                    dist[lvl] = row['cnt']
        return dist

    def add_sentence(
        self,
        es: str,
        fr: str,
        level: str = 'A1',
        score: Optional[float] = None,
        score_lexical: Optional[float] = None,
        score_ml: Optional[float] = None,
        source: str = 'custom',
        metadata: Optional[Dict[str, Any]] = None,
        **extra_langs
    ) -> Dict[str, Any]:
        """
        Adds a single sentence pair (or multi-language tuple) to the database.
        """
        lex_score = score_lexical if score_lexical is not None else (score if score is not None else 1.0)
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)

        # Check and ensure extra language columns exist
        for lang, val in extra_langs.items():
            if val is not None:
                self.add_language_column(lang)

        cols = ['fr', 'es', 'level', 'score_lexical', 'score_ml', 'source', 'metadata']
        vals = [fr, es, level, float(lex_score), float(score_ml) if score_ml is not None else None, source, meta_str]

        for lang, val in extra_langs.items():
            cols.append(lang.lower())
            vals.append(val)

        col_str = ", ".join(cols)
        placeholder_str = ", ".join(["?"] * len(cols))

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO sentences ({col_str}) VALUES ({placeholder_str});",
                vals
            )
            new_id = cursor.lastrowid

        return self.get_sentence_by_id(new_id)

    def bulk_insert_sentences(
        self,
        sentences: List[Dict[str, Any]],
        clear_existing: bool = False
    ) -> int:
        """
        Efficiently inserts or updates a large batch of sentences.
        """
        if not sentences:
            return 0

        # Collect any additional language columns
        existing_cols = set(self.get_existing_columns())
        for s in sentences:
            for k in s.keys():
                if k not in SYSTEM_COLUMNS and k not in existing_cols and k not in ('score',):
                    self.add_language_column(k)
                    existing_cols.add(k)

        cols_to_insert = ['fr', 'es', 'level', 'score_lexical', 'score_ml', 'source', 'metadata']
        extra_lang_cols = [c for c in existing_cols if c not in SYSTEM_COLUMNS and c not in ('fr', 'es')]
        all_cols = cols_to_insert + extra_lang_cols

        if any('id' in s and s['id'] is not None for s in sentences):
            all_cols = ['id'] + all_cols

        col_str = ", ".join(all_cols)
        placeholder_str = ", ".join(["?"] * len(all_cols))
        sql = f"INSERT OR REPLACE INTO sentences ({col_str}) VALUES ({placeholder_str});"

        rows = []
        for s in sentences:
            lex_score = s.get('score_lexical')
            if lex_score is None:
                lex_score = s.get('score', 1.0)

            ml_score = s.get('score_ml')
            meta = s.get('metadata', {})
            meta_str = json.dumps(meta) if isinstance(meta, dict) else str(meta)

            row = []
            if 'id' in all_cols:
                row.append(s.get('id'))
            row.extend([
                s.get('fr', ''),
                s.get('es', ''),
                s.get('level', 'A1'),
                float(lex_score),
                float(ml_score) if ml_score is not None else None,
                s.get('source', 'custom'),
                meta_str
            ])
            for lang in extra_lang_cols:
                row.append(s.get(lang))
            rows.append(row)

        with self._get_connection() as conn:
            if clear_existing:
                conn.execute("DELETE FROM sentences;")
            conn.executemany(sql, rows)

        return len(rows)

    def update_sentence(
        self,
        item_id: int,
        es: Optional[str] = None,
        fr: Optional[str] = None,
        level: Optional[str] = None,
        score: Optional[float] = None,
        score_lexical: Optional[float] = None,
        score_ml: Optional[float] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **extra_langs
    ) -> bool:
        updates = []
        params = []

        if es is not None:
            updates.append("es = ?")
            params.append(es)
        if fr is not None:
            updates.append("fr = ?")
            params.append(fr)
        if level is not None:
            updates.append("level = ?")
            params.append(level)

        lex_score = score_lexical if score_lexical is not None else score
        if lex_score is not None:
            updates.append("score_lexical = ?")
            params.append(float(lex_score))

        if score_ml is not None:
            updates.append("score_ml = ?")
            params.append(float(score_ml))

        if source is not None:
            updates.append("source = ?")
            params.append(source)

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata, ensure_ascii=False))

        for lang, val in extra_langs.items():
            self.add_language_column(lang)
            updates.append(f"{lang.lower()} = ?")
            params.append(val)

        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        set_str = ", ".join(updates)
        params.append(item_id)

        with self._get_connection() as conn:
            cursor = conn.execute(f"UPDATE sentences SET {set_str} WHERE id = ?;", params)
            return cursor.rowcount > 0

    def delete_sentence(self, item_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM sentences WHERE id = ?;", (item_id,))
            return cursor.rowcount > 0

    # ==========================================
    # User Management Methods
    # ==========================================

    def get_or_create_user(self, user_id: str = 'default_user', name: str = 'Default User', avatar: str = 'owl') -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE id = ?;", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            conn.execute(
                "INSERT INTO users (id, name, avatar, score, level) VALUES (?, ?, ?, 1.20, 'A1');",
                (user_id, name, avatar)
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
                            "score": d["audio_score"] if d["audio_score"] is not None else 1.20,
                            "level": d["audio_level"] or 'A1',
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
                VALUES (?, ?, ?, 1.20, 'A1', 0, 0, 0);
            """, (user_id, clean_name, clean_avatar))
            
            conn.execute("""
                INSERT OR IGNORE INTO user_modes (user_id, mode, score, level, streak, total_completed, total_skipped)
                VALUES (?, 'translation', 1.20, 'A1', 0, 0, 0);
            """, (user_id,))
            conn.execute("""
                INSERT OR IGNORE INTO user_modes (user_id, mode, score, level, streak, total_completed, total_skipped)
                VALUES (?, 'transcription', 1.20, 'A1', 0, 0, 0);
            """, (user_id,))

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
                VALUES (?, ?, 1.20, 'A1', 0, 0, 0);
                """,
                (user_id, mode)
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
        start_score: float = 1.20,
        level: str = 'A1'
    ) -> Dict[str, Any]:
        """Resets the score and statistics for a given mode."""
        start_score = max(1.0, min(6.0, float(start_score)))
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE user_modes
                SET score = ?, level = ?, streak = 0, total_completed = 0, total_skipped = 0, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND mode = ?;
            """, (start_score, level, user_id, mode))
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
            "score": trans_mode.get("score", 1.20),
            "level": trans_mode.get("level", "A1"),
            "streak": trans_mode.get("streak", 0),
            "total_completed": trans_mode.get("total_completed", 0),
            "total_skipped": trans_mode.get("total_skipped", 0),
            "modes": {
                "translation": trans_mode,
                "transcription": audio_mode
            }
        }

    def record_exercise_history(
        self,
        user_id: str = 'default_user',
        mode: str = 'translation',
        sentence_id: Optional[int] = None,
        quality: float = 0.0,
        score_before: float = 1.20,
        score_after: float = 1.20,
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
        limit: int = 50
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

    def get_audio_cache(self, text_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached audio record by text_hash if it exists on disk."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM audio_cache WHERE text_hash = ?;", (text_hash,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                if os.path.exists(d.get("file_path", "")):
                    return d
            return None

    def save_audio_cache(self, text_hash: str, text: str, file_path: str, sentence_id: Optional[int] = None):
        """Saves a synthesized audio file mapping into the audio_cache table."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO audio_cache (sentence_id, text_hash, text, file_path)
                VALUES (?, ?, ?, ?);
            """, (sentence_id, text_hash, text, file_path))

    # ==========================================
    # Lexemas & Sentence Junction Methods
    # ==========================================

    def get_lexemas_dict(self) -> Dict[int, Dict[str, Any]]:
        """Returns in-memory cached dictionary of all lexemas {id: {lemma, level, is_base}}."""
        with self._lexemas_cache_lock:
            if self._lexemas_cache is None:
                with self._get_connection() as conn:
                    lex_cursor = conn.execute("SELECT id, lemma, level, is_base FROM lexemas;")
                    self._lexemas_cache = {
                        r['id']: {'lemma': r['lemma'], 'level': r['level'], 'is_base': bool(r['is_base'])}
                        for r in lex_cursor.fetchall()
                    }
            return self._lexemas_cache

    def invalidate_lexemas_cache(self):
        """Invalidates the in-memory lexemas cache."""
        with self._lexemas_cache_lock:
            self._lexemas_cache = None

    def upsert_lexema(
        self,
        lemma: str,
        pos: Optional[str] = None,
        level: str = 'A1',
        frequency: float = 0.0,
        is_base: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        lemma = lemma.lower().strip()
        pos = (pos or '').upper().strip() or None
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)
        self.invalidate_lexemas_cache()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO lexemas (lemma, pos, level, frequency, is_base, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(lemma, pos) DO UPDATE SET
                    level = CASE WHEN excluded.level != 'A1' THEN excluded.level ELSE lexemas.level END,
                    frequency = CASE WHEN excluded.frequency > lexemas.frequency THEN excluded.frequency ELSE lexemas.frequency END,
                    is_base = CASE WHEN excluded.is_base = 1 THEN 1 ELSE lexemas.is_base END;
            """, (lemma, pos, level, float(frequency), 1 if is_base else 0, meta_str))
            cursor = conn.execute("SELECT id FROM lexemas WHERE lemma = ? AND (pos = ? OR (pos IS NULL AND ? IS NULL));", (lemma, pos, pos))
            row = cursor.fetchone()
            return row['id'] if row else -1

    def bulk_upsert_lexemas(self, lexemas: List[Dict[str, Any]]) -> Dict[tuple, int]:
        if not lexemas:
            return {}
        self.invalidate_lexemas_cache()
        with self._get_connection() as conn:
            for item in lexemas:
                lemma = item['lemma'].lower().strip()
                pos = (item.get('pos') or '').upper().strip() or None
                lvl = item.get('level', 'A1')
                freq = float(item.get('frequency', item.get('freq', 0.0)))
                is_base = 1 if item.get('is_base') else 0
                meta = json.dumps(item.get('metadata', {}), ensure_ascii=False)
                conn.execute("""
                    INSERT INTO lexemas (lemma, pos, level, frequency, is_base, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(lemma, pos) DO UPDATE SET
                        level = CASE WHEN excluded.level != 'A1' THEN excluded.level ELSE lexemas.level END,
                        frequency = CASE WHEN excluded.frequency > lexemas.frequency THEN excluded.frequency ELSE lexemas.frequency END,
                        is_base = CASE WHEN excluded.is_base = 1 THEN 1 ELSE lexemas.is_base END;
                """, (lemma, pos, lvl, freq, is_base, meta))

            # Query back ids
            cursor = conn.execute("SELECT id, lemma, pos FROM lexemas;")
            return {(r['lemma'], r['pos']): r['id'] for r in cursor.fetchall()}

    def link_sentence_lexemas(self, sentence_id: int, lexema_ids: List[int]):
        if not lexema_ids:
            return
        with self._get_connection() as conn:
            rows = [(sentence_id, lid) for lid in set(lexema_ids)]
            conn.executemany(
                "INSERT OR IGNORE INTO sentence_lexemas (sentence_id, lexema_id) VALUES (?, ?);",
                rows
            )

    def bulk_link_sentence_lexemas(self, links: List[tuple]):
        if not links:
            return
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO sentence_lexemas (sentence_id, lexema_id) VALUES (?, ?);",
                links
            )

    def get_sentence_lexemas(self, sentence_id: int) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT l.* FROM lexemas l
                JOIN sentence_lexemas sl ON l.id = sl.lexema_id
                WHERE sl.sentence_id = ?
                ORDER BY l.lemma ASC;
            """, (sentence_id,))
            return [dict(r) for r in cursor.fetchall()]

    # ==========================================
    # Vocabulary Tracking & Appearance Counting
    # ==========================================

    def get_user_known_lexema_ids(self, user_id: str = 'default_user', unlock_threshold: int = 2) -> Set[int]:
        """
        Returns the set of lexema IDs known by the user:
        - Base foundational words (is_base = 1)
        - Words marked is_known = 1
        - Words with appearance_count >= unlock_threshold
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT id FROM lexemas WHERE is_base = 1
                UNION
                SELECT lexema_id FROM user_lexema_counts
                WHERE user_id = ? AND (is_known = 1 OR appearance_count >= ?);
            """, (user_id, unlock_threshold))
            return {r[0] for r in cursor.fetchall()}

    def record_lexema_appearances(
        self,
        user_id: str,
        sentence_id: int,
        is_success: bool = True,
        unlock_threshold: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Increments the appearance count for every word root in the given sentence.
        If is_success is True or appearance threshold is reached, sets is_known = 1.
        Returns the updated list of lexemas with their new counts.
        """
        self.get_or_create_user(user_id)
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT lexema_id FROM sentence_lexemas WHERE sentence_id = ?;",
                (sentence_id,)
            )
            lexema_ids = [r[0] for r in cursor.fetchall()]
            if not lexema_ids:
                return []

            success_inc = 1 if is_success else 0
            for lid in lexema_ids:
                conn.execute("""
                    INSERT INTO user_lexema_counts (user_id, lexema_id, appearance_count, success_count, is_known, last_seen_at)
                    VALUES (?, ?, 1, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, lexema_id) DO UPDATE SET
                        appearance_count = user_lexema_counts.appearance_count + 1,
                        success_count = user_lexema_counts.success_count + ?,
                        is_known = CASE
                            WHEN user_lexema_counts.is_known = 1 THEN 1
                            WHEN (user_lexema_counts.appearance_count + 1) >= ? THEN 1
                            WHEN (user_lexema_counts.success_count + ?) >= 1 THEN 1
                            ELSE 0
                        END,
                        last_seen_at = CURRENT_TIMESTAMP;
                """, (user_id, lid, success_inc, 1 if success_inc else 0, success_inc, unlock_threshold, success_inc))

            placeholders = ",".join(["?"] * len(lexema_ids))
            cursor = conn.execute(f"""
                SELECT l.id, l.lemma, l.pos, l.level, ulc.appearance_count, ulc.success_count, ulc.is_known
                FROM lexemas l
                JOIN user_lexema_counts ulc ON l.id = ulc.lexema_id
                WHERE ulc.user_id = ? AND l.id IN ({placeholders});
            """, [user_id] + lexema_ids)
            return [dict(r) for r in cursor.fetchall()]

    def get_user_vocabulary_stats(self, user_id: str = 'default_user') -> Dict[str, Any]:
        """Returns vocabulary progress diagnostics for the user."""
        with self._get_connection() as conn:
            known_cursor = conn.execute("""
                SELECT COUNT(DISTINCT id) FROM (
                    SELECT id FROM lexemas WHERE is_base = 1
                    UNION
                    SELECT lexema_id as id FROM user_lexema_counts WHERE user_id = ? AND is_known = 1
                );
            """, (user_id,))
            total_known = known_cursor.fetchone()[0]

            learning_cursor = conn.execute("""
                SELECT COUNT(*) FROM user_lexema_counts
                WHERE user_id = ? AND is_known = 0 AND appearance_count > 0;
            """, (user_id,))
            total_learning = learning_cursor.fetchone()[0]

            total_roots_cursor = conn.execute("SELECT COUNT(*) FROM lexemas;")
            total_roots = total_roots_cursor.fetchone()[0]

            lvl_cursor = conn.execute("""
                SELECT l.level, COUNT(DISTINCT l.id) as cnt
                FROM lexemas l
                WHERE l.is_base = 1 OR l.id IN (
                    SELECT lexema_id FROM user_lexema_counts WHERE user_id = ? AND is_known = 1
                )
                GROUP BY l.level;
            """, (user_id,))
            by_level = {r['level']: r['cnt'] for r in lvl_cursor.fetchall()}

            recent_cursor = conn.execute("""
                SELECT l.lemma, l.level, ulc.appearance_count, ulc.success_count, ulc.is_known, ulc.last_seen_at
                FROM user_lexema_counts ulc
                JOIN lexemas l ON ulc.lexema_id = l.id
                WHERE ulc.user_id = ?
                ORDER BY ulc.last_seen_at DESC LIMIT 10;
            """, (user_id,))
            recent_roots = [dict(r) for r in recent_cursor.fetchall()]

            return {
                "user_id": user_id,
                "total_known": total_known,
                "total_learning": total_learning,
                "total_available_roots": total_roots,
                "by_level": by_level,
                "recent_roots": recent_roots
            }

    # ==========================================
    # Progressive Sentence Selection ($i + 1$)
    # ==========================================

    def get_progressive_sentences(
        self,
        user_id: str = 'default_user',
        target_score: float = 1.20,
        count: int = 3,
        max_unknown: int = 2,
        exclude_ids: Optional[List[int]] = None,
        mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Progressively picks sentences where the user knows almost all words,
        and only a small number (0 to max_unknown) of words they do not know.
        If max_unknown == 0 or mode == 'transcription', guarantees that returned
        sentences contain 0 unknown words (only words the user is familiar with).
        """
        import random
        if mode == 'transcription':
            max_unknown = 0

        known_lexema_ids = self.get_user_known_lexema_ids(user_id)
        exclude_set = set(exclude_ids or [])

        # Tiered candidate query to leverage idx_sentences_score_lexical range scans
        min_candidates = max(10, count * 4)
        with self._get_connection() as conn:
            # Tier 1: narrow window (+/- 0.8)
            cursor = conn.execute("""
                SELECT s.*, GROUP_CONCAT(sl.lexema_id) as lexema_ids
                FROM sentences s
                JOIN sentence_lexemas sl ON s.id = sl.sentence_id
                WHERE s.score_lexical BETWEEN ? AND ?
                GROUP BY s.id
                ORDER BY ABS(s.score_lexical - ?) ASC
                LIMIT 150;
            """, (target_score - 0.8, target_score + 0.8, target_score))
            all_rows = cursor.fetchall()

            # Tier 2: wider window (+/- 1.6) if insufficient candidates
            if len(all_rows) < min_candidates:
                cursor = conn.execute("""
                    SELECT s.*, GROUP_CONCAT(sl.lexema_id) as lexema_ids
                    FROM sentences s
                    JOIN sentence_lexemas sl ON s.id = sl.sentence_id
                    WHERE s.score_lexical BETWEEN ? AND ?
                    GROUP BY s.id
                    ORDER BY ABS(s.score_lexical - ?) ASC
                    LIMIT 250;
                """, (target_score - 1.6, target_score + 1.6, target_score))
                all_rows = cursor.fetchall()

            # Tier 3: full corpus fallback if still insufficient
            if len(all_rows) < min_candidates:
                cursor = conn.execute("""
                    SELECT s.*, GROUP_CONCAT(sl.lexema_id) as lexema_ids
                    FROM sentences s
                    JOIN sentence_lexemas sl ON s.id = sl.sentence_id
                    GROUP BY s.id
                    ORDER BY ABS(s.score_lexical - ?) ASC
                    LIMIT 350;
                """, (target_score,))
                all_rows = cursor.fetchall()

        if not all_rows:
            all_s = self.get_all_sentences()
            candidates = [s for s in all_s if s['id'] not in exclude_set]
            if not candidates:
                candidates = all_s
            return candidates[:count]

        reinforcement_candidates = []
        excluded_reinforcement = []
        progression_candidates = []
        fallback_candidates = []

        lex_dict = self.get_lexemas_dict()
        with self._get_connection() as conn:
            user_counts_cursor = conn.execute(
                "SELECT lexema_id, appearance_count, is_known FROM user_lexema_counts WHERE user_id = ?;",
                (user_id,)
            )
            user_counts = {r['lexema_id']: {'count': r['appearance_count'], 'is_known': bool(r['is_known'])} for r in user_counts_cursor.fetchall()}

        for row in all_rows:
            raw_lids = row['lexema_ids']
            if not raw_lids:
                continue

            sent_lids = {int(x) for x in raw_lids.split(',') if x}
            if not sent_lids:
                continue

            s_dict = self._row_to_dict(row)
            is_excluded = s_dict['id'] in exclude_set
            unknown_lids = sent_lids - known_lexema_ids

            s_dict['unknown_count'] = len(unknown_lids)
            s_dict['_sent_lids'] = sent_lids
            s_dict['_unknown_lids'] = unknown_lids
            score_diff = abs(s_dict.get('score', 1.0) - target_score)
            s_dict['_score_diff'] = score_diff

            if len(unknown_lids) == 0:
                if not is_excluded:
                    reinforcement_candidates.append(s_dict)
                else:
                    excluded_reinforcement.append(s_dict)
            elif max_unknown > 0:
                if not is_excluded:
                    if 1 <= len(unknown_lids) <= max_unknown:
                        progression_candidates.append(s_dict)
                    else:
                        fallback_candidates.append(s_dict)

        reinforcement_candidates.sort(key=lambda x: x['_score_diff'])
        excluded_reinforcement.sort(key=lambda x: x['_score_diff'])
        progression_candidates.sort(key=lambda x: (x['_score_diff'], x['unknown_count']))
        fallback_candidates.sort(key=lambda x: (x['unknown_count'], x['_score_diff']))

        selected = []
        if max_unknown == 0:
            # Transcription / familiar words only: strictly zero unknown words
            top_reinf = reinforcement_candidates[:max(10, count * 3)]
            random.shuffle(top_reinf)
            selected.extend(top_reinf[:count])

            # If not enough unexcluded familiar sentences, recycle familiar sentences that were excluded
            needed = count - len(selected)
            if needed > 0 and excluded_reinforcement:
                top_ex_reinf = excluded_reinforcement[:max(10, needed * 3)]
                random.shuffle(top_ex_reinf)
                selected.extend(top_ex_reinf[:needed])

            # In transcription mode, if still needed, search full corpus for familiar sentences
            if len(selected) < count:
                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        SELECT s.*, GROUP_CONCAT(sl.lexema_id) as lexema_ids
                        FROM sentences s
                        JOIN sentence_lexemas sl ON s.id = sl.sentence_id
                        GROUP BY s.id
                        ORDER BY ABS(s.score_lexical - ?) ASC;
                    """, (target_score,))
                    extra_rows = cursor.fetchall()
                selected_ids = {s['id'] for s in selected}
                for row in extra_rows:
                    raw_lids = row['lexema_ids']
                    if not raw_lids:
                        continue
                    sent_lids = {int(x) for x in raw_lids.split(',') if x}
                    if not sent_lids:
                        continue
                    if sent_lids.issubset(known_lexema_ids):
                        s_dict = self._row_to_dict(row)
                        if s_dict['id'] not in selected_ids:
                            s_dict['unknown_count'] = 0
                            s_dict['_sent_lids'] = sent_lids
                            s_dict['_unknown_lids'] = set()
                            selected.append(s_dict)
                            selected_ids.add(s_dict['id'])
                            if len(selected) == count:
                                break
        else:
            num_prog = max(1, round(count * 0.67))
            num_reinf = count - num_prog

            top_prog = progression_candidates[:max(10, num_prog * 3)]
            random.shuffle(top_prog)
            selected.extend(top_prog[:num_prog])

            needed_reinf = count - len(selected)
            top_reinf = reinforcement_candidates[:max(10, needed_reinf * 3)]
            random.shuffle(top_reinf)
            selected.extend(top_reinf[:needed_reinf])

            if len(selected) < count:
                selected_ids = {s['id'] for s in selected}
                for pool in [progression_candidates, reinforcement_candidates, fallback_candidates]:
                    for item in pool:
                        if item['id'] not in selected_ids:
                            selected.append(item)
                            selected_ids.add(item['id'])
                            if len(selected) == count:
                                break
                    if len(selected) == count:
                        break

        # Resolve accurate token derivations only for the final selected sentences
        try:
            from backend.vocab_engine import get_vocab_engine
        except ImportError:
            try:
                from vocab_engine import get_vocab_engine
            except ImportError:
                get_vocab_engine = None

        engine = None
        if get_vocab_engine is not None:
            try:
                engine = get_vocab_engine()
            except Exception:
                engine = None

        for s in selected:
            sent_lids = s.pop('_sent_lids', set())
            unknown_lids = s.pop('_unknown_lids', set())
            known_lids = sent_lids - unknown_lids

            s['known_lexemas'] = [lex_dict[lid]['lemma'] for lid in known_lids if lid in lex_dict]
            s['new_lexemas'] = [lex_dict[lid]['lemma'] for lid in unknown_lids if lid in lex_dict]

            first_time_lids = [
                lid for lid in sent_lids
                if not lex_dict.get(lid, {}).get('is_base', False)
                and (lid not in user_counts or user_counts[lid]['count'] <= 2)
            ]
            s['first_time_lexemas'] = [lex_dict[lid]['lemma'] for lid in first_time_lids if lid in lex_dict]
            if not s['first_time_lexemas'] and s['new_lexemas']:
                s['first_time_lexemas'] = s['new_lexemas']

            s.pop('_score_diff', None)

            # Match surface words in target French sentence for each first-time root
            pairs = []
            if s.get('first_time_lexemas'):
                if engine is not None:
                    try:
                        pairs = engine.extract_sentence_token_pairs(s.get('fr', ''), s['first_time_lexemas'])
                    except Exception:
                        pairs = []
                if not pairs:
                    fr_tokens = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", s.get('fr', ''), re.UNICODE)
                    for lem in s['first_time_lexemas']:
                        lem_clean = re.sub(r"^(se\s+|s['’]|l['’]|d['’]|j['’]|c['’]|m['’]|n['’]|t['’])", "", lem.lower()).strip()
                        matched_word = lem_clean
                        for tok in fr_tokens:
                            tok_clean = re.sub(r"^[ldjcmnstLDJCMNST]['’]", "", tok).lower().strip()
                            if not tok_clean:
                                continue
                            if tok_clean == lem_clean:
                                matched_word = re.sub(r"^[ldjcmnstLDJCMNST]['’]", "", tok)
                                break
                            min_len = min(len(tok_clean), len(lem_clean))
                            if min_len >= 4 and tok_clean[:4] == lem_clean[:4]:
                                matched_word = re.sub(r"^[ldjcmnstLDJCMNST]['’]", "", tok)
                            elif min_len >= 3 and tok_clean[:3] == lem_clean[:3] and abs(len(tok_clean) - len(lem_clean)) <= 2:
                                matched_word = re.sub(r"^[ldjcmnstLDJCMNST]['’]", "", tok)
                        pairs.append({'word': matched_word, 'lemma': lem})
            s['first_time_pairs'] = pairs

        return selected
