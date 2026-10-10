import os
import sqlite3
import threading
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from contextlib import contextmanager

from backend.config import (
    DEFAULT_CEFR_LEVEL,
    DEFAULT_CORRECTION_THRESHOLD,
    DEFAULT_CARDS_PER_EXERCISE,
    DEFAULT_MIXED_TRANSLATION_RATIO,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.environ.get("DEEPTUTOR_DB_PATH") or os.environ.get("FRENCHTUTOR_DB_PATH") or os.path.join(ROOT_DIR, 'data', 'app.db')

SYSTEM_COLUMNS = {
    'id', 'level', 'score_ml', 'source',
    'metadata'
}

class DBConnection:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DB_PATH
        self._lemmas_cache: Optional[Dict[int, Dict[str, Any]]] = None
        self._lemmas_cache_lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    @property
    def _lexemas_cache(self) -> Optional[Dict[int, Dict[str, Any]]]:
        return self._lemmas_cache

    @_lexemas_cache.setter
    def _lexemas_cache(self, value: Optional[Dict[int, Dict[str, Any]]]):
        self._lemmas_cache = value

    @property
    def _lexemas_cache_lock(self) -> threading.Lock:
        return self._lemmas_cache_lock

    @contextmanager
    def _get_connection(self):
        conn = None
        try:
            try:
                conn = sqlite3.connect(self.db_path, timeout=15.0)
            except sqlite3.OperationalError as e:
                if os.path.exists(self.db_path):
                    uri_path = f"file:{os.path.abspath(self.db_path)}?immutable=1"
                    conn = sqlite3.connect(uri_path, uri=True, timeout=15.0)
                else:
                    raise e
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 10000;")
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()
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
        self._set_wal_mode()
        try:
            with self._get_connection() as conn:
                self._create_tables(conn)
                self._run_migrations(conn)
                self._seed_defaults(conn)
        except sqlite3.OperationalError:
            if os.path.exists(self.db_path):
                return
            raise

    def _set_wal_mode(self):
        try:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute("PRAGMA foreign_keys = ON;")
        except sqlite3.OperationalError:
            pass

    def _create_tables(self, conn: sqlite3.Connection):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fr TEXT NOT NULL,
                es TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'A1' COLLATE NOCASE,
                score_ml REAL DEFAULT NULL,
                source TEXT NOT NULL DEFAULT 'custom',
                metadata TEXT DEFAULT '{}'
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_level ON sentences(level COLLATE NOCASE);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_score_ml ON sentences(score_ml);")

        # Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT 'Default User'
            );
        """)

        # Lemmas table (canonical French word roots)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lemmas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                pos TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'A1',
                frequency REAL DEFAULT 0.0,
                is_base BOOLEAN NOT NULL DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(lemma, pos)
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lemmas_pos ON lemmas(pos);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lemmas_level ON lemmas(level);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lemmas_is_base ON lemmas(is_base);")

        # Words table (inflected surface forms linked to lemmas)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL,
                lemma_id INTEGER NOT NULL,
                pos TEXT NOT NULL,
                frequency REAL DEFAULT 0.0,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (lemma_id) REFERENCES lemmas(id) ON DELETE CASCADE,
                UNIQUE(word, lemma_id, pos)
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_words_pos ON words(pos);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_words_lemma_id ON words(lemma_id);")

        # Sentence-Lemmas junction table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentence_lemmas (
                sentence_id INTEGER NOT NULL,
                lemma_id INTEGER NOT NULL,
                PRIMARY KEY (sentence_id, lemma_id),
                FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE,
                FOREIGN KEY (lemma_id) REFERENCES lemmas(id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sentence_lemmas_lemma ON sentence_lemmas(lemma_id);")

        # Sentence-Words junction table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentence_words (
                sentence_id INTEGER NOT NULL,
                word_id INTEGER NOT NULL,
                PRIMARY KEY (sentence_id, word_id),
                FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE,
                FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sentence_words_word ON sentence_words(word_id);")

        # User lemma tracking & appearance counts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_lemmas (
                user_id TEXT NOT NULL,
                lemma_id INTEGER NOT NULL,
                level TEXT DEFAULT 'A1',
                appearance_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, lemma_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (lemma_id) REFERENCES lemmas(id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_lemmas_user_success ON user_lemmas(user_id, success_count, lemma_id);")

        # User word tracking & appearance counts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_words (
                user_id TEXT NOT NULL,
                word_id INTEGER NOT NULL,
                appearance_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, word_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE
            );
        """)

        # User modes table for separate exercise mode tracking
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS user_modes (
                user_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT '{DEFAULT_CEFR_LEVEL}',
                streak INTEGER NOT NULL DEFAULT 0,
                total_completed INTEGER NOT NULL DEFAULT 0,
                total_skipped INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, mode),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # Exercise history table for tracking progress and batches
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exercise_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'translation',
                sentence_id INTEGER,
                quality REAL,
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

        # User settings table
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                theme TEXT DEFAULT 'system',
                preferred_voice TEXT DEFAULT 'default',
                correction_threshold REAL DEFAULT {DEFAULT_CORRECTION_THRESHOLD},
                cards_per_exercise INTEGER DEFAULT {DEFAULT_CARDS_PER_EXERCISE},
                auto_jump INTEGER DEFAULT 1,
                auto_play_audio INTEGER DEFAULT 1,
                transcription_retry INTEGER DEFAULT 1,
                blind_retry INTEGER DEFAULT 0,
                active_mode TEXT DEFAULT 'translation',
                mixed_translation_ratio INTEGER DEFAULT {DEFAULT_MIXED_TRANSLATION_RATIO},
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

    def _run_migrations(self, conn: sqlite3.Connection):
        try:
            # 1. Drop redundant indexes if they exist
            redundant_indexes = [
                "idx_sentence_lemmas_sentence",
                "idx_sentence_words_sentence",
                "idx_user_lemmas_user",
                "idx_user_words_user",
                "idx_user_modes_user",
                "idx_lemmas_lemma",
                "idx_words_word",
            ]
            for idx_name in redundant_indexes:
                conn.execute(f"DROP INDEX IF EXISTS {idx_name};")

            # 2. Recreate sentences(level) index with COLLATE NOCASE
            conn.execute("DROP INDEX IF EXISTS idx_sentences_level;")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_level ON sentences(level COLLATE NOCASE);")

            # 3. Create composite covering index for user_lemmas CTE lookup
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_lemmas_user_success
                ON user_lemmas(user_id, success_count, lemma_id);
            """)
        except Exception:
            pass

        # Dynamic migration check for user_settings columns
        try:
            settings_cols = [c[1] for c in conn.execute("PRAGMA table_info(user_settings);").fetchall()]
            if 'auto_jump' not in settings_cols:
                conn.execute("ALTER TABLE user_settings ADD COLUMN auto_jump INTEGER DEFAULT 1;")
            if 'auto_play_audio' not in settings_cols:
                conn.execute("ALTER TABLE user_settings ADD COLUMN auto_play_audio INTEGER DEFAULT 1;")
            if 'transcription_retry' not in settings_cols:
                conn.execute("ALTER TABLE user_settings ADD COLUMN transcription_retry INTEGER DEFAULT 1;")
            if 'blind_retry' not in settings_cols:
                conn.execute("ALTER TABLE user_settings ADD COLUMN blind_retry INTEGER DEFAULT 0;")
            if 'active_mode' not in settings_cols:
                conn.execute("ALTER TABLE user_settings ADD COLUMN active_mode TEXT DEFAULT 'translation';")
            if 'mixed_translation_ratio' not in settings_cols:
                conn.execute(f"ALTER TABLE user_settings ADD COLUMN mixed_translation_ratio INTEGER DEFAULT {DEFAULT_MIXED_TRANSLATION_RATIO};")
        except Exception:
            pass

    def _seed_defaults(self, conn: sqlite3.Connection):
        # Seed default user if not exists
        conn.execute("""
            INSERT OR IGNORE INTO users (id, name)
            VALUES ('default_user', 'Default User');
        """)

        # Seed user_modes for default_user if not exists
        trans_mode = conn.execute("SELECT mode FROM user_modes WHERE user_id = 'default_user' AND mode = 'translation';").fetchone()
        if not trans_mode:
            conn.execute("""
                INSERT OR IGNORE INTO user_modes (user_id, mode, level, streak, total_completed, total_skipped)
                VALUES ('default_user', 'translation', ?, 0, 0, 0);
            """, (DEFAULT_CEFR_LEVEL,))

        audio_mode = conn.execute("SELECT mode FROM user_modes WHERE user_id = 'default_user' AND mode = 'transcription';").fetchone()
        if not audio_mode:
            conn.execute("""
                INSERT OR IGNORE INTO user_modes (user_id, mode, level, streak, total_completed, total_skipped)
                VALUES ('default_user', 'transcription', ?, 0, 0, 0);
            """, (DEFAULT_CEFR_LEVEL,))

        # Seed user_settings for default_user if not exists
        conn.execute("""
            INSERT OR IGNORE INTO user_settings (user_id, theme, preferred_voice, correction_threshold, cards_per_exercise, auto_jump, auto_play_audio, transcription_retry, blind_retry, active_mode, mixed_translation_ratio)
            VALUES ('default_user', 'system', 'default', ?, ?, 1, 1, 1, 0, 'translation', ?);
        """, (DEFAULT_CORRECTION_THRESHOLD, DEFAULT_CARDS_PER_EXERCISE, DEFAULT_MIXED_TRANSLATION_RATIO))


    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        score_ml = d.get('score_ml')
        if score_ml is not None:
            try:
                score_float = float(score_ml)
            except (ValueError, TypeError):
                score_float = 1.0
        else:
            score_float = 1.0
        d['score'] = score_float
        return d
