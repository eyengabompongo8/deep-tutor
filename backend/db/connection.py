import os
import sqlite3
import threading
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from contextlib import contextmanager

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.environ.get("FRENCHTUTOR_DB_PATH") or os.path.join(ROOT_DIR, 'data', 'app.db')

SYSTEM_COLUMNS = {
    'id', 'level', 'score_lexical', 'score_ml', 'source',
    'metadata', 'created_at', 'updated_at'
}

class DBConnection:
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
                        level TEXT DEFAULT 'A1',
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

                try:
                    ulc_cols = [c[1] for c in conn.execute("PRAGMA table_info(user_lexema_counts);").fetchall()]
                    if 'level' not in ulc_cols:
                        conn.execute("ALTER TABLE user_lexema_counts ADD COLUMN level TEXT DEFAULT 'A1';")
                except Exception:
                    pass

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
