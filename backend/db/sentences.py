import json
import re
import sqlite3
from typing import List, Dict, Any, Optional
from backend.db.connection import SYSTEM_COLUMNS

class SentencesMixin:
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
                    "SELECT * FROM sentences WHERE level = ? COLLATE NOCASE ORDER BY id ASC;",
                    (level.strip(),)
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

        set_str = ", ".join(updates)
        params.append(item_id)

        with self._get_connection() as conn:
            cursor = conn.execute(f"UPDATE sentences SET {set_str} WHERE id = ?;", params)
            return cursor.rowcount > 0

    def delete_sentence(self, item_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM sentences WHERE id = ?;", (item_id,))
            return cursor.rowcount > 0
