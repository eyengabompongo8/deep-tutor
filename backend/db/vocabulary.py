import json
import os
import re
from typing import List, Dict, Any, Optional, Set
from backend.config import (
    VOCABULARY_UNLOCK_THRESHOLD,
    DEFAULT_CEFR_LEVEL,
    DEFAULT_CARDS_PER_EXERCISE,
    MAX_UNKNOWN_WORDS,
)

class VocabularyMixin:
    def get_lemmas_dict(self) -> Dict[int, Dict[str, Any]]:
        """Returns in-memory cached dictionary of all lemmas {id: {lemma, pos, level, is_base}}."""
        with self._lemmas_cache_lock:
            if self._lemmas_cache is None:
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT id, lemma, pos, level, is_base FROM lemmas;")
                    self._lemmas_cache = {
                        r['id']: {
                            'lemma': r['lemma'],
                            'pos': r['pos'],
                            'level': r['level'],
                            'is_base': bool(r['is_base'])
                        }
                        for r in cursor.fetchall()
                    }
            return self._lemmas_cache

    get_lexemas_dict = get_lemmas_dict

    def invalidate_lemmas_cache(self):
        """Invalidates the in-memory lemmas cache and candidate sentence buffer."""
        with self._lemmas_cache_lock:
            self._lemmas_cache = None
            self._user_candidates_buffer = {}

    invalidate_lexemas_cache = invalidate_lemmas_cache

    def upsert_lemma(
        self,
        lemma: str,
        pos: Optional[str] = None,
        level: str = DEFAULT_CEFR_LEVEL,
        frequency: float = 0.0,
        is_base: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        lemma = lemma.lower().strip()
        pos = (pos or 'X').upper().strip()
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)
        self.invalidate_lemmas_cache()
        with self._get_connection() as conn:
            conn.execute(f"""
                INSERT INTO lemmas (lemma, pos, level, frequency, is_base, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(lemma, pos) DO UPDATE SET
                    level = CASE WHEN excluded.level != '{DEFAULT_CEFR_LEVEL}' THEN excluded.level ELSE lemmas.level END,
                    frequency = CASE WHEN excluded.frequency > lemmas.frequency THEN excluded.frequency ELSE lemmas.frequency END,
                    is_base = CASE WHEN excluded.is_base = 1 THEN 1 ELSE lemmas.is_base END;
            """, (lemma, pos, level, float(frequency), 1 if is_base else 0, meta_str))
            cursor = conn.execute("SELECT id FROM lemmas WHERE lemma = ? AND pos = ?;", (lemma, pos))
            row = cursor.fetchone()
            return row['id'] if row else -1

    upsert_lexema = upsert_lemma

    def bulk_upsert_lemmas(self, lemmas: List[Dict[str, Any]]) -> Dict[tuple, int]:
        if not lemmas:
            return {}
        self.invalidate_lemmas_cache()
        with self._get_connection() as conn:
            for item in lemmas:
                lemma = item['lemma'].lower().strip()
                pos = (item.get('pos') or 'X').upper().strip()
                lvl = item.get('level', DEFAULT_CEFR_LEVEL)
                freq = float(item.get('frequency', item.get('freq', 0.0)))
                is_base = 1 if item.get('is_base') else 0
                meta = json.dumps(item.get('metadata', {}), ensure_ascii=False)
                conn.execute(f"""
                    INSERT INTO lemmas (lemma, pos, level, frequency, is_base, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(lemma, pos) DO UPDATE SET
                        level = CASE WHEN excluded.level != '{DEFAULT_CEFR_LEVEL}' THEN excluded.level ELSE lemmas.level END,
                        frequency = CASE WHEN excluded.frequency > lemmas.frequency THEN excluded.frequency ELSE lemmas.frequency END,
                        is_base = CASE WHEN excluded.is_base = 1 THEN 1 ELSE lemmas.is_base END;
                """, (lemma, pos, lvl, freq, is_base, meta))

            # Query back ids
            cursor = conn.execute("SELECT id, lemma, pos FROM lemmas;")
            return {(r['lemma'], r['pos']): r['id'] for r in cursor.fetchall()}

    bulk_upsert_lexemas = bulk_upsert_lemmas

    def upsert_word(
        self,
        word: str,
        lemma_id: int,
        pos: Optional[str] = None,
        frequency: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        word = word.lower().strip()
        pos = (pos or 'X').upper().strip()
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO words (word, lemma_id, pos, frequency, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(word, lemma_id, pos) DO UPDATE SET
                    frequency = CASE WHEN excluded.frequency > words.frequency THEN excluded.frequency ELSE words.frequency END;
            """, (word, lemma_id, pos, float(frequency), meta_str))
            cursor = conn.execute("SELECT id FROM words WHERE word = ? AND lemma_id = ? AND pos = ?;", (word, lemma_id, pos))
            row = cursor.fetchone()
            return row['id'] if row else -1

    def bulk_upsert_words(self, words: List[Dict[str, Any]]) -> Dict[tuple, int]:
        if not words:
            return {}
        with self._get_connection() as conn:
            for item in words:
                word = item['word'].lower().strip()
                lemma_id = item['lemma_id']
                pos = (item.get('pos') or 'X').upper().strip()
                freq = float(item.get('frequency', item.get('freq', 0.0)))
                meta = json.dumps(item.get('metadata', {}), ensure_ascii=False)
                conn.execute("""
                    INSERT INTO words (word, lemma_id, pos, frequency, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(word, lemma_id, pos) DO UPDATE SET
                        frequency = CASE WHEN excluded.frequency > words.frequency THEN excluded.frequency ELSE words.frequency END;
                """, (word, lemma_id, pos, freq, meta))

            cursor = conn.execute("SELECT id, word, lemma_id, pos FROM words;")
            return {(r['word'], r['lemma_id'], r['pos']): r['id'] for r in cursor.fetchall()}

    def link_sentence_lemmas(self, sentence_id: int, lemma_ids: List[int]):
        if not lemma_ids:
            return
        with self._get_connection() as conn:
            rows = [(sentence_id, lid) for lid in set(lemma_ids)]
            conn.executemany(
                "INSERT OR IGNORE INTO sentence_lemmas (sentence_id, lemma_id) VALUES (?, ?);",
                rows
            )

    link_sentence_lexemas = link_sentence_lemmas

    def bulk_link_sentence_lemmas(self, links: List[tuple]):
        if not links:
            return
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO sentence_lemmas (sentence_id, lemma_id) VALUES (?, ?);",
                links
            )

    bulk_link_sentence_lexemas = bulk_link_sentence_lemmas

    def link_sentence_words(self, sentence_id: int, word_ids: List[int]):
        if not word_ids:
            return
        with self._get_connection() as conn:
            rows = [(sentence_id, wid) for wid in set(word_ids)]
            conn.executemany(
                "INSERT OR IGNORE INTO sentence_words (sentence_id, word_id) VALUES (?, ?);",
                rows
            )

    def bulk_link_sentence_words(self, links: List[tuple]):
        if not links:
            return
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO sentence_words (sentence_id, word_id) VALUES (?, ?);",
                links
            )

    def get_sentence_lemmas(self, sentence_id: int) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT l.* FROM lemmas l
                JOIN sentence_lemmas sl ON l.id = sl.lemma_id
                WHERE sl.sentence_id = ?
                ORDER BY l.lemma ASC;
            """, (sentence_id,))
            return [dict(r) for r in cursor.fetchall()]

    get_sentence_lexemas = get_sentence_lemmas

    def get_sentence_words(self, sentence_id: int) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT w.* FROM words w
                JOIN sentence_words sw ON w.id = sw.word_id
                WHERE sw.sentence_id = ?
                ORDER BY w.word ASC;
            """, (sentence_id,))
            return [dict(r) for r in cursor.fetchall()]

    # ==========================================
    # Vocabulary Tracking & Appearance Counting
    # ==========================================

    def get_user_known_lemma_ids(
        self,
        user_id: str = 'default_user',
        target_level: Optional[str] = None,
        unlock_threshold: int = VOCABULARY_UNLOCK_THRESHOLD
    ) -> Set[int]:
        """
        Returns the set of lemma IDs known by the user:
        - Base foundational words (is_base = 1)
        - Words with CEFR level strictly below the user's level
        - Words with success_count >= unlock_threshold
        """
        levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        with self._get_connection() as conn:
            if target_level is None:
                row = conn.execute("SELECT level FROM user_modes WHERE user_id = ? AND mode = 'translation';", (user_id,)).fetchone()
                target_level = row['level'] if row and row['level'] else DEFAULT_CEFR_LEVEL

            idx = levels.index(target_level) if target_level in levels else 0
            prior_levels = levels[:idx]

            if prior_levels:
                placeholders = ",".join(["?"] * len(prior_levels))
                cursor = conn.execute(f"""
                    SELECT id FROM lemmas WHERE is_base = 1 OR level IN ({placeholders})
                    UNION
                    SELECT lemma_id FROM user_lemmas
                    WHERE user_id = ? AND success_count >= ?;
                """, list(prior_levels) + [user_id, unlock_threshold])
            else:
                cursor = conn.execute("""
                    SELECT id FROM lemmas WHERE is_base = 1
                    UNION
                    SELECT lemma_id FROM user_lemmas
                    WHERE user_id = ? AND success_count >= ?;
                """, (user_id, unlock_threshold))
            return {r[0] for r in cursor.fetchall()}

    get_user_known_lexema_ids = get_user_known_lemma_ids

    def record_token_appearances(
        self,
        user_id: str,
        sentence_id: int,
        is_success: bool = True,
        unlock_threshold: int = VOCABULARY_UNLOCK_THRESHOLD,
        is_retry: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Increments the appearance count and success count for every word root in the given sentence.
        If is_retry is True, skips incrementing counts and returns current counts/progress.
        Returns the updated list of lexemas with their new counts and derived is_known.
        """
        self.get_or_create_user(user_id)
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT lemma_id FROM sentence_lemmas WHERE sentence_id = ?;",
                (sentence_id,)
            )
            lemma_ids = [r[0] for r in cursor.fetchall()]

            cursor = conn.execute(
                "SELECT word_id FROM sentence_words WHERE sentence_id = ?;",
                (sentence_id,)
            )
            word_ids = [r[0] for r in cursor.fetchall()]

            if not is_retry:
                success_inc = 1 if is_success else 0
                for lid in lemma_ids:
                    conn.execute("""
                        INSERT INTO user_lemmas (user_id, lemma_id, level, appearance_count, success_count)
                        SELECT ?, ?, level, 1, ? FROM lemmas WHERE id = ?
                        ON CONFLICT(user_id, lemma_id) DO UPDATE SET
                            appearance_count = user_lemmas.appearance_count + 1,
                            success_count = user_lemmas.success_count + ?;
                    """, (user_id, lid, success_inc, lid, success_inc))

            for wid in word_ids:
                conn.execute("""
                    INSERT INTO user_words (user_id, word_id, appearance_count, success_count)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(user_id, word_id) DO UPDATE SET
                        appearance_count = user_words.appearance_count + 1,
                        success_count = user_words.success_count + ?;
                """, (user_id, wid, success_inc, success_inc))

            if not lemma_ids:
                return []

            # Determine prior levels for active user level
            u_mode = conn.execute("SELECT level FROM user_modes WHERE user_id = ? AND mode = 'translation';", (user_id,)).fetchone()
            target_level = u_mode['level'] if u_mode and u_mode['level'] else DEFAULT_CEFR_LEVEL
            levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
            idx = levels.index(target_level) if target_level in levels else 0
            prior_levels = set(levels[:idx])

            placeholders = ",".join(["?"] * len(lemma_ids))
            cursor = conn.execute(f"""
                SELECT l.id, l.lemma, l.pos, l.level, l.is_base, ul.appearance_count, ul.success_count
                FROM lemmas l
                JOIN user_lemmas ul ON l.id = ul.lemma_id
                WHERE ul.user_id = ? AND l.id IN ({placeholders});
            """, [user_id] + lemma_ids)

            result = []
            for r in cursor.fetchall():
                d = dict(r)
                is_base = bool(d.pop('is_base', False))
                lvl = d.get('level', 'A1')
                d['is_known'] = 1 if (is_base or (lvl in prior_levels) or (d.get('success_count', 0) >= unlock_threshold)) else 0
                result.append(d)
            return result

    record_lexema_appearances = record_token_appearances

    def unlock_user_vocabulary_for_level(self, user_id: str, target_level: str):
        """
        Synchronizes user vocabulary for target_level.
        Relocks any unpracticed vocabulary (appearance_count = 0 and success_count = 0)
        at or above target_level.
        """
        levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        if target_level not in levels:
            return

        idx = levels.index(target_level)
        current_and_higher = levels[idx:]

        with self._get_connection() as conn:
            if current_and_higher:
                relock_placeholders = ",".join(["?"] * len(current_and_higher))
                conn.execute(f"""
                    DELETE FROM user_lemmas
                    WHERE user_id = ?
                      AND level IN ({relock_placeholders})
                      AND appearance_count = 0
                      AND success_count = 0;
                """, [user_id] + current_and_higher)

    def get_user_vocabulary_stats(self, user_id: str = 'default_user', target_level: Optional[str] = None) -> Dict[str, Any]:
        """Returns vocabulary progress diagnostics for the user."""
        levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        with self._get_connection() as conn:
            if target_level is None:
                row = conn.execute("SELECT level FROM user_modes WHERE user_id = ? AND mode = 'translation';", (user_id,)).fetchone()
                target_level = row['level'] if row and row['level'] else DEFAULT_CEFR_LEVEL

            idx = levels.index(target_level) if target_level in levels else 0
            prior_levels = levels[:idx]

            known_conditions = ["is_base = 1"]
            if prior_levels:
                pl_placeholders = ",".join(["?"] * len(prior_levels))
                known_conditions.append(f"level IN ({pl_placeholders})")

            known_lem_sql = "SELECT id FROM lemmas WHERE " + " OR ".join(known_conditions)
            pl_params = list(prior_levels)

            known_cursor = conn.execute(f"""
                SELECT COUNT(DISTINCT id) FROM (
                    {known_lem_sql}
                    UNION
                    SELECT lemma_id as id FROM user_lemmas WHERE user_id = ? AND success_count >= 1
                );
            """, pl_params + [user_id])
            total_known = known_cursor.fetchone()[0]

            learning_conditions = "ul.user_id = ? AND ul.success_count = 0 AND ul.appearance_count > 0 AND l.is_base = 0"
            learning_params = [user_id]
            if prior_levels:
                learning_conditions += f" AND l.level NOT IN ({pl_placeholders})"
                learning_params.extend(prior_levels)

            learning_cursor = conn.execute(f"""
                SELECT COUNT(DISTINCT ul.lemma_id) FROM user_lemmas ul
                JOIN lemmas l ON ul.lemma_id = l.id
                WHERE {learning_conditions};
            """, learning_params)
            total_learning = learning_cursor.fetchone()[0]

            total_roots_cursor = conn.execute("SELECT COUNT(*) FROM lemmas;")
            total_roots = total_roots_cursor.fetchone()[0]

            lvl_conditions = ["l.is_base = 1"]
            lvl_params = []
            if prior_levels:
                lvl_conditions.append(f"l.level IN ({pl_placeholders})")
                lvl_params.extend(prior_levels)
            lvl_conditions.append("l.id IN (SELECT lemma_id FROM user_lemmas WHERE user_id = ? AND success_count >= 1)")
            lvl_params.append(user_id)

            lvl_cursor = conn.execute(f"""
                SELECT l.level, COUNT(DISTINCT l.id) as cnt
                FROM lemmas l
                WHERE {" OR ".join(lvl_conditions)}
                GROUP BY l.level;
            """, lvl_params)
            by_level = {r['level']: r['cnt'] for r in lvl_cursor.fetchall()}

            known_ids = self.get_user_known_lemma_ids(user_id, target_level=target_level)
            recent_cursor = conn.execute("""
                SELECT l.id, l.lemma, l.level, ul.appearance_count, ul.success_count
                FROM user_lemmas ul
                JOIN lemmas l ON ul.lemma_id = l.id
                WHERE ul.user_id = ?
                ORDER BY ul.lemma_id DESC LIMIT 10;
            """, (user_id,))
            recent_roots = []
            for r in recent_cursor.fetchall():
                d = dict(r)
                d['is_known'] = 1 if d['id'] in known_ids else 0
                recent_roots.append(d)

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
        target_level: str = DEFAULT_CEFR_LEVEL,
        count: int = DEFAULT_CARDS_PER_EXERCISE,
        max_unknown: int = 2,
        exclude_ids: Optional[List[int]] = None,
        mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Progressively selects sentences filtered strictly by CEFR level and number of unknown lemmas (0 to max_unknown).
        Maintains an in-memory buffer to pop sentences quickly for subsequent exercises without redundant SQL queries.
        """
        import random

        if mode == 'transcription':
            max_unknown = 0

        target_level = str(target_level).upper().strip() if target_level else DEFAULT_CEFR_LEVEL
        exclude_set = set(exclude_ids or [])
        buffer_key = f"{user_id}_{target_level}_{mode}"

        if not hasattr(self, '_user_candidates_buffer') or self._user_candidates_buffer is None:
            self._user_candidates_buffer = {}

        if buffer_key not in self._user_candidates_buffer:
            self._user_candidates_buffer[buffer_key] = {'reinforcement': [], 'progression': []}

        buffer = self._user_candidates_buffer[buffer_key]

        # Clean out any excludes from the buffer
        buffer['reinforcement'] = [c for c in buffer['reinforcement'] if c.get('id') not in exclude_set]
        buffer['progression'] = [c for c in buffer['progression'] if c.get('id') not in exclude_set]

        num_prog = max(1, round(count * 0.67)) if mode != 'transcription' else 0
        num_reinf = count - num_prog if mode != 'transcription' else count

        # If the buffer is running low, perform the SQL query to refill it
        if len(buffer['reinforcement']) < num_reinf or len(buffer['progression']) < num_prog:
            levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
            idx = levels.index(target_level) if target_level in levels else 0
            prior_levels = levels[:idx]

            known_conditions = ["is_base = 1"]
            params = []
            if prior_levels:
                pl_placeholders = ",".join(["?"] * len(prior_levels))
                known_conditions.append(f"level IN ({pl_placeholders})")
                params.extend(prior_levels)

            known_sql = " OR ".join(known_conditions)

            query = f"""
                WITH known_lemmas AS (
                    SELECT id FROM lemmas WHERE {known_sql}
                    UNION
                    SELECT lemma_id FROM user_lemmas WHERE user_id = ? AND success_count >= ?
                )
                SELECT 
                    s.*, 
                    GROUP_CONCAT(sl.lemma_id) as all_lemma_ids,
                    SUM(CASE WHEN kl.id IS NULL THEN 1 ELSE 0 END) as unknown_count,
                    GROUP_CONCAT(CASE WHEN kl.id IS NULL THEN sl.lemma_id ELSE NULL END) as unknown_lemma_ids
                FROM sentences s
                JOIN sentence_lemmas sl ON s.id = sl.sentence_id
                LEFT JOIN known_lemmas kl ON sl.lemma_id = kl.id
                WHERE s.level = ?
                GROUP BY s.id
                HAVING unknown_count <= ?
            """

            params.extend([user_id, VOCABULARY_UNLOCK_THRESHOLD, target_level, max_unknown])

            with self._get_connection() as conn:
                cursor = conn.execute(query, params)
                all_rows = cursor.fetchall()

            new_reinf = []
            new_prog = []
            excluded_reinf = []

            for row in all_rows:
                s_dict = self._row_to_dict(row)
                unknown_count = row['unknown_count']
                s_dict['unknown_count'] = unknown_count

                raw_all = row['all_lemma_ids']
                s_dict['_sent_lids'] = {int(x) for x in raw_all.split(',')} if raw_all else set()

                raw_unk = row['unknown_lemma_ids']
                s_dict['_unknown_lids'] = {int(x) for x in raw_unk.split(',')} if raw_unk else set()

                if s_dict['id'] in exclude_set:
                    if unknown_count == 0:
                        excluded_reinf.append(s_dict)
                    continue

                if unknown_count == 0:
                    new_reinf.append(s_dict)
                else:
                    new_prog.append(s_dict)

            # Shuffle and refill buffer
            random.shuffle(new_reinf)
            random.shuffle(new_prog)
            random.shuffle(excluded_reinf)
            buffer['reinforcement'] = new_reinf
            buffer['progression'] = new_prog
            buffer['excluded_reinforcement'] = excluded_reinf

        selected = []

        if mode == 'transcription':
            actual_reinf = min(count, len(buffer['reinforcement']))
            for _ in range(actual_reinf):
                selected.append(buffer['reinforcement'].pop())

            # In transcription mode, if still needed, recycle familiar sentences from excluded
            if len(selected) < count and buffer.get('excluded_reinforcement'):
                needed = count - len(selected)
                recycled = buffer['excluded_reinforcement'][:needed]
                selected.extend(recycled)
        else:
            actual_prog = min(num_prog, len(buffer['progression']))
            for _ in range(actual_prog):
                selected.append(buffer['progression'].pop())

            needed_reinf = count - len(selected)
            actual_reinf = min(needed_reinf, len(buffer['reinforcement']))
            for _ in range(actual_reinf):
                selected.append(buffer['reinforcement'].pop())

            # If still needed and progression has remaining cards, use them
            if len(selected) < count and buffer['progression']:
                extra_prog = min(count - len(selected), len(buffer['progression']))
                for _ in range(extra_prog):
                    selected.append(buffer['progression'].pop())

        if not selected:
            return []

        # Load lem_dict and user_counts for token derivation
        lem_dict = self.get_lemmas_dict()
        with self._get_connection() as conn:
            user_counts_cursor = conn.execute(
                "SELECT lemma_id, appearance_count, success_count FROM user_lemmas WHERE user_id = ?;",
                (user_id,)
            )
            user_counts = {
                r['lemma_id']: {
                    'count': r['appearance_count'],
                    'success_count': r['success_count'],
                }
                for r in user_counts_cursor.fetchall()
            }

        # Resolve accurate token derivations only for the final selected sentences
        try:
            from backend.nlp.vocab_engine import get_vocab_engine
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

            known_lemmas_list = [lem_dict[lid]['lemma'] for lid in known_lids if lid in lem_dict]
            new_lemmas_list = [lem_dict[lid]['lemma'] for lid in unknown_lids if lid in lem_dict]
            s['known_lemmas'] = known_lemmas_list
            s['known_lexemas'] = known_lemmas_list
            s['new_lemmas'] = new_lemmas_list
            s['new_lexemas'] = new_lemmas_list

            first_time_lids = [
                lid for lid in sent_lids
                if not lem_dict.get(lid, {}).get('is_base', False)
                and (lid not in user_counts or user_counts[lid]['count'] <= VOCABULARY_UNLOCK_THRESHOLD)
            ]
            first_time_lemmas_list = [lem_dict[lid]['lemma'] for lid in first_time_lids if lid in lem_dict]
            if not first_time_lemmas_list and s['new_lemmas']:
                first_time_lemmas_list = s['new_lemmas']
            s['first_time_lemmas'] = first_time_lemmas_list
            s['first_time_lexemas'] = first_time_lemmas_list

            s.pop('_score_diff', None)

            # Match surface words in target French sentence for each first-time root
            pairs = []
            if s.get('first_time_lemmas'):
                if engine is not None:
                    try:
                        pairs = engine.extract_sentence_token_pairs(s.get('fr', ''), s['first_time_lemmas'])
                    except Exception:
                        pairs = []
                if not pairs:
                    fr_tokens = re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", s.get('fr', ''), re.UNICODE)
                    elision_prefix = r"^(se\s+|qu['’]|jusqu['’]|lorsqu['’]|puisqu['’]|[ldjcmnstLDJCMNST]['’])"
                    for lem in s['first_time_lemmas']:
                        lem_clean = re.sub(elision_prefix, "", lem.lower(), flags=re.IGNORECASE).strip()
                        matched_word = lem_clean
                        for tok in fr_tokens:
                            tok_clean = re.sub(elision_prefix, "", tok.lower(), flags=re.IGNORECASE).strip()
                            if not tok_clean:
                                continue
                            if tok_clean == lem_clean:
                                matched_word = re.sub(elision_prefix, "", tok, flags=re.IGNORECASE)
                                break
                            min_len = min(len(tok_clean), len(lem_clean))
                            if min_len >= 4 and tok_clean[:4] == lem_clean[:4]:
                                matched_word = re.sub(elision_prefix, "", tok, flags=re.IGNORECASE)
                            elif min_len >= 3 and tok_clean[:3] == lem_clean[:3] and abs(len(tok_clean) - len(lem_clean)) <= 2:
                                matched_word = re.sub(elision_prefix, "", tok, flags=re.IGNORECASE)
                        pairs.append({'word': matched_word, 'lemma': lem})
            s['first_time_pairs'] = pairs

        return selected

