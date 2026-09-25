import json
import os
import re
from typing import List, Dict, Any, Optional, Set
from backend.config import (
    VOCABULARY_UNLOCK_THRESHOLD,
    DEFAULT_CEFR_LEVEL,
    STARTING_SCORE,
    DEFAULT_CARDS_PER_EXERCISE,
    MAX_UNKNOWN_WORDS,
)

class VocabularyMixin:
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
        """Invalidates the in-memory lexemas cache and candidate sentence buffer."""
        with self._lexemas_cache_lock:
            self._lexemas_cache = None
            self._user_candidates_buffer = {}

    def upsert_lexema(
        self,
        lemma: str,
        pos: Optional[str] = None,
        level: str = DEFAULT_CEFR_LEVEL,
        frequency: float = 0.0,
        is_base: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        lemma = lemma.lower().strip()
        pos = (pos or '').upper().strip() or None
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)
        self.invalidate_lexemas_cache()
        with self._get_connection() as conn:
            conn.execute(f"""
                INSERT INTO lexemas (lemma, pos, level, frequency, is_base, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(lemma, pos) DO UPDATE SET
                    level = CASE WHEN excluded.level != '{DEFAULT_CEFR_LEVEL}' THEN excluded.level ELSE lexemas.level END,
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
                lvl = item.get('level', DEFAULT_CEFR_LEVEL)
                freq = float(item.get('frequency', item.get('freq', 0.0)))
                is_base = 1 if item.get('is_base') else 0
                meta = json.dumps(item.get('metadata', {}), ensure_ascii=False)
                conn.execute(f"""
                    INSERT INTO lexemas (lemma, pos, level, frequency, is_base, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(lemma, pos) DO UPDATE SET
                        level = CASE WHEN excluded.level != '{DEFAULT_CEFR_LEVEL}' THEN excluded.level ELSE lexemas.level END,
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

    def get_user_known_lexema_ids(
        self,
        user_id: str = 'default_user',
        target_level: Optional[str] = None,
        unlock_threshold: int = VOCABULARY_UNLOCK_THRESHOLD
    ) -> Set[int]:
        """
        Returns the set of lexema IDs known by the user:
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
                    SELECT id FROM lexemas WHERE is_base = 1 OR level IN ({placeholders})
                    UNION
                    SELECT lexema_id FROM user_lexema_counts
                    WHERE user_id = ? AND success_count >= ?;
                """, list(prior_levels) + [user_id, unlock_threshold])
            else:
                cursor = conn.execute("""
                    SELECT id FROM lexemas WHERE is_base = 1
                    UNION
                    SELECT lexema_id FROM user_lexema_counts
                    WHERE user_id = ? AND success_count >= ?;
                """, (user_id, unlock_threshold))
            return {r[0] for r in cursor.fetchall()}

    def record_lexema_appearances(
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
                "SELECT lexema_id FROM sentence_lexemas WHERE sentence_id = ?;",
                (sentence_id,)
            )
            lexema_ids = [r[0] for r in cursor.fetchall()]
            if not lexema_ids:
                return []

            if not is_retry:
                success_inc = 1 if is_success else 0
                for lid in lexema_ids:
                    conn.execute("""
                        INSERT INTO user_lexema_counts (user_id, lexema_id, level, appearance_count, success_count)
                        SELECT ?, ?, level, 1, ? FROM lexemas WHERE id = ?
                        ON CONFLICT(user_id, lexema_id) DO UPDATE SET
                            appearance_count = user_lexema_counts.appearance_count + 1,
                            success_count = user_lexema_counts.success_count + ?;
                    """, (user_id, lid, success_inc, lid, success_inc))

            # Determine prior levels for active user level
            u_mode = conn.execute("SELECT level FROM user_modes WHERE user_id = ? AND mode = 'translation';", (user_id,)).fetchone()
            target_level = u_mode['level'] if u_mode and u_mode['level'] else DEFAULT_CEFR_LEVEL
            levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
            idx = levels.index(target_level) if target_level in levels else 0
            prior_levels = set(levels[:idx])

            placeholders = ",".join(["?"] * len(lexema_ids))
            cursor = conn.execute(f"""
                SELECT l.id, l.lemma, l.pos, l.level, l.is_base, ulc.appearance_count, ulc.success_count
                FROM lexemas l
                JOIN user_lexema_counts ulc ON l.id = ulc.lexema_id
                WHERE ulc.user_id = ? AND l.id IN ({placeholders});
            """, [user_id] + lexema_ids)

            result = []
            for r in cursor.fetchall():
                d = dict(r)
                is_base = bool(d.pop('is_base', False))
                lvl = d.get('level', 'A1')
                d['is_known'] = 1 if (is_base or (lvl in prior_levels) or (d.get('success_count', 0) >= unlock_threshold)) else 0
                result.append(d)
            return result

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
                    DELETE FROM user_lexema_counts
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

            known_lex_sql = "SELECT id FROM lexemas WHERE " + " OR ".join(known_conditions)
            pl_params = list(prior_levels)

            known_cursor = conn.execute(f"""
                SELECT COUNT(DISTINCT id) FROM (
                    {known_lex_sql}
                    UNION
                    SELECT lexema_id as id FROM user_lexema_counts WHERE user_id = ? AND success_count >= 1
                );
            """, pl_params + [user_id])
            total_known = known_cursor.fetchone()[0]

            learning_conditions = "ulc.user_id = ? AND ulc.success_count = 0 AND ulc.appearance_count > 0 AND l.is_base = 0"
            learning_params = [user_id]
            if prior_levels:
                learning_conditions += f" AND l.level NOT IN ({pl_placeholders})"
                learning_params.extend(prior_levels)

            learning_cursor = conn.execute(f"""
                SELECT COUNT(DISTINCT ulc.lexema_id) FROM user_lexema_counts ulc
                JOIN lexemas l ON ulc.lexema_id = l.id
                WHERE {learning_conditions};
            """, learning_params)
            total_learning = learning_cursor.fetchone()[0]

            total_roots_cursor = conn.execute("SELECT COUNT(*) FROM lexemas;")
            total_roots = total_roots_cursor.fetchone()[0]

            lvl_conditions = ["l.is_base = 1"]
            lvl_params = []
            if prior_levels:
                lvl_conditions.append(f"l.level IN ({pl_placeholders})")
                lvl_params.extend(prior_levels)
            lvl_conditions.append("l.id IN (SELECT lexema_id FROM user_lexema_counts WHERE user_id = ? AND success_count >= 1)")
            lvl_params.append(user_id)

            lvl_cursor = conn.execute(f"""
                SELECT l.level, COUNT(DISTINCT l.id) as cnt
                FROM lexemas l
                WHERE {" OR ".join(lvl_conditions)}
                GROUP BY l.level;
            """, lvl_params)
            by_level = {r['level']: r['cnt'] for r in lvl_cursor.fetchall()}

            known_ids = self.get_user_known_lexema_ids(user_id, target_level=target_level)
            recent_cursor = conn.execute("""
                SELECT l.id, l.lemma, l.level, ulc.appearance_count, ulc.success_count
                FROM user_lexema_counts ulc
                JOIN lexemas l ON ulc.lexema_id = l.id
                WHERE ulc.user_id = ?
                ORDER BY ulc.lexema_id DESC LIMIT 10;
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
        target_score: float = STARTING_SCORE,
        count: int = DEFAULT_CARDS_PER_EXERCISE,
        max_unknown: int = 2,
        exclude_ids: Optional[List[int]] = None,
        mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Progressively selects sentences filtered strictly by CEFR level and number of unknown lexemas (0 to max_unknown).
        Maintains an in-memory buffer to pop sentences quickly for subsequent exercises without redundant SQL queries.
        """
        import random
        from backend.scoring.cefr import score_to_level

        if mode == 'transcription':
            max_unknown = 0

        target_level = score_to_level(target_score)
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
                WITH known_lexemas AS (
                    SELECT id FROM lexemas WHERE {known_sql}
                    UNION
                    SELECT lexema_id FROM user_lexema_counts WHERE user_id = ? AND success_count >= ?
                )
                SELECT 
                    s.*, 
                    GROUP_CONCAT(sl.lexema_id) as all_lexema_ids,
                    SUM(CASE WHEN kl.id IS NULL THEN 1 ELSE 0 END) as unknown_count,
                    GROUP_CONCAT(CASE WHEN kl.id IS NULL THEN sl.lexema_id ELSE NULL END) as unknown_lexema_ids
                FROM sentences s
                JOIN sentence_lexemas sl ON s.id = sl.sentence_id
                LEFT JOIN known_lexemas kl ON sl.lexema_id = kl.id
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

                raw_all = row['all_lexema_ids']
                s_dict['_sent_lids'] = {int(x) for x in raw_all.split(',')} if raw_all else set()

                raw_unk = row['unknown_lexema_ids']
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

        # Load lex_dict and user_counts for token derivation
        lex_dict = self.get_lexemas_dict()
        with self._get_connection() as conn:
            user_counts_cursor = conn.execute(
                "SELECT lexema_id, appearance_count, success_count FROM user_lexema_counts WHERE user_id = ?;",
                (user_id,)
            )
            user_counts = {
                r['lexema_id']: {
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

            s['known_lexemas'] = [lex_dict[lid]['lemma'] for lid in known_lids if lid in lex_dict]
            s['new_lexemas'] = [lex_dict[lid]['lemma'] for lid in unknown_lids if lid in lex_dict]

            first_time_lids = [
                lid for lid in sent_lids
                if not lex_dict.get(lid, {}).get('is_base', False)
                and (lid not in user_counts or user_counts[lid]['count'] <= VOCABULARY_UNLOCK_THRESHOLD)
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
                    fr_tokens = re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", s.get('fr', ''), re.UNICODE)
                    elision_prefix = r"^(se\s+|qu['’]|jusqu['’]|lorsqu['’]|puisqu['’]|[ldjcmnstLDJCMNST]['’])"
                    for lem in s['first_time_lexemas']:
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

