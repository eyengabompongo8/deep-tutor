import json
import os
import re
from typing import List, Dict, Any, Optional, Set

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
            from scripts.vocab_engine import get_vocab_engine
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
