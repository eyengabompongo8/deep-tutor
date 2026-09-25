import pytest
from backend.db import DataManager
from backend.scoring.engine import ScoringEngine


class TestProgressiveVocabulary:
    @pytest.fixture(autouse=True)
    def setup_vocab(self, temp_db_path):
        self.dm = DataManager(db_path=temp_db_path)
        self.scoring = ScoringEngine(data_manager=self.dm)

        # Seed test lexemas
        self.l1 = self.dm.upsert_lexema(lemma="être", pos="VERB", level="A1", is_base=True)
        self.l2 = self.dm.upsert_lexema(lemma="avoir", pos="VERB", level="A1", is_base=True)
        self.l3 = self.dm.upsert_lexema(lemma="chat", pos="NOUN", level="A1", is_base=False)
        self.l4 = self.dm.upsert_lexema(lemma="noir", pos="ADJ", level="A1", is_base=False)
        self.l5 = self.dm.upsert_lexema(lemma="dormir", pos="VERB", level="A1", is_base=False)
        self.l6 = self.dm.upsert_lexema(lemma="astrophysique", pos="NOUN", level="C1", is_base=False)

        # Add sentences
        # s1: all base words ("être" + "avoir") -> 0 unknown words
        self.s1 = self.dm.add_sentence(es="Tengo y soy", fr="J'ai et je suis", level="A1", score=1.0)
        self.dm.link_sentence_lexemas(self.s1["id"], [self.l1, self.l2])

        # s2: base words + 1 unknown word ("chat") -> 1 unknown word
        self.s2 = self.dm.add_sentence(es="El gato", fr="Le chat", level="A1", score=1.2)
        self.dm.link_sentence_lexemas(self.s2["id"], [self.l1, self.l3])

        # s3: base words + 2 unknown words ("chat", "noir") -> 2 unknown words
        self.s3 = self.dm.add_sentence(es="El gato negro", fr="Le chat noir", level="A1", score=1.3)
        self.dm.link_sentence_lexemas(self.s3["id"], [self.l1, self.l3, self.l4])

        # s4: base words + 3 unknown words ("chat", "noir", "dormir") -> 3 unknown words
        self.s4 = self.dm.add_sentence(es="El gato negro duerme", fr="Le chat noir dort", level="A1", score=1.5)
        self.dm.link_sentence_lexemas(self.s4["id"], [self.l3, self.l4, self.l5])

        # s5: advanced C1 sentence with astrophysique
        self.s5 = self.dm.add_sentence(es="La astrofísica", fr="L'astrophysique", level="C1", score=5.2)
        self.dm.link_sentence_lexemas(self.s5["id"], [self.l6])

    def test_user_initialization(self):
        user = self.dm.get_or_create_user("test_user", "Test Student")
        assert user["id"] == "test_user"
        assert user["name"] == "Test Student"
        profile = self.dm.get_user_modes_profile("test_user")
        assert profile["level"] == "A1"

    def test_base_vocabulary_unlocked_initially(self):
        known_ids = self.dm.get_user_known_lexema_ids("test_user")
        assert self.l1 in known_ids
        assert self.l2 in known_ids
        assert self.l3 not in known_ids
        assert self.l4 not in known_ids

    def test_progressive_sentence_selection(self):
        batch = self.dm.get_progressive_sentences(user_id="test_user", target_level="A1", count=3, max_unknown=2)
        assert len(batch) > 0
        for s in batch:
            assert "unknown_count" in s
            assert "new_lexemas" in s
            assert "known_lexemas" in s
            assert s["unknown_count"] <= 2

    def test_appearance_counting_and_unlock(self):
        progress1 = self.dm.record_lexema_appearances(user_id="test_user", sentence_id=self.s2["id"], is_success=True)
        assert any(p["id"] == self.l3 and p["is_known"] == 1 for p in progress1)

        with self.dm._get_connection() as conn:
            row = conn.execute(
                "SELECT appearance_count, success_count FROM user_lexema_counts WHERE user_id = 'test_user' AND lexema_id = ?;",
                (self.l3,)
            ).fetchone()
            assert row is not None
            assert row["appearance_count"] == 1
            assert row["success_count"] == 1

        known_ids_after = self.dm.get_user_known_lexema_ids("test_user")
        assert self.l3 in known_ids_after

    def test_retry_lexema_appearances_consolidation(self):
        # First attempt: failure
        self.dm.record_lexema_appearances(user_id="test_user", sentence_id=self.s2["id"], is_success=False, is_retry=False)
        with self.dm._get_connection() as conn:
            row = conn.execute(
                "SELECT appearance_count, success_count FROM user_lexema_counts WHERE user_id = 'test_user' AND lexema_id = ?;",
                (self.l3,)
            ).fetchone()
            assert row is not None
            assert row["appearance_count"] == 1
            assert row["success_count"] == 0

        # Retry attempt on same card: success, is_retry=True
        progress = self.dm.record_lexema_appearances(user_id="test_user", sentence_id=self.s2["id"], is_success=True, is_retry=True)
        assert len(progress) > 0
        with self.dm._get_connection() as conn:
            row = conn.execute(
                "SELECT appearance_count, success_count FROM user_lexema_counts WHERE user_id = 'test_user' AND lexema_id = ?;",
                (self.l3,)
            ).fetchone()
            assert row is not None
            # Metrics remain consolidated to first try!
            assert row["appearance_count"] == 1
            assert row["success_count"] == 0

        # Different session / subsequent encounter: is_retry=False, success=True
        self.dm.record_lexema_appearances(user_id="test_user", sentence_id=self.s2["id"], is_success=True, is_retry=False)
        with self.dm._get_connection() as conn:
            row = conn.execute(
                "SELECT appearance_count, success_count FROM user_lexema_counts WHERE user_id = 'test_user' AND lexema_id = ?;",
                (self.l3,)
            ).fetchone()
            assert row is not None
            # New appearance counted
            assert row["appearance_count"] == 2
            assert row["success_count"] == 1

    def test_vocabulary_stats(self):
        self.dm.record_lexema_appearances(user_id="test_user", sentence_id=self.s2["id"], is_success=True)
        stats = self.dm.get_user_vocabulary_stats("test_user")
        assert "total_known" in stats
        assert "total_learning" in stats
        assert "by_level" in stats
        assert stats["total_known"] >= 3

    def test_scoring_engine_integration(self):
        batch = self.scoring.get_batch_sentences(
            all_sentences=self.dm.get_all_sentences(),
            target_level="A1",
            count=3,
            data_manager=self.dm,
            user_id="test_user",
            max_unknown=2
        )
        assert len(batch) == 3
        for s in batch:
            assert "fr" in s
            assert "es" in s

    def test_transcription_mode_zero_new_words(self):
        batch = self.dm.get_progressive_sentences(
            user_id="test_user",
            target_level="A1",
            count=1,
            mode="transcription"
        )
        assert len(batch) == 1
        assert batch[0]["id"] == self.s1["id"]
        assert batch[0]["unknown_count"] == 0
        assert batch[0]["new_lexemas"] == []

        batch_se = self.scoring.get_batch_sentences(
            all_sentences=self.dm.get_all_sentences(),
            target_level="A1",
            count=1,
            data_manager=self.dm,
            user_id="test_user",
            mode="transcription"
        )
        assert len(batch_se) == 1
        assert batch_se[0]["id"] == self.s1["id"]
        assert batch_se[0]["unknown_count"] == 0
        assert batch_se[0]["new_lexemas"] == []

    def test_transcription_mode_never_falls_back_to_new_words(self):
        batch = self.dm.get_progressive_sentences(
            user_id="test_user",
            target_level="A1",
            count=1,
            exclude_ids=[self.s1["id"]],
            mode="transcription"
        )
        assert len(batch) == 1
        assert batch[0]["id"] == self.s1["id"]
        assert batch[0]["unknown_count"] == 0
        assert batch[0]["new_lexemas"] == []

    def test_irregular_word_derivation_matching(self):
        l_aller = self.dm.upsert_lexema(lemma="aller", pos="VERB", level="A1", is_base=False)
        s_aller = self.dm.add_sentence(es="Voy a la playa", fr="Je vais à la plage", level="A1", score=1.2)
        self.dm.link_sentence_lexemas(s_aller["id"], [l_aller])

        batch = self.dm.get_progressive_sentences(user_id="test_user", target_level="A1", count=5)
        matched_s = next((s for s in batch if s["id"] == s_aller["id"]), None)
        assert matched_s is not None
        assert "first_time_pairs" in matched_s
        aller_pair = next((p for p in matched_s["first_time_pairs"] if p["lemma"] == "aller"), None)
        assert aller_pair is not None
        assert aller_pair["word"] == "vais"

    def test_regular_word_derivation_matching(self):
        batch = self.dm.get_progressive_sentences(user_id="test_user", target_level="A1", count=5)
        matched_s = next((s for s in batch if s["id"] == self.s2["id"]), None)
        assert matched_s is not None
        assert "first_time_pairs" in matched_s
        chat_pair = next((p for p in matched_s["first_time_pairs"] if p["lemma"] == "chat"), None)
        assert chat_pair is not None
        assert chat_pair["word"] == "chat"

    def test_contraction_and_elision_derivation_matching(self):
        l_obstiner = self.dm.upsert_lexema(lemma="obstiner", pos="VERB", level="B2", is_base=False)
        s_obst = self.dm.add_sentence(es="Ella es obstinada", fr="Elle est l'obstinée de la classe", level="B2", score=4.5)
        self.dm.link_sentence_lexemas(s_obst["id"], [l_obstiner])

        batch = self.dm.get_progressive_sentences(user_id="test_user", target_level="B2", count=5)
        matched_s = next((s for s in batch if s["id"] == s_obst["id"]), None)
        assert matched_s is not None
        assert "first_time_pairs" in matched_s
        pair = next((p for p in matched_s["first_time_pairs"] if p["lemma"] == "obstiner"), None)
        assert pair is not None
        assert pair["word"] == "obstinée"

    def test_reflexive_verb_derivation_matching(self):
        l_refl = self.dm.upsert_lexema(lemma="s'obstiner", pos="VERB", level="B2", is_base=False)
        s_refl = self.dm.add_sentence(es="Ella es obstinada", fr="Elle s'obstine toujours", level="B2", score=4.5)
        self.dm.link_sentence_lexemas(s_refl["id"], [l_refl])

        batch = self.dm.get_progressive_sentences(user_id="test_user", target_level="B2", count=5)
        matched_s = next((s for s in batch if s["id"] == s_refl["id"]), None)
        assert matched_s is not None
        assert "first_time_pairs" in matched_s
        pair = next((p for p in matched_s["first_time_pairs"] if "obstin" in p["lemma"]), None)
        assert pair is not None
        assert pair["word"] == "obstine"

    def test_manual_level_lowering_relocks_unpracticed_vocab(self):
        # 1. Reset user to B2 -> unlocks A1, A2, B1 lexemas
        self.scoring.reset_profile(level="B2", mode="translation", user_id="relock_user")
        known_b2 = self.dm.get_user_known_lexema_ids("relock_user")
        assert self.l3 in known_b2
        
        # 2. Simulate practicing a specific A1 word (l3: 'chat')
        self.dm.record_lexema_appearances("relock_user", self.s2["id"], is_success=True)

        # 3. Lower level manually to A1
        self.scoring.reset_profile(level="A1", mode="translation", user_id="relock_user")
        known_a1 = self.dm.get_user_known_lexema_ids("relock_user")

        # Base words are still known
        assert self.l1 in known_a1
        assert self.l2 in known_a1

        # Practiced word 'chat' (l3) remains known because it was actually practiced
        assert self.l3 in known_a1

        # Unpracticed non-base word 'noir' (l4) is relocked and no longer known
        assert self.l4 not in known_a1
