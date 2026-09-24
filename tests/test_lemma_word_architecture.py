import sqlite3
import pytest
from backend.db import DataManager
from backend.nlp.vocab_engine import VocabEngine


class TestLemmaWordArchitecture:
    @pytest.fixture
    def dm(self, temp_db_path):
        return DataManager(db_path=temp_db_path)

    def test_database_integrity_and_cascade(self, dm):
        # 1. Unique constraint on lemmas (lemma, pos)
        l1_id = dm.upsert_lemma(lemma="danser", pos="VER", level="A1", is_base=True)
        assert l1_id > 0

        # Same lemma with different pos is allowed
        l2_id = dm.upsert_lemma(lemma="danser", pos="NOM", level="B1", is_base=False)
        assert l2_id > 0
        assert l1_id != l2_id

        # Direct duplicate insert raises IntegrityError
        with dm._get_connection() as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO lemmas (lemma, pos, level) VALUES (?, ?, ?);",
                    ("danser", "VER", "A2")
                )

        # 2. Surface words linked to lemma
        w1_id = dm.upsert_word(word="danse", lemma_id=l1_id, pos="VER", frequency=18.4)
        w2_id = dm.upsert_word(word="danse", lemma_id=l2_id, pos="NOM", frequency=41.0)
        assert w1_id > 0 and w2_id > 0
        assert w1_id != w2_id

        # Unique constraint on words (word, lemma_id, pos)
        with dm._get_connection() as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO words (word, lemma_id, pos) VALUES (?, ?, ?);",
                    ("danse", l1_id, "VER")
                )

        # 3. Foreign key cascade: deleting a lemma cascades to its words
        with dm._get_connection() as conn:
            conn.execute("DELETE FROM lemmas WHERE id = ?;", (l1_id,))
            word_count = conn.execute("SELECT COUNT(*) FROM words WHERE id = ?;", (w1_id,)).fetchone()[0]
            assert word_count == 0
            # w2 (attached to l2_id) remains untouched
            w2_count = conn.execute("SELECT COUNT(*) FROM words WHERE id = ?;", (w2_id,)).fetchone()[0]
            assert w2_count == 1

    def test_sentence_junctions_and_token_appearances(self, dm):
        # Seed lemmas & words
        l_chat = dm.upsert_lemma(lemma="chat", pos="NOM", level="A1", is_base=False)
        l_noir = dm.upsert_lemma(lemma="noir", pos="ADJ", level="A1", is_base=False)

        w_chat = dm.upsert_word(word="chat", lemma_id=l_chat, pos="NOM")
        w_noir = dm.upsert_word(word="noir", lemma_id=l_noir, pos="ADJ")

        # Create sentence
        sent = dm.add_sentence(fr="Le chat noir.", es="El gato negro.", level="A1", score=1.2)
        s_id = sent["id"]

        # Link sentence to lemmas & words
        dm.link_sentence_lemmas(s_id, [l_chat, l_noir])
        dm.link_sentence_words(s_id, [w_chat, w_noir])

        # Verify junctions
        s_lemmas = dm.get_sentence_lemmas(s_id)
        assert len(s_lemmas) == 2
        assert {r["id"] for r in s_lemmas} == {l_chat, l_noir}

        s_words = dm.get_sentence_words(s_id)
        assert len(s_words) == 2
        assert {r["id"] for r in s_words} == {w_chat, w_noir}

        # 1. Successful card completion -> updates both user_lemmas and user_words
        progress = dm.record_token_appearances(user_id="alice", sentence_id=s_id, is_success=True)
        assert len(progress) == 2

        with dm._get_connection() as conn:
            # Check user_lemmas
            ul_rows = conn.execute("SELECT lemma_id, appearance_count, success_count FROM user_lemmas WHERE user_id = 'alice';").fetchall()
            assert len(ul_rows) == 2
            for row in ul_rows:
                assert row["appearance_count"] == 1
                assert row["success_count"] == 1

            # Check user_words
            uw_rows = conn.execute("SELECT word_id, appearance_count, success_count FROM user_words WHERE user_id = 'alice';").fetchall()
            assert len(uw_rows) == 2
            for row in uw_rows:
                assert row["appearance_count"] == 1
                assert row["success_count"] == 1

        # 2. Skipped card -> increments appearances, keeps success_count unchanged
        dm.record_token_appearances(user_id="alice", sentence_id=s_id, is_success=False)

        with dm._get_connection() as conn:
            # Check user_lemmas: appearance_count == 2, success_count == 1
            ul_rows = conn.execute("SELECT lemma_id, appearance_count, success_count FROM user_lemmas WHERE user_id = 'alice';").fetchall()
            for row in ul_rows:
                assert row["appearance_count"] == 2
                assert row["success_count"] == 1

            # Check user_words: appearance_count == 2, success_count == 1
            uw_rows = conn.execute("SELECT word_id, appearance_count, success_count FROM user_words WHERE user_id = 'alice';").fetchall()
            for row in uw_rows:
                assert row["appearance_count"] == 2
                assert row["success_count"] == 1

    def test_progressive_sentences_isolated_by_lemmas(self, dm):
        # Ensure candidate sentence selection continues to strictly filter by lemmas
        l1 = dm.upsert_lemma(lemma="être", pos="VER", level="A1", is_base=True)
        l2 = dm.upsert_lemma(lemma="ami", pos="NOM", level="A1", is_base=False)

        s1 = dm.add_sentence(fr="Je suis un ami.", es="Soy un amigo.", level="A1", score=1.1)
        dm.link_sentence_lemmas(s1["id"], [l1, l2])

        candidates = dm.get_progressive_sentences(user_id="bob", target_score=1.1, count=1, max_unknown=2)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand["id"] == s1["id"]
        assert "unknown_count" in cand

    def test_vocab_engine_token_resolution(self, dm):
        engine = VocabEngine()
        l_id = dm.upsert_lemma(lemma="danser", pos="VER", level="A1")
        w_id = dm.upsert_word(word="danse", lemma_id=l_id, pos="VER")

        doc = engine.nlp("Elle danse souvent.")
        token_danse = [t for t in doc if t.text == "danse"][0]

        lemma_pos_to_id = {("danse", "VER"): l_id, ("danser", "VER"): l_id}
        lemma_fallback = {"danse": l_id, "danser": l_id}

        resolved_lid = engine.resolve_token_lemma_id(token_danse, lemma_pos_to_id, lemma_fallback)
        assert resolved_lid == l_id

        word_pos_to_id = {("danse", "VER"): w_id}
        word_fallback = {"danse": w_id}
        resolved_wid = engine.resolve_token_word_id(token_danse, word_pos_to_id, word_fallback)
        assert resolved_wid == w_id
