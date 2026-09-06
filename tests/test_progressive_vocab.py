import os
import unittest
import tempfile
import sqlite3
from data_manager import DataManager
from scoring_engine import ScoringEngine


class TestProgressiveVocabulary(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_sentences.db")
        self.dm = DataManager(db_path=self.db_path)
        self.scoring = ScoringEngine()

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

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_user_initialization(self):
        user = self.dm.get_or_create_user("test_user", "Test Student")
        self.assertEqual(user["id"], "test_user")
        self.assertEqual(user["name"], "Test Student")
        self.assertEqual(user["score"], 1.20)
        self.assertEqual(user["level"], "A1")

    def test_base_vocabulary_unlocked_initially(self):
        known_ids = self.dm.get_user_known_lexema_ids("test_user")
        self.assertIn(self.l1, known_ids)
        self.assertIn(self.l2, known_ids)
        self.assertNotIn(self.l3, known_ids)
        self.assertNotIn(self.l4, known_ids)

    def test_progressive_sentence_selection(self):
        # A beginner with only base words should get sentences with <= 2 unknown words (s1, s2, s3)
        # s4 has 3 unknown words, s5 has C1 score
        batch = self.dm.get_progressive_sentences(user_id="test_user", target_score=1.2, count=3, max_unknown=2)
        batch_ids = [s["id"] for s in batch]

        self.assertTrue(len(batch) > 0)
        # s1 (0 unknown), s2 (1 unknown), s3 (2 unknown) are the primary candidates
        for s in batch:
            self.assertIn("unknown_count", s)
            self.assertIn("new_lexemas", s)
            self.assertIn("known_lexemas", s)
            self.assertLessEqual(s["unknown_count"], 2)

    def test_appearance_counting_and_unlock(self):
        # Test appearance increment
        progress1 = self.dm.record_lexema_appearances(user_id="test_user", sentence_id=self.s2["id"], is_success=True)
        self.assertTrue(any(p["id"] == self.l3 for p in progress1))

        # Check in user_lexema_counts table
        with self.dm._get_connection() as conn:
            row = conn.execute(
                "SELECT appearance_count, success_count, is_known FROM user_lexema_counts WHERE user_id = 'test_user' AND lexema_id = ?;",
                (self.l3,)
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["appearance_count"], 1)
            self.assertEqual(row["success_count"], 1)
            self.assertEqual(row["is_known"], 1)  # Unlocked because success >= 1

        # Now "chat" (l3) is in known lexemas
        known_ids_after = self.dm.get_user_known_lexema_ids("test_user")
        self.assertIn(self.l3, known_ids_after)

    def test_vocabulary_stats(self):
        self.dm.record_lexema_appearances(user_id="test_user", sentence_id=self.s2["id"], is_success=True)
        stats = self.dm.get_user_vocabulary_stats("test_user")
        self.assertIn("total_known", stats)
        self.assertIn("total_learning", stats)
        self.assertIn("by_level", stats)
        self.assertGreaterEqual(stats["total_known"], 3)  # l1, l2, l3

    def test_scoring_engine_integration(self):
        batch = self.scoring.get_batch_sentences(
            all_sentences=self.dm.get_all_sentences(),
            target_score=1.2,
            count=3,
            data_manager=self.dm,
            user_id="test_user",
            max_unknown=2
        )
        self.assertEqual(len(batch), 3)
        for s in batch:
            self.assertIn("fr", s)
            self.assertIn("es", s)

    def test_transcription_mode_zero_new_words(self):
        # In transcription mode, sentences must contain ONLY familiar words (0 unknown words)
        batch = self.dm.get_progressive_sentences(
            user_id="test_user",
            target_score=1.2,
            count=1,
            mode="transcription"
        )
        self.assertEqual(len(batch), 1)
        # s1 only contains base words (l1, l2) which test_user knows initially
        self.assertEqual(batch[0]["id"], self.s1["id"])
        self.assertEqual(batch[0]["unknown_count"], 0)
        self.assertEqual(batch[0]["new_lexemas"], [])

        # Scoring engine integration with mode='transcription'
        batch_se = self.scoring.get_batch_sentences(
            all_sentences=self.dm.get_all_sentences(),
            target_score=1.2,
            count=1,
            data_manager=self.dm,
            user_id="test_user",
            mode="transcription"
        )
        self.assertEqual(len(batch_se), 1)
        self.assertEqual(batch_se[0]["id"], self.s1["id"])
        self.assertEqual(batch_se[0]["unknown_count"], 0)
        self.assertEqual(batch_se[0]["new_lexemas"], [])

    def test_transcription_mode_never_falls_back_to_new_words(self):
        # Even if the familiar sentence is in exclude_ids, transcription must recycle familiar words
        # and NEVER pick sentences with new words (s2, s3, s4, s5)
        batch = self.dm.get_progressive_sentences(
            user_id="test_user",
            target_score=1.2,
            count=1,
            exclude_ids=[self.s1["id"]],
            mode="transcription"
        )
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0]["id"], self.s1["id"])
        self.assertEqual(batch[0]["unknown_count"], 0)
        self.assertEqual(batch[0]["new_lexemas"], [])


if __name__ == "__main__":
    unittest.main()
