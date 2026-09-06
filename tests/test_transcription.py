import sys
import os
import unittest
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scoring_engine import ScoringEngine, score_to_level
from data_manager import DataManager


class TestTranscriptionEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.dm = DataManager(db_path=self.temp_db.name)
        self.engine = ScoringEngine(data_manager=self.dm)

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            try:
                os.remove(self.temp_db.name)
            except OSError:
                pass

    def test_exact_match(self):
        ref = "Bonjour tout le monde, comment allez-vous ?"
        stu = "Bonjour tout le monde, comment allez-vous ?"
        res = self.engine.evaluate_transcription(stu, ref)
        self.assertTrue(res["is_exact"])
        self.assertGreaterEqual(res["quality"], 0.98)
        self.assertEqual(res["composite_score"], 100.0)
        self.assertTrue(all(d["type"] == "exact" for d in res["diff"]))

    def test_case_and_punctuation_tolerance(self):
        ref = "Il fait très beau aujourd'hui !"
        stu = "il fait très beau aujourd'hui"
        res = self.engine.evaluate_transcription(stu, ref)
        self.assertGreaterEqual(res["quality"], 0.95)
        self.assertTrue(all(d["type"] == "exact" for d in res["diff"]))

    def test_accent_detection(self):
        ref = "Il est très fatigué."
        stu = "Il est tres fatigue."
        res = self.engine.evaluate_transcription(stu, ref)
        accent_diffs = [d for d in res["diff"] if d["type"] == "accent"]
        self.assertGreaterEqual(len(accent_diffs), 1)
        self.assertGreater(res["quality"], 0.70)
        self.assertLess(res["quality"], 1.00)

    def test_typo_detection(self):
        ref = "Nous allons au restaurant ce soir."
        stu = "Nous alons au restautant ce soir."
        res = self.engine.evaluate_transcription(stu, ref)
        typos = [d for d in res["diff"] if d["type"] == "typo"]
        self.assertGreaterEqual(len(typos), 1)
        self.assertGreater(res["quality"], 0.60)

    def test_missing_and_extra_words(self):
        ref = "Je dois partir maintenant."
        stu = "Je dois."
        res = self.engine.evaluate_transcription(stu, ref)
        missings = [d for d in res["diff"] if d["type"] == "missing"]
        self.assertGreaterEqual(len(missings), 1)
        self.assertLess(res["quality"], 0.70)

    def test_skipped_transcription(self):
        ref = "Bonjour le monde."
        res = self.engine.evaluate_transcription("", ref, is_skipped=True)
        self.assertEqual(res["quality"], 0.0)
        self.assertEqual(res["composite_score"], 0.0)
        self.assertFalse(res["is_exact"])

    def test_multimode_scoring_isolation(self):
        # Initial state for translation and transcription
        prof_trans = self.engine.get_profile(user_id="default_user", mode="translation")
        prof_audio = self.engine.get_profile(user_id="default_user", mode="transcription")
        self.assertEqual(prof_trans["score"], 1.20)
        self.assertEqual(prof_audio["score"], 1.20)

        # Complete batch in transcription mode with high quality
        items = [
            {"sentence_score": 1.0, "quality": 0.95, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.90, "is_skipped": False},
            {"sentence_score": 2.0, "quality": 0.92, "is_skipped": False},
        ]
        res = self.engine.calculate_score_adjustment(
            current_score=1.20,
            evaluated_items=items,
            mode="transcription",
            user_id="default_user"
        )
        self.assertGreater(res["delta"], 0)
        self.assertGreater(res["new_score"], 1.20)

        # Check that transcription mode score updated in SQLite
        updated_audio = self.engine.get_profile(user_id="default_user", mode="transcription")
        self.assertEqual(updated_audio["score"], res["new_score"])
        self.assertEqual(updated_audio["streak"], 1)

        # Verify translation mode score remained at initial 1.20
        updated_trans = self.engine.get_profile(user_id="default_user", mode="translation")
        self.assertEqual(updated_trans["score"], 1.20)
        self.assertEqual(updated_trans["streak"], 0)

    def test_transcription_batch_excludes_new_words(self):
        # Seed lexemas: 2 base words, 1 advanced unknown word
        l1 = self.dm.upsert_lexema(lemma="bonjour", pos="NOUN", level="A1", is_base=True)
        l2 = self.dm.upsert_lexema(lemma="monde", pos="NOUN", level="A1", is_base=True)
        l3 = self.dm.upsert_lexema(lemma="épistémologie", pos="NOUN", level="C2", is_base=False)

        # s1: familiar words only (0 new words)
        s1 = self.dm.add_sentence(es="Hola mundo", fr="Bonjour monde", level="A1", score=1.0)
        self.dm.link_sentence_lexemas(s1["id"], [l1, l2])

        # s2: has an unknown word (l3)
        s2 = self.dm.add_sentence(es="La epistemología", fr="L'épistémologie", level="C2", score=5.8)
        self.dm.link_sentence_lexemas(s2["id"], [l3])

        all_s = self.dm.get_all_sentences()
        batch = self.engine.get_batch_sentences(
            all_sentences=all_s,
            target_score=1.20,
            count=2,
            data_manager=self.dm,
            user_id="default_user",
            mode="transcription"
        )

        self.assertGreater(len(batch), 0)
        for s in batch:
            self.assertEqual(s.get("unknown_count", 0), 0)
            self.assertEqual(s.get("new_lexemas", []), [])
            self.assertNotEqual(s["id"], s2["id"])

    def test_retry_scoring_adjustment(self):
        # Initial complete batch
        items = [
            {"sentence_score": 1.0, "quality": 0.50, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.50, "is_skipped": False},
        ]
        res1 = self.engine.calculate_score_adjustment(
            current_score=1.20,
            evaluated_items=items,
            mode="transcription",
            user_id="default_user",
            is_retry=False
        )
        audio_prof1 = self.engine.get_profile(user_id="default_user", mode="transcription")
        initial_completed = audio_prof1["total_completed"]
        initial_streak = audio_prof1["streak"]

        # Retry one sentence with 1.0 quality: recalculate with is_retry=True from base score
        items_retry = [
            {"sentence_score": 1.0, "quality": 1.00, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.50, "is_skipped": False},
        ]
        res2 = self.engine.calculate_score_adjustment(
            current_score=1.20,
            evaluated_items=items_retry,
            mode="transcription",
            user_id="default_user",
            is_retry=True
        )
        audio_prof2 = self.engine.get_profile(user_id="default_user", mode="transcription")

        # Score must be higher with retry quality improvement
        self.assertGreater(res2["new_score"], res1["new_score"])
        # Total completed count should NOT be incremented again
        self.assertEqual(audio_prof2["total_completed"], initial_completed)
        # Streak should NOT be incremented again
    def test_first_try_score_used_for_level(self):
        # Sentence 1: first try had quality 0.40 (errors), retry achieved 1.00
        # Sentence 2: first try had quality 0.85
        # Per specification: the final score level calculation strictly uses the first try!
        first_try_items = [
            {"sentence_score": 1.0, "quality": 0.40, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.85, "is_skipped": False},
        ]
        res = self.engine.calculate_score_adjustment(
            current_score=1.20,
            evaluated_items=first_try_items,
            mode="transcription",
            user_id="default_user"
        )
        audio_prof = self.engine.get_profile(user_id="default_user", mode="transcription")
        self.assertEqual(audio_prof["score"], res["new_score"])
        # Delta reflects the first-try quality (avg 0.62)
        self.assertEqual(res["avg_quality"], 0.62)


if __name__ == '__main__':
    unittest.main()
