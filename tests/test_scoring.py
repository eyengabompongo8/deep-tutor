import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scoring_engine import ScoringEngine, score_to_level
from data_manager import DataManager

import tempfile

class TestScoringEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_profile_path = os.path.join(self.temp_dir.name, 'test_profile.json')
        self.test_db_path = os.path.join(self.temp_dir.name, 'test_app.db')
        self.dm = DataManager(db_path=self.test_db_path)
        self.dm.bulk_insert_sentences([
            {"es": "Hola mundo", "fr": "Bonjour le monde", "level": "A1", "score_lexical": 1.0},
            {"es": "El gato", "fr": "Le chat", "level": "A1", "score_lexical": 1.2},
            {"es": "El perro", "fr": "Le chien", "level": "A1", "score_lexical": 1.5},
            {"es": "La casa", "fr": "La maison", "level": "A2", "score_lexical": 2.1},
        ])
        self.engine = ScoringEngine(profile_file=self.test_profile_path, data_manager=self.dm)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_score_to_level(self):
        self.assertEqual(score_to_level(1.00), 'A1')
        self.assertEqual(score_to_level(1.99), 'A1')
        self.assertEqual(score_to_level(2.00), 'A2')
        self.assertEqual(score_to_level(3.50), 'B1')
        self.assertEqual(score_to_level(4.20), 'B2')
        self.assertEqual(score_to_level(5.80), 'C1')
        self.assertEqual(score_to_level(6.00), 'C2')

    def test_batch_sampling(self):
        all_s = self.dm.get_all_sentences()
        batch = self.engine.get_batch_sentences(all_s, target_score=1.5, count=3)
        self.assertEqual(len(batch), 3)
        for item in batch:
            self.assertIn('id', item)
            self.assertIn('es', item)
            self.assertIn('fr', item)
            self.assertIn('score', item)

    def test_score_progression_positive(self):
        items = [
            {"sentence_score": 1.0, "quality": 0.95, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.90, "is_skipped": False},
            {"sentence_score": 2.0, "quality": 0.85, "is_skipped": False},
        ]
        res = self.engine.calculate_score_adjustment(1.20, items)
        self.assertGreater(res["delta"], 0)
        self.assertGreater(res["new_score"], 1.20)
        self.assertEqual(res["new_score"], self.engine.profile["score"])

    def test_score_progression_negative_skips(self):
        items = [
            {"sentence_score": 1.0, "quality": 0.0, "is_skipped": True},
            {"sentence_score": 1.0, "quality": 0.0, "is_skipped": True},
            {"sentence_score": 1.0, "quality": 0.0, "is_skipped": True},
        ]
        res = self.engine.calculate_score_adjustment(2.00, items)
        self.assertLess(res["delta"], 0)
        self.assertLess(res["new_score"], 2.00)

if __name__ == "__main__":
    unittest.main()
