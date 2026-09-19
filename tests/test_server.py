import sys
import os
import unittest
import tempfile
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import server
from data_manager import DataManager
from scoring_engine import ScoringEngine
from tts_engine import get_tts_engine

class TestServerAPI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = os.path.join(self.temp_dir.name, 'test_server.db')
        self.test_profile_path = os.path.join(self.temp_dir.name, 'test_profile.json')
        self.dm = DataManager(db_path=self.test_db_path)
        self.dm.bulk_insert_sentences([
            {"es": "Hola mundo", "fr": "Bonjour le monde", "level": "A1", "score_lexical": 1.0, "score_ml": 85.0},
            {"es": "El gato duerme", "fr": "Le chat dort", "level": "A1", "score_lexical": 1.2, "score_ml": 88.0},
            {"es": "Tengo que irme ahora.", "fr": "Je dois partir maintenant.", "level": "A2", "score_lexical": 2.2, "score_ml": 90.0},
            {"es": "El restaurante está abierto", "fr": "Le restaurant est ouvert", "level": "B1", "score_lexical": 3.2, "score_ml": 92.0},
        ])
        self.scoring = ScoringEngine(profile_file=self.test_profile_path, data_manager=self.dm)
        
        self.orig_dm = server.data_manager
        self.orig_se = server.scoring_engine
        self.orig_tts = server.tts_engine
        
        server.data_manager = self.dm
        server.scoring_engine = self.scoring
        server.tts_engine = get_tts_engine(data_manager=self.dm)
        self.client = TestClient(server.app)

    def tearDown(self):
        server.data_manager = self.orig_dm
        server.scoring_engine = self.orig_se
        server.tts_engine = self.orig_tts
        self.temp_dir.cleanup()

    def test_get_profile(self):
        res = self.client.get("/api/user/profile")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("score", data)
        self.assertIn("level", data)

    def test_get_batch(self):
        res = self.client.get("/api/exercises/batch?score=1.5&count=3")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["count"], 3)
        self.assertEqual(len(data["items"]), 3)
        for item in data["items"]:
            self.assertIn("es", item)
            self.assertIn("fr", item)

    def test_evaluate_skipped_sentence(self):
        payload = {
            "sentence_id": 999,
            "source_text": "Hola mundo",
            "student_text": "",
            "reference_text": "Bonjour le monde",
            "sentence_score": 1.0,
            "is_skipped": True
        }
        res = self.client.post("/api/exercises/evaluate-single", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["is_skipped"])
        self.assertEqual(data["quality"], 0.0)

    def test_complete_batch(self):
        payload = {
            "current_score": 1.20,
            "evaluated_items": [
                {"sentence_id": 1, "sentence_score": 1.0, "quality": 0.90, "is_skipped": False},
                {"sentence_id": 2, "sentence_score": 1.0, "quality": 0.85, "is_skipped": False},
            ]
        }
        res = self.client.post("/api/exercises/complete-batch", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("new_score", data)
        self.assertIn("delta", data)
        self.assertGreater(data["delta"], 0)

    def test_stats(self):
        res = self.client.get("/api/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_sentences", data)
        self.assertIn("level_distribution", data)

    @pytest.mark.ml
    def test_hint_endpoint(self):
        payload = {
            "sentence_id": 1,
            "source_text": "Tengo que irme ahora.",
            "reference_text": "Je dois partir maintenant.",
            "selected_indices": [0, 1]
        }
        res = self.client.post("/api/exercises/hint", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("src_words", data)
        self.assertIn("tgt_words", data)
        self.assertIn("aligned_targets_map", data)
        self.assertIn("hint_translation", data)

    @pytest.mark.ml
    def test_evaluate_valid_french_sentence(self):
        payload = {
            "sentence_id": 1,
            "source_text": "Tengo que irme ahora.",
            "student_text": "Je dois partir maintenant.",
            "reference_text": "Je dois partir maintenant.",
            "sentence_score": 1.0,
            "is_skipped": False
        }
        res = self.client.post("/api/exercises/evaluate-single", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["is_skipped"])
        self.assertGreater(data["composite_score"], 70.0)
        self.assertGreater(data["quality"], 0.70)
        self.assertTrue(data["is_french"])
        self.assertIn("s_sem", data)
        self.assertIn("s_nmt", data)
        self.assertIn("s_fluency", data)
        self.assertIn("vocabulary_progress", data)
        self.assertEqual(len(data.get("syntactical_errors", [])), 0)

    def test_get_vocabulary_endpoints(self):
        res1 = self.client.get("/api/user/vocabulary")
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertIn("total_known", data1)
        self.assertIn("by_level", data1)

        res2 = self.client.get("/api/user/vocab")
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data1["total_known"], data2["total_known"])

    def test_transcription_endpoints(self):
        # 1. Profile with mode
        res_prof = self.client.get("/api/user/profile?mode=transcription")
        self.assertEqual(res_prof.status_code, 200)
        data_prof = res_prof.json()
        self.assertEqual(data_prof.get("current_mode"), "transcription")
        self.assertIn("modes", data_prof)
        self.assertIn("transcription", data_prof["modes"])

        # 2. Evaluate transcription
        eval_payload = {
            "sentence_id": 42,
            "student_text": "Bonjour tout le monde",
            "reference_text": "Bonjour tout le monde",
            "sentence_score": 1.0,
            "is_skipped": False
        }
        res_eval = self.client.post("/api/exercises/evaluate-transcription", json=eval_payload)
        self.assertEqual(res_eval.status_code, 200)
        data_eval = res_eval.json()
        self.assertEqual(data_eval["sentence_id"], 42)
        self.assertTrue(data_eval["is_exact"])
        self.assertEqual(data_eval["composite_score"], 100.0)
        self.assertIn("diff", data_eval)

        # 3. Batch in transcription mode
        res_batch = self.client.get("/api/exercises/batch?mode=transcription&count=2")
        self.assertEqual(res_batch.status_code, 200)
        data_batch = res_batch.json()
        self.assertEqual(data_batch.get("mode"), "transcription")
        self.assertEqual(len(data_batch.get("items", [])), 2)

        # 4. Synthesize audio endpoint
        res_audio = self.client.get("/api/audio/synthesize?text=Bonjour")
        self.assertEqual(res_audio.status_code, 200)
        self.assertIn(res_audio.headers.get("content-type"), ["audio/mpeg", "audio/wav"])
        self.assertGreater(len(res_audio.content), 0)

    def test_user_endpoints(self):
        # 1. GET /api/users
        res = self.client.get("/api/users")
        self.assertEqual(res.status_code, 200)
        users = res.json()
        self.assertIsInstance(users, list)
        self.assertGreaterEqual(len(users), 1)

        # 2. POST /api/users
        create_res = self.client.post("/api/users", json={"name": "Émile", "avatar": "sparkles"})
        self.assertEqual(create_res.status_code, 200)
        new_u = create_res.json()
        self.assertEqual(new_u["name"], "Émile")
        self.assertEqual(new_u["avatar"], "sparkles")
        user_id = new_u["user_id"]

        # 3. GET /api/user/profile?user_id=...
        prof_res = self.client.get(f"/api/user/profile?user_id={user_id}")
        self.assertEqual(prof_res.status_code, 200)
        prof = prof_res.json()
        self.assertEqual(prof["name"], "Émile")
        self.assertEqual(prof["avatar"], "sparkles")

        # 4. PUT /api/users/{user_id}
        put_res = self.client.put(f"/api/users/{user_id}", json={"name": "Émile Zola", "avatar": "book"})
        self.assertEqual(put_res.status_code, 200)
        updated_u = put_res.json()
        self.assertEqual(updated_u["name"], "Émile Zola")
        self.assertEqual(updated_u["avatar"], "book")

        # 5. DELETE /api/users/{user_id}
        del_res = self.client.delete(f"/api/users/{user_id}")
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(del_res.json()["status"], "deleted")

        # 6. Cannot delete the only remaining user
        rem_users = self.client.get("/api/users").json()
        if len(rem_users) == 1:
            err_del = self.client.delete(f"/api/users/{rem_users[0]['id']}")
            self.assertEqual(err_del.status_code, 400)

    def test_user_settings_endpoints(self):
        # 1. GET /api/user/settings
        res = self.client.get("/api/user/settings?user_id=default_user")
        self.assertEqual(res.status_code, 200)
        settings = res.json()
        self.assertEqual(settings["user_id"], "default_user")
        self.assertEqual(settings["cards_per_exercise"], 3)
        self.assertEqual(settings["daily_goal"], 10)

        # 2. PUT /api/user/settings
        put_res = self.client.put("/api/user/settings?user_id=default_user", json={
            "cards_per_exercise": 2,
            "theme": "dark",
            "daily_goal": 20
        })
        self.assertEqual(put_res.status_code, 200)
        updated = put_res.json()
        self.assertEqual(updated["cards_per_exercise"], 2)
        self.assertEqual(updated["theme"], "dark")
        self.assertEqual(updated["daily_goal"], 20)

        # 3. Verify get_exercises_batch without count uses configured cards_per_exercise (2)
        batch_res = self.client.get("/api/exercises/batch?user_id=default_user")
        self.assertEqual(batch_res.status_code, 200)
        batch_data = batch_res.json()
        self.assertEqual(len(batch_data["items"]), 2)

if __name__ == "__main__":
    unittest.main()
