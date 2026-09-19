import unittest
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_manager import DataManager

class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.dm = DataManager(db_path=self.temp_db.name)

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            try:
                os.remove(self.temp_db.name)
            except OSError:
                pass

    def test_add_and_get_sentence(self):
        res = self.dm.add_sentence(
            es="Hola mundo",
            fr="Bonjour le monde",
            level="A1",
            score_lexical=1.5,
            score_ml=88.5,
            source="custom"
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["es"], "Hola mundo")
        self.assertEqual(res["fr"], "Bonjour le monde")
        self.assertEqual(res["level"], "A1")
        self.assertEqual(res["score_lexical"], 1.5)
        # Check backward compatibility alias
        self.assertEqual(res["score"], 1.5)
        self.assertEqual(res["score_ml"], 88.5)
        self.assertIn("created_at", res)

    def test_dynamic_language_columns(self):
        # Adding sentence with extra language column 'en'
        res = self.dm.add_sentence(
            es="Buenos días",
            fr="Bonjour",
            level="A1",
            score_lexical=1.0,
            en="Good morning"
        )
        self.assertEqual(res["en"], "Good morning")
        self.assertIn("en", self.dm.get_languages())

        # Add another language column 'de'
        self.dm.add_language_column("de")
        self.assertIn("de", self.dm.get_languages())

        # Update sentence with German
        self.dm.update_sentence(res["id"], de="Guten Morgen")
        updated = self.dm.get_sentence_by_id(res["id"])
        self.assertEqual(updated["de"], "Guten Morgen")

    def test_bulk_insert_and_distribution(self):
        items = [
            {"fr": "Un", "es": "Uno", "level": "A1", "score_lexical": 1.0},
            {"fr": "Deux", "es": "Dos", "level": "A1", "score_lexical": 1.2},
            {"fr": "Trois", "es": "Tres", "level": "A2", "score_lexical": 2.1},
            {"fr": "Quatre", "es": "Cuatro", "level": "B1", "score_lexical": 3.4},
        ]
        inserted = self.dm.bulk_insert_sentences(items, clear_existing=True)
        self.assertEqual(inserted, 4)

        all_s = self.dm.get_all_sentences()
        self.assertEqual(len(all_s), 4)

        a1_s = self.dm.get_all_sentences(level="A1")
        self.assertEqual(len(a1_s), 2)

        dist = self.dm.get_level_distribution()
        self.assertEqual(dist["A1"], 2)
        self.assertEqual(dist["A2"], 1)
        self.assertEqual(dist["B1"], 1)
        self.assertEqual(dist["B2"], 0)

    def test_update_and_delete(self):
        item = self.dm.add_sentence(es="Gato", fr="Chat", level="A1", score=1.0)
        item_id = item["id"]

        # Update
        success = self.dm.update_sentence(item_id, fr="Le chat noir", level="A2", score_lexical=2.2, score_ml=95.0)
        self.assertTrue(success)

        updated = self.dm.get_sentence_by_id(item_id)
        self.assertEqual(updated["fr"], "Le chat noir")
        self.assertEqual(updated["level"], "A2")
        self.assertEqual(updated["score_lexical"], 2.2)
        self.assertEqual(updated["score"], 2.2)
        self.assertEqual(updated["score_ml"], 95.0)

        # Delete
        del_success = self.dm.delete_sentence(item_id)
        self.assertTrue(del_success)
        self.assertIsNone(self.dm.get_sentence_by_id(item_id))

    def test_user_modes_and_history(self):
        # Default modes profile
        p = self.dm.get_user_modes_profile('test_user')
        self.assertIn('modes', p)
        self.assertIn('translation', p['modes'])
        self.assertIn('transcription', p['modes'])
        self.assertEqual(p['modes']['transcription']['score'], 1.20)

        # Update transcription mode independently
        self.dm.update_user_mode('test_user', 'transcription', score=2.50, level='A2', streak=3)
        trans_p = self.dm.get_user_mode('test_user', 'transcription')
        self.assertEqual(trans_p['score'], 2.50)
        self.assertEqual(trans_p['level'], 'A2')
        self.assertEqual(trans_p['streak'], 3)

        # Ensure translation mode remained unchanged
        trad_p = self.dm.get_user_mode('test_user', 'translation')
        self.assertEqual(trad_p['score'], 1.20)

        # History recording
        self.dm.record_exercise_history('test_user', 'transcription', quality=0.88, score_before=2.0, score_after=2.5, delta=0.5, is_skipped=False)
        hist = self.dm.get_exercise_history('test_user', 'transcription', limit=5)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]['delta'], 0.5)

        # Reset mode
        self.dm.reset_user_mode('test_user', 'transcription', start_score=1.50)
        reset_p = self.dm.get_user_mode('test_user', 'transcription')
        self.assertEqual(reset_p['score'], 1.50)
        self.assertEqual(reset_p['streak'], 0)

    def test_audio_cache(self):
        import hashlib
        text = "Bonjour tout le monde"
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        fake_path = "/tmp/fake_test_audio.wav"
        
        # Non-existent file should return None
        self.assertIsNone(self.dm.get_audio_cache(text_hash))

        # Create temporary file to test retrieval
        with open(fake_path, 'w') as f:
            f.write("dummy audio")
        try:
            sent = self.dm.add_sentence(es="Hola mundo", fr=text, level="A1", score=1.0)
            self.dm.save_audio_cache(text_hash=text_hash, text=text, file_path=fake_path, sentence_id=sent['id'])
            cached = self.dm.get_audio_cache(text_hash)
            self.assertIsNotNone(cached)
            self.assertEqual(cached['file_path'], fake_path)
            self.assertEqual(cached['text'], text)
        finally:
            if os.path.exists(fake_path):
                os.remove(fake_path)

    def test_user_management(self):
        # 1. list_users should contain default_user
        users = self.dm.list_users()
        self.assertGreaterEqual(len(users), 1)
        default_u = [u for u in users if u['id'] == 'default_user'][0]
        self.assertEqual(default_u['avatar'], 'owl')

        # 2. create_user
        new_u = self.dm.create_user(name="Camille", avatar="fox")
        self.assertIn("user_id", new_u)
        self.assertEqual(new_u["name"], "Camille")
        self.assertEqual(new_u["avatar"], "fox")
        user_id = new_u["user_id"]

        # 3. list_users should now include Camille
        users_after = self.dm.list_users()
        self.assertEqual(len(users_after), len(users) + 1)
        found = [u for u in users_after if u['id'] == user_id]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['avatar'], 'fox')

        # 4. update_user
        self.dm.update_user(user_id=user_id, name="Camille Claudel", avatar="palette")
        updated_u = self.dm.get_user(user_id)
        self.assertEqual(updated_u["name"], "Camille Claudel")
        self.assertEqual(updated_u["avatar"], "palette")

        # 5. delete_user
        del_success = self.dm.delete_user(user_id)
        self.assertTrue(del_success)
        self.assertIsNone(self.dm.get_user(user_id))

        # 6. Cannot delete the only remaining user
        rem_users = self.dm.list_users()
        if len(rem_users) == 1:
            cannot_del = self.dm.delete_user(rem_users[0]['id'])
            self.assertFalse(cannot_del)

    def test_user_settings(self):
        # 1. Default settings for default_user
        settings = self.dm.get_user_settings("default_user")
        self.assertEqual(settings["user_id"], "default_user")
        self.assertEqual(settings["daily_goal"], 10)
        self.assertEqual(settings["theme"], "system")
        self.assertEqual(settings["preferred_voice"], "default")
        self.assertEqual(settings["correction_threshold"], 75.0)
        self.assertEqual(settings["cards_per_exercise"], 3)

        # 2. Update settings
        updated = self.dm.update_user_settings(
            "default_user",
            daily_goal=15,
            theme="dark",
            preferred_voice="female",
            correction_threshold=80.0,
            cards_per_exercise=5
        )
        self.assertEqual(updated["daily_goal"], 15)
        self.assertEqual(updated["theme"], "dark")
        self.assertEqual(updated["preferred_voice"], "female")
        self.assertEqual(updated["correction_threshold"], 80.0)
        self.assertEqual(updated["cards_per_exercise"], 5)

        # 3. Create another user and fetch settings (auto creates)
        user_b = self.dm.create_user("User B")
        b_settings = self.dm.get_user_settings(user_b["user_id"])
        self.assertEqual(b_settings["daily_goal"], 10)
        self.assertEqual(b_settings["cards_per_exercise"], 3)

if __name__ == '__main__':
    unittest.main()
