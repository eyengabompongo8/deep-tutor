import os
from backend.db import DataManager


class TestDataManager:
    def test_add_and_get_sentence(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        res = dm.add_sentence(
            es="Hola mundo",
            fr="Bonjour le monde",
            level="A1",
            score_ml=88.5,
            source="custom"
        )
        assert res is not None
        assert res["es"] == "Hola mundo"
        assert res["fr"] == "Bonjour le monde"
        assert res["level"] == "A1"
        assert res["score_ml"] == 88.5
        # Check backward compatibility alias
        assert res["score"] == 88.5

    def test_dynamic_language_columns(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        # Adding sentence with extra language column 'en'
        res = dm.add_sentence(
            es="Buenos días",
            fr="Bonjour",
            level="A1",
            score_ml=1.0,
            en="Good morning"
        )
        assert res["en"] == "Good morning"
        assert "en" in dm.get_languages()

        # Add another language column 'de'
        dm.add_language_column("de")
        assert "de" in dm.get_languages()

        # Update sentence with German
        dm.update_sentence(res["id"], de="Guten Morgen")
        updated = dm.get_sentence_by_id(res["id"])
        assert updated["de"] == "Guten Morgen"

    def test_bulk_insert_and_distribution(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        items = [
            {"fr": "Un", "es": "Uno", "level": "A1", "score_ml": 1.0},
            {"fr": "Deux", "es": "Dos", "level": "A1", "score_ml": 1.2},
            {"fr": "Trois", "es": "Tres", "level": "A2", "score_ml": 2.1},
            {"fr": "Quatre", "es": "Cuatro", "level": "B1", "score_ml": 3.4},
        ]
        inserted = dm.bulk_insert_sentences(items, clear_existing=True)
        assert inserted == 4

        all_s = dm.get_all_sentences()
        assert len(all_s) == 4

        a1_s = dm.get_all_sentences(level="A1")
        assert len(a1_s) == 2

        dist = dm.get_level_distribution()
        assert dist["A1"] == 2
        assert dist["A2"] == 1
        assert dist["B1"] == 1
        assert dist["B2"] == 0

    def test_update_and_delete(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        item = dm.add_sentence(es="Gato", fr="Chat", level="A1", score=1.0)
        item_id = item["id"]

        # Update
        success = dm.update_sentence(item_id, fr="Le chat noir", level="A2", score_ml=95.0)
        assert success is True

        updated = dm.get_sentence_by_id(item_id)
        assert updated["fr"] == "Le chat noir"
        assert updated["level"] == "A2"
        assert updated["score"] == 95.0
        assert updated["score_ml"] == 95.0

        # Delete
        del_success = dm.delete_sentence(item_id)
        assert del_success is True
        assert dm.get_sentence_by_id(item_id) is None

    def test_user_modes_and_history(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        # Default modes profile
        p = dm.get_user_modes_profile('test_user')
        assert 'modes' in p
        assert 'translation' in p['modes']
        assert 'transcription' in p['modes']
        assert p['modes']['transcription']['level'] == 'A1'

        # Update transcription mode independently
        dm.update_user_mode('test_user', 'transcription', level='A2', streak=3)
        trans_p = dm.get_user_mode('test_user', 'transcription')
        assert trans_p['level'] == 'A2'
        assert trans_p['streak'] == 3

        # Ensure translation mode remained unchanged
        trad_p = dm.get_user_mode('test_user', 'translation')
        assert trad_p['level'] == 'A1'

        # History recording
        dm.record_exercise_history('test_user', 'transcription', quality=0.88, is_skipped=False)
        hist = dm.get_exercise_history('test_user', 'transcription', limit=5)
        assert len(hist) == 1
        assert hist[0]['quality'] == 0.88

        # Reset mode
        dm.reset_user_mode('test_user', 'transcription', level='B1')
        reset_p = dm.get_user_mode('test_user', 'transcription')
        assert reset_p['level'] == 'B1'
        assert reset_p['streak'] == 0

    def test_audio_cache(self, temp_db_path):
        import hashlib
        dm = DataManager(db_path=temp_db_path)
        text = "Bonjour tout le monde"
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        fake_path = "/tmp/fake_test_audio.wav"
        
        # Non-existent file should return None
        assert dm.get_audio_cache(text_hash) is None

        # Create temporary file to test retrieval
        with open(fake_path, 'w') as f:
            f.write("dummy audio")
        try:
            sent = dm.add_sentence(es="Hola mundo", fr=text, level="A1", score=1.0)
            dm.save_audio_cache(text_hash=text_hash, text=text, file_path=fake_path, sentence_id=sent['id'])
            cached = dm.get_audio_cache(text_hash)
            assert cached is not None
            assert cached['file_path'] == fake_path
            assert cached['text'] == text
        finally:
            if os.path.exists(fake_path):
                os.remove(fake_path)

    def test_user_management(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        # 1. list_users should contain default_user
        users = dm.list_users()
        assert len(users) >= 1
        default_u = [u for u in users if u['id'] == 'default_user'][0]
        assert default_u['name'] == 'Default User'

        # 2. create_user
        new_u = dm.create_user(name="Camille")
        assert "user_id" in new_u
        assert new_u["name"] == "Camille"
        user_id = new_u["user_id"]

        # 3. list_users should now include Camille
        users_after = dm.list_users()
        assert len(users_after) == len(users) + 1
        found = [u for u in users_after if u['id'] == user_id]
        assert len(found) == 1
        assert found[0]['name'] == 'Camille'

        # 4. update_user
        dm.update_user(user_id=user_id, name="Camille Claudel")
        updated_u = dm.get_user(user_id)
        assert updated_u["name"] == "Camille Claudel"

        # 5. delete_user
        del_success = dm.delete_user(user_id)
        assert del_success is True
        assert dm.get_user(user_id) is None

        # 6. Cannot delete the only remaining user
        rem_users = dm.list_users()
        if len(rem_users) == 1:
            cannot_del = dm.delete_user(rem_users[0]['id'])
            assert cannot_del is False

    def test_user_settings(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        # 1. Default settings for default_user
        settings = dm.get_user_settings("default_user")
        assert settings["user_id"] == "default_user"
        assert settings["theme"] == "system"
        assert settings["preferred_voice"] == "default"
        assert settings["correction_threshold"] == 75.0
        assert settings["cards_per_exercise"] == 3
        assert settings["auto_jump"] is True
        assert settings["auto_play_audio"] is True
        assert settings["transcription_retry"] is True
        assert settings["blind_retry"] is False
        assert settings["active_mode"] == "translation"

        # 2. Update settings
        updated = dm.update_user_settings(
            "default_user",
            theme="dark",
            preferred_voice="female",
            correction_threshold=80.0,
            cards_per_exercise=5,
            auto_jump=False,
            auto_play_audio=False,
            transcription_retry=False,
            blind_retry=True,
            active_mode="transcription"
        )
        assert updated["theme"] == "dark"
        assert updated["preferred_voice"] == "female"
        assert updated["correction_threshold"] == 80.0
        assert updated["cards_per_exercise"] == 5
        assert updated["auto_jump"] is False
        assert updated["auto_play_audio"] is False
        assert updated["transcription_retry"] is False
        assert updated["blind_retry"] is True
        assert updated["active_mode"] == "transcription"

        # 3. Create another user and fetch settings (auto creates)
        user_b = dm.create_user("User B")
        b_settings = dm.get_user_settings(user_b["user_id"])
        assert b_settings["cards_per_exercise"] == 3
        assert b_settings["auto_jump"] is True
        assert b_settings["transcription_retry"] is True
        assert b_settings["blind_retry"] is False
        assert b_settings["active_mode"] == "translation"

    def test_vocabulary_lemmas_and_words(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)

        # 1. Upsert lemmas
        l1 = dm.upsert_lemma(lemma="manger", pos="VERB", level="A1", is_base=True)
        l2 = dm.upsert_lemma(lemma="pomme", pos="NOUN", level="A1", is_base=False)
        assert l1 > 0
        assert l2 > 0

        # Bulk upsert lemmas
        bulk_res = dm.bulk_upsert_lemmas([
            {"lemma": "rouge", "pos": "ADJ", "level": "A1", "is_base": False},
            {"lemma": "manger", "pos": "VERB", "level": "A1", "is_base": True},
        ])
        assert ("rouge", "ADJ") in bulk_res
        l3 = bulk_res[("rouge", "ADJ")]

        # Test lemmas dict cache
        lemmas_dict = dm.get_lemmas_dict()
        assert l1 in lemmas_dict
        assert lemmas_dict[l1]["lemma"] == "manger"
        assert lemmas_dict[l1]["pos"] == "VERB"
        assert lemmas_dict[l1]["is_base"] is True

        # 2. Upsert words
        w1 = dm.upsert_word(word="mange", lemma_id=l1, pos="VERB")
        w2 = dm.upsert_word(word="pommes", lemma_id=l2, pos="NOUN")
        assert w1 > 0
        assert w2 > 0

        # Bulk upsert words
        w_bulk = dm.bulk_upsert_words([
            {"word": "rouges", "lemma_id": l3, "pos": "ADJ"}
        ])
        assert ("rouges", l3, "ADJ") in w_bulk
        w3 = w_bulk[("rouges", l3, "ADJ")]

        # 3. Add sentence and link lemmas & words
        sent = dm.add_sentence(fr="Je mange des pommes rouges", es="Como manzanas rojas", level="A1")
        sent_id = sent["id"]

        dm.link_sentence_lemmas(sent_id, [l1, l2, l3])
        dm.link_sentence_words(sent_id, [w1, w2, w3])

        # Test sentence retrieval
        sent_lemmas = dm.get_sentence_lemmas(sent_id)
        assert len(sent_lemmas) == 3
        sent_lemma_names = {r["lemma"] for r in sent_lemmas}
        assert sent_lemma_names == {"manger", "pomme", "rouge"}

        sent_words = dm.get_sentence_words(sent_id)
        assert len(sent_words) == 3
        sent_word_names = {r["word"] for r in sent_words}
        assert sent_word_names == {"mange", "pommes", "rouges"}

        # 4. Record token appearances
        progress = dm.record_token_appearances(user_id="default_user", sentence_id=sent_id, is_success=True)
        assert len(progress) == 3

        # Check user_lemmas and user_words
        with dm._get_connection() as conn:
            ul_rows = conn.execute("SELECT lemma_id, appearance_count, success_count FROM user_lemmas WHERE user_id = 'default_user';").fetchall()
            assert len(ul_rows) == 3
            assert all(r["appearance_count"] == 1 and r["success_count"] == 1 for r in ul_rows)

            uw_rows = conn.execute("SELECT word_id, appearance_count, success_count FROM user_words WHERE user_id = 'default_user';").fetchall()
            assert len(uw_rows) == 3
            assert all(r["appearance_count"] == 1 and r["success_count"] == 1 for r in uw_rows)

        # 5. Known lemmas
        known_ids = dm.get_user_known_lemma_ids(user_id="default_user", target_level="A1")
        assert l1 in known_ids
        assert l2 in known_ids
        assert l3 in known_ids

