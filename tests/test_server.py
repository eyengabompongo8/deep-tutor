import pytest


class TestProfileEndpoints:
    def test_get_profile(self, test_client):
        res = test_client.get("/api/user/profile")
        assert res.status_code == 200
        data = res.json()
        assert "score" in data
        assert "level" in data

    def test_get_profile_with_mode(self, test_client):
        res = test_client.get("/api/user/profile?mode=transcription")
        assert res.status_code == 200
        data = res.json()
        assert data.get("current_mode") == "transcription"
        assert "modes" in data
        assert "transcription" in data["modes"]


class TestExerciseEndpoints:
    def test_get_batch(self, test_client):
        res = test_client.get("/api/exercises/batch?score=1.5&count=3")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 3
        assert len(data["items"]) == 3
        for item in data["items"]:
            assert "es" in item
            assert "fr" in item

    def test_evaluate_skipped_sentence(self, test_client):
        payload = {
            "sentence_id": 999,
            "source_text": "Hola mundo",
            "student_text": "",
            "reference_text": "Bonjour le monde",
            "sentence_score": 1.0,
            "is_skipped": True
        }
        res = test_client.post("/api/exercises/evaluate-single", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["is_skipped"] is True
        assert data["quality"] == 0.0

    def test_evaluate_skipped_sentence_with_vocab(self, test_client, isolated_data_manager):
        lid = isolated_data_manager.upsert_lexema(lemma="bonjour", pos="INT", level="A1")
        isolated_data_manager.link_sentence_lexemas(1, [lid])

        payload = {
            "sentence_id": 1,
            "source_text": "Hola mundo",
            "student_text": "",
            "reference_text": "Bonjour le monde",
            "sentence_score": 1.0,
            "is_skipped": True,
            "user_id": "default_user"
        }
        res = test_client.post("/api/exercises/evaluate-single", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["is_skipped"] is True
        assert len(data["vocabulary_progress"]) == 1
        item = data["vocabulary_progress"][0]
        assert item["lemma"] == "bonjour"
        assert item["lexema"] == "bonjour"
        assert item["id"] == lid
        assert item["lemma_id"] == lid
        assert item["appearance_count"] == 1
        assert item["appearances"] == 1
        assert "is_known" in item
        assert "mastered" in item

        # Retry evaluation of same card
        retry_payload = {
            "sentence_id": 1,
            "source_text": "Hola mundo",
            "student_text": "",
            "reference_text": "Bonjour le monde",
            "sentence_score": 1.0,
            "is_skipped": True,
            "user_id": "default_user",
            "is_retry": True
        }
        res_retry = test_client.post("/api/exercises/evaluate-single", json=retry_payload)
        assert res_retry.status_code == 200
        data_retry = res_retry.json()
        assert len(data_retry["vocabulary_progress"]) == 1
        retry_item = data_retry["vocabulary_progress"][0]
        assert retry_item["appearance_count"] == 1  # Unchanged!

    def test_complete_batch(self, test_client):
        payload = {
            "current_score": 1.20,
            "evaluated_items": [
                {"sentence_id": 1, "sentence_score": 1.0, "quality": 0.90, "is_skipped": False},
                {"sentence_id": 2, "sentence_score": 1.0, "quality": 0.85, "is_skipped": False},
            ]
        }
        res = test_client.post("/api/exercises/complete-batch", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "new_score" in data
        assert "delta" in data
        assert data["delta"] > 0

    def test_stats(self, test_client):
        res = test_client.get("/api/stats")
        assert res.status_code == 200
        data = res.json()
        assert "total_sentences" in data
        assert "level_distribution" in data

    def test_get_vocabulary_endpoints(self, test_client):
        res1 = test_client.get("/api/user/vocabulary")
        assert res1.status_code == 200
        data1 = res1.json()
        assert "total_known" in data1
        assert "by_level" in data1

        res2 = test_client.get("/api/user/vocab")
        assert res2.status_code == 200
        data2 = res2.json()
        assert data1["total_known"] == data2["total_known"]


class TestTranscriptionEndpoints:
    def test_evaluate_transcription(self, test_client):
        eval_payload = {
            "sentence_id": 42,
            "student_text": "Bonjour tout le monde",
            "reference_text": "Bonjour tout le monde",
            "sentence_score": 1.0,
            "is_skipped": False
        }
        res_eval = test_client.post("/api/exercises/evaluate-transcription", json=eval_payload)
        assert res_eval.status_code == 200
        data_eval = res_eval.json()
        assert data_eval["sentence_id"] == 42
        assert data_eval["is_exact"] is True
        assert data_eval["is_correct"] is True
        assert data_eval["quality"] == 1.0
        assert data_eval["composite_score"] == 100.0
        assert "diff" in data_eval
        assert data_eval["corrected_text"] == "Bonjour tout le monde"
        assert data_eval["autocorrect_msgs"] == []

    def test_evaluate_transcription_autocorrects_punctuation(self, test_client):
        eval_payload = {
            "sentence_id": 43,
            "student_text": "Comment vas-tu",
            "reference_text": "Comment vas-tu ?",
            "sentence_score": 1.0,
            "is_skipped": False
        }
        res_eval = test_client.post("/api/exercises/evaluate-transcription", json=eval_payload)
        assert res_eval.status_code == 200
        data_eval = res_eval.json()
        assert data_eval["student_text"] == "Comment vas-tu"
        assert data_eval["corrected_text"] == "Comment vas-tu ?"
        assert "added '?'" in data_eval["autocorrect_msgs"]
        assert data_eval["is_exact"] is True
        assert data_eval["is_correct"] is True
        assert data_eval["quality"] == 1.0

    def test_evaluate_transcription_with_vocab(self, test_client, isolated_data_manager):
        lid = isolated_data_manager.upsert_lexema(lemma="bonjour", pos="INT", level="A1")
        isolated_data_manager.link_sentence_lexemas(1, [lid])

        eval_payload = {
            "sentence_id": 1,
            "student_text": "Bonjour le monde",
            "reference_text": "Bonjour le monde",
            "sentence_score": 1.0,
            "is_skipped": False,
            "user_id": "default_user",
        }
        res_eval = test_client.post("/api/exercises/evaluate-transcription", json=eval_payload)
        assert res_eval.status_code == 200
        data_eval = res_eval.json()
        assert len(data_eval["vocabulary_progress"]) == 1
        item = data_eval["vocabulary_progress"][0]
        assert item["lemma"] == "bonjour"
        assert item["lexema"] == "bonjour"
        assert item["id"] == lid
        assert item["lemma_id"] == lid
        assert item["appearance_count"] == 1
        assert item["success_count"] == 1
        assert "is_known" in item
        assert "mastered" in item

    def test_transcription_batch(self, test_client):
        res_batch = test_client.get("/api/exercises/batch?mode=transcription&count=2")
        assert res_batch.status_code == 200
        data_batch = res_batch.json()
        assert data_batch.get("mode") == "transcription"
        assert len(data_batch.get("items", [])) == 2

    def test_synthesize_audio_endpoint(self, test_client):
        res_audio = test_client.get("/api/audio/synthesize?text=Bonjour")
        assert res_audio.status_code == 200
        assert res_audio.headers.get("content-type") in ["audio/mpeg", "audio/wav"]
        assert len(res_audio.content) > 0


class TestUserEndpoints:
    def test_user_lifecycle(self, test_client):
        # 1. GET /api/users
        res = test_client.get("/api/users")
        assert res.status_code == 200
        users = res.json()
        assert isinstance(users, list)
        assert len(users) >= 1

        # 2. POST /api/users
        create_res = test_client.post("/api/users", json={"name": "Émile"})
        assert create_res.status_code == 200
        new_u = create_res.json()
        assert new_u["name"] == "Émile"
        user_id = new_u["user_id"]

        # 3. GET /api/user/profile?user_id=...
        prof_res = test_client.get(f"/api/user/profile?user_id={user_id}")
        assert prof_res.status_code == 200
        prof = prof_res.json()
        assert prof["name"] == "Émile"

        # 4. PUT /api/users/{user_id}
        put_res = test_client.put(f"/api/users/{user_id}", json={"name": "Émile Zola"})
        assert put_res.status_code == 200
        updated_u = put_res.json()
        assert updated_u["name"] == "Émile Zola"

        # 5. DELETE /api/users/{user_id}
        del_res = test_client.delete(f"/api/users/{user_id}")
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "deleted"

        # 6. Cannot delete the only remaining user
        rem_users = test_client.get("/api/users").json()
        if len(rem_users) == 1:
            err_del = test_client.delete(f"/api/users/{rem_users[0]['id']}")
            assert err_del.status_code == 400


class TestSettingsEndpoints:
    def test_user_settings(self, test_client):
        # 1. GET /api/user/settings
        res = test_client.get("/api/user/settings?user_id=default_user")
        assert res.status_code == 200
        settings = res.json()
        assert settings["user_id"] == "default_user"
        assert settings["cards_per_exercise"] == 3
        assert settings["auto_jump"] is True
        assert settings["auto_play_audio"] is True
        assert settings["transcription_retry"] is True
        assert settings["blind_retry"] is False
        assert settings["active_mode"] == "translation"

        # 2. PUT /api/user/settings
        put_res = test_client.put("/api/user/settings?user_id=default_user", json={
            "cards_per_exercise": 2,
            "theme": "dark",
            "auto_jump": False,
            "auto_play_audio": False,
            "transcription_retry": False,
            "blind_retry": True,
            "active_mode": "transcription"
        })
        assert put_res.status_code == 200
        updated = put_res.json()
        assert updated["cards_per_exercise"] == 2
        assert updated["theme"] == "dark"
        assert updated["auto_jump"] is False
        assert updated["auto_play_audio"] is False
        assert updated["transcription_retry"] is False
        assert updated["blind_retry"] is True
        assert updated["active_mode"] == "transcription"

        # 3. Verify get_exercises_batch without count uses configured cards_per_exercise (2)
        batch_res = test_client.get("/api/exercises/batch?user_id=default_user")
        assert batch_res.status_code == 200
        batch_data = batch_res.json()
        assert len(batch_data["items"]) == 2

    def test_mixed_mode_settings_and_batch(self, test_client):
        # 1. Update mixed_translation_ratio and active_mode to mixed
        put_res = test_client.put("/api/user/settings?user_id=default_user", json={
            "mixed_translation_ratio": 60,
            "active_mode": "mixed"
        })
        assert put_res.status_code == 200
        data = put_res.json()
        assert data["mixed_translation_ratio"] == 60
        assert data["active_mode"] == "mixed"

        # 2. Request mixed batch
        res = test_client.get("/api/exercises/batch?mode=mixed&count=4&user_id=default_user")
        assert res.status_code == 200
        batch = res.json()
        assert "items" in batch
        assert len(batch["items"]) == 4

        # Verify card_type tags
        card_types = [item.get("card_type") for item in batch["items"]]
        assert "translation" in card_types
        assert "transcription" in card_types

        # Verify transcription cards have unknown_count == 0 (no new lemmas)
        for item in batch["items"]:
            if item.get("card_type") == "transcription":
                assert item.get("unknown_count", 0) == 0

        # 3. Complete batch in mixed mode: score must not change (delta == 0.0)
        profile_before = test_client.get("/api/user/profile?mode=mixed&user_id=default_user").json()
        score_before = profile_before["score"]

        complete_res = test_client.post("/api/exercises/complete-batch", json={
            "current_score": score_before,
            "mode": "mixed",
            "user_id": "default_user",
            "evaluated_items": [
                {"sentence_id": batch["items"][0]["id"], "sentence_score": 1.0, "quality": 1.0, "is_correct": True, "is_skipped": False},
                {"sentence_id": batch["items"][1]["id"], "sentence_score": 1.0, "quality": 1.0, "is_correct": True, "is_skipped": False},
            ]
        })
        assert complete_res.status_code == 200
        summary = complete_res.json()
        assert summary["delta"] == 0.0
        assert summary["new_score"] == score_before

        profile_after = test_client.get("/api/user/profile?mode=mixed&user_id=default_user").json()
        assert profile_after["score"] == score_before


@pytest.mark.ml
class TestMLEndpoints:
    def test_hint_endpoint(self, test_client):
        payload = {
            "sentence_id": 1,
            "source_text": "Tengo que irme ahora.",
            "reference_text": "Je dois partir maintenant.",
            "selected_indices": [0, 1]
        }
        res = test_client.post("/api/exercises/hint", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "src_words" in data
        assert "tgt_words" in data
        assert "aligned_targets_map" in data
        assert "hint_translation" in data

    def test_evaluate_valid_french_sentence(self, test_client):
        payload = {
            "sentence_id": 1,
            "source_text": "Tengo que irme ahora.",
            "student_text": "Je dois partir maintenant.",
            "reference_text": "Je dois partir maintenant.",
            "sentence_score": 1.0,
            "is_skipped": False
        }
        res = test_client.post("/api/exercises/evaluate-single", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["is_skipped"] is False
        assert data["composite_score"] > 70.0
        assert data["quality"] > 0.70
        assert data["is_french"] is True
        assert "s_sem" in data
        assert "s_nmt" in data
        assert "s_fluency" in data
        assert "vocabulary_progress" in data
        assert len(data.get("syntactical_errors", [])) == 0


class TestVocabProgressSchema:
    def test_schema_compat_modern(self):
        from backend.api.schemas import VocabProgressItem
        item = VocabProgressItem.model_validate({
            "lemma_id": 1,
            "lemma": "parler",
            "appearance_count": 2,
            "success_count": 2,
            "is_known": 1,
        })
        assert item.id == 1
        assert item.lemma_id == 1
        assert item.lemma == "parler"
        assert item.lexema == "parler"
        assert item.appearance_count == 2
        assert item.appearances == 2
        assert item.is_known == 1
        assert item.mastered is True

    def test_schema_compat_legacy(self):
        from backend.api.schemas import VocabProgressItem
        item = VocabProgressItem.model_validate({
            "id": 2,
            "lexema": "écouter",
            "appearances": 3,
            "mastered": False,
        })
        assert item.id == 2
        assert item.lemma_id == 2
        assert item.lemma == "écouter"
        assert item.lexema == "écouter"
        assert item.appearance_count == 3
        assert item.appearances == 3
        assert item.is_known == 0
        assert item.mastered is False
