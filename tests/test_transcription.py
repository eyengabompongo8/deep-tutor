from backend.scoring import ScoringEngine
from backend.db import DataManager


class TestTranscriptionEvaluation:
    def test_exact_match(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        engine = ScoringEngine(data_manager=dm)
        ref = "Bonjour tout le monde, comment allez-vous ?"
        stu = "Bonjour tout le monde, comment allez-vous ?"
        res = engine.evaluate_transcription(stu, ref)
        assert res["is_exact"] is True
        assert res["is_correct"] is True
        assert res["quality"] == 1.0
        assert res["composite_score"] == 100.0
        assert all(d["type"] == "exact" for d in res["diff"])

    def test_case_and_punctuation_tolerance(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        engine = ScoringEngine(data_manager=dm)
        ref = "Il fait très beau aujourd'hui !"
        stu = "il fait très beau aujourd'hui"
        res = engine.evaluate_transcription(stu, ref)
        assert res["is_exact"] is True
        assert res["is_correct"] is True
        assert res["quality"] == 1.0
        assert all(d["type"] == "exact" for d in res["diff"])

    def test_accent_detection(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        engine = ScoringEngine(data_manager=dm)
        ref = "Il est très fatigué."
        stu = "Il est tres fatigue."
        res = engine.evaluate_transcription(stu, ref)
        accent_diffs = [d for d in res["diff"] if d["type"] == "accent"]
        assert len(accent_diffs) >= 1
        assert res["is_correct"] is False
        assert res["quality"] == 0.0

    def test_typo_detection(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        engine = ScoringEngine(data_manager=dm)
        ref = "Nous allons au restaurant ce soir."
        stu = "Nous alons au restautant ce soir."
        res = engine.evaluate_transcription(stu, ref)
        typos = [d for d in res["diff"] if d["type"] == "typo"]
        assert len(typos) >= 1
        assert res["is_correct"] is False
        assert res["quality"] == 0.0

    def test_missing_and_extra_words(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        engine = ScoringEngine(data_manager=dm)
        ref = "Je dois partir maintenant."
        stu = "Je dois."
        res = engine.evaluate_transcription(stu, ref)
        missings = [d for d in res["diff"] if d["type"] == "missing"]
        assert len(missings) >= 1
        assert res["is_correct"] is False
        assert res["quality"] == 0.0

    def test_skipped_transcription(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        engine = ScoringEngine(data_manager=dm)
        ref = "Bonjour le monde."
        res = engine.evaluate_transcription("", ref, is_skipped=True)
        assert res["quality"] == 0.0
        assert res["composite_score"] == 0.0
        assert res["is_exact"] is False
        assert res["is_correct"] is False

    def test_multimode_scoring_isolation(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        engine = ScoringEngine(data_manager=dm)
        # Initial state for translation and transcription
        prof_trans = engine.get_profile(user_id="default_user", mode="translation")
        prof_audio = engine.get_profile(user_id="default_user", mode="transcription")
        assert prof_trans["score"] == 1.20
        assert prof_audio["score"] == 1.20

        # Complete batch in transcription mode with high quality
        items = [
            {"sentence_score": 1.0, "quality": 0.95, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.90, "is_skipped": False},
            {"sentence_score": 2.0, "quality": 0.92, "is_skipped": False},
        ]
        res = engine.calculate_score_adjustment(
            current_score=1.20,
            evaluated_items=items,
            mode="transcription",
            user_id="default_user"
        )
        assert res["delta"] > 0
        assert res["new_score"] > 1.20

        # Check that transcription mode score updated in SQLite
        updated_audio = engine.get_profile(user_id="default_user", mode="transcription")
        assert updated_audio["score"] == res["new_score"]
        assert updated_audio["streak"] == 1

        # Verify translation mode score remained at initial 1.20
        updated_trans = engine.get_profile(user_id="default_user", mode="translation")
        assert updated_trans["score"] == 1.20
        assert updated_trans["streak"] == 0

    def test_transcription_batch_excludes_new_words(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        engine = ScoringEngine(data_manager=dm)
        # Seed lexemas: 2 base words, 1 advanced unknown word
        l1 = dm.upsert_lexema(lemma="bonjour", pos="NOUN", level="A1", is_base=True)
        l2 = dm.upsert_lexema(lemma="monde", pos="NOUN", level="A1", is_base=True)
        l3 = dm.upsert_lexema(lemma="épistémologie", pos="NOUN", level="C2", is_base=False)

        # s1: familiar words only (0 new words)
        s1 = dm.add_sentence(es="Hola mundo", fr="Bonjour monde", level="A1", score=1.0)
        dm.link_sentence_lexemas(s1["id"], [l1, l2])

        # s2: has an unknown word (l3)
        s2 = dm.add_sentence(es="La epistemología", fr="L'épistémologie", level="C2", score=5.8)
        dm.link_sentence_lexemas(s2["id"], [l3])

        all_s = dm.get_all_sentences()
        batch = engine.get_batch_sentences(
            all_sentences=all_s,
            target_score=1.20,
            count=2,
            data_manager=dm,
            user_id="default_user",
            mode="transcription"
        )

        assert len(batch) > 0
        for s in batch:
            assert s.get("unknown_count", 0) == 0
            assert s.get("new_lexemas", []) == []
            assert s["id"] != s2["id"]

    def test_retry_scoring_adjustment(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        engine = ScoringEngine(data_manager=dm)
        # Initial complete batch
        items = [
            {"sentence_score": 1.0, "quality": 0.50, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.50, "is_skipped": False},
        ]
        res1 = engine.calculate_score_adjustment(
            current_score=1.20,
            evaluated_items=items,
            mode="transcription",
            user_id="default_user",
            is_retry=False
        )
        audio_prof1 = engine.get_profile(user_id="default_user", mode="transcription")
        initial_completed = audio_prof1["total_completed"]

        # Retry one sentence with 1.0 quality: recalculate with is_retry=True from base score
        items_retry = [
            {"sentence_score": 1.0, "quality": 1.00, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.50, "is_skipped": False},
        ]
        res2 = engine.calculate_score_adjustment(
            current_score=1.20,
            evaluated_items=items_retry,
            mode="transcription",
            user_id="default_user",
            is_retry=True
        )
        audio_prof2 = engine.get_profile(user_id="default_user", mode="transcription")

        # Score must be higher with retry quality improvement
        assert res2["new_score"] > res1["new_score"]
        # Total completed count should NOT be incremented again
        assert audio_prof2["total_completed"] == initial_completed

    def test_first_try_score_used_for_level(self, temp_db_path):
        dm = DataManager(db_path=temp_db_path)
        engine = ScoringEngine(data_manager=dm)
        # Sentence 1: first try had quality 0.40 (errors), retry achieved 1.00
        # Sentence 2: first try had quality 0.85
        # Per specification: the final score level calculation strictly uses the first try!
        first_try_items = [
            {"sentence_score": 1.0, "quality": 0.40, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.85, "is_skipped": False},
        ]
        res = engine.calculate_score_adjustment(
            current_score=1.20,
            evaluated_items=first_try_items,
            mode="transcription",
            user_id="default_user"
        )
        audio_prof = engine.get_profile(user_id="default_user", mode="transcription")
        assert audio_prof["score"] == res["new_score"]
        # Delta reflects the first-try quality (avg 0.62)
        assert res["avg_quality"] == 0.62
