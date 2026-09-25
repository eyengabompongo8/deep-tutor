from backend.scoring import ScoringEngine, score_to_level
from backend.db import DataManager


class TestScoringEngine:
    def test_score_to_level(self):
        assert score_to_level(1.00) == 'A1'
        assert score_to_level(1.99) == 'A1'
        assert score_to_level(2.00) == 'A2'
        assert score_to_level(3.50) == 'B1'
        assert score_to_level(4.20) == 'B2'
        assert score_to_level(5.80) == 'C1'
        assert score_to_level(6.00) == 'C2'

    def test_batch_sampling(self, isolated_data_manager):
        engine = ScoringEngine(data_manager=isolated_data_manager)
        all_s = isolated_data_manager.get_all_sentences()
        batch = engine.get_batch_sentences(all_s, target_level='A1', count=3)
        assert len(batch) == 3
        for item in batch:
            assert 'id' in item
            assert 'es' in item
            assert 'fr' in item

    def test_score_progression_positive(self, isolated_data_manager):
        engine = ScoringEngine(data_manager=isolated_data_manager)
        items = [
            {"sentence_score": 1.0, "quality": 0.95, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.90, "is_skipped": False},
            {"sentence_score": 2.0, "quality": 0.85, "is_skipped": False},
        ]
        res = engine.calculate_score_adjustment(items)
        assert res["accuracy_rate"] == 1.0
        assert res["streak"] == 1
        assert res["correct_count"] == 3

    def test_score_progression_negative_skips(self, isolated_data_manager):
        engine = ScoringEngine(data_manager=isolated_data_manager)
        items = [
            {"sentence_score": 1.0, "quality": 0.0, "is_skipped": True},
            {"sentence_score": 1.0, "quality": 0.0, "is_skipped": True},
            {"sentence_score": 1.0, "quality": 0.0, "is_skipped": True},
        ]
        res = engine.calculate_score_adjustment(items)
        assert res["accuracy_rate"] == 0.0
        assert res["streak"] == 0
        assert res["correct_count"] == 0

    def test_in_memory_fallback_mode(self):
        engine = ScoringEngine(data_manager=None)
        items = [
            {"sentence_score": 1.0, "quality": 0.95, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.90, "is_skipped": False},
            {"sentence_score": 2.0, "quality": 0.85, "is_skipped": False},
        ]
        res = engine.calculate_score_adjustment(items, mode="translation")
        assert res["accuracy_rate"] == 1.0
        assert res["streak"] == 1
        assert engine.profile["modes"]["translation"]["streak"] == 1
        assert engine.profile["modes"]["translation"]["total_completed"] == 3

    def test_empty_batch_returns_current_profile(self, isolated_data_manager):
        engine = ScoringEngine(data_manager=isolated_data_manager)
        res = engine.calculate_score_adjustment([])
        assert res["accuracy_rate"] == 0.0
        assert "profile" in res
