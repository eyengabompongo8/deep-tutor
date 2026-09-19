from backend.scoring_engine import ScoringEngine, score_to_level
from backend.data_manager import DataManager


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
        batch = engine.get_batch_sentences(all_s, target_score=1.5, count=3)
        assert len(batch) == 3
        for item in batch:
            assert 'id' in item
            assert 'es' in item
            assert 'fr' in item
            assert 'score' in item

    def test_score_progression_positive(self, isolated_data_manager):
        engine = ScoringEngine(data_manager=isolated_data_manager)
        items = [
            {"sentence_score": 1.0, "quality": 0.95, "is_skipped": False},
            {"sentence_score": 1.0, "quality": 0.90, "is_skipped": False},
            {"sentence_score": 2.0, "quality": 0.85, "is_skipped": False},
        ]
        res = engine.calculate_score_adjustment(1.20, items)
        assert res["delta"] > 0
        assert res["new_score"] > 1.20
        assert res["new_score"] == engine.profile["score"]

    def test_score_progression_negative_skips(self, isolated_data_manager):
        engine = ScoringEngine(data_manager=isolated_data_manager)
        items = [
            {"sentence_score": 1.0, "quality": 0.0, "is_skipped": True},
            {"sentence_score": 1.0, "quality": 0.0, "is_skipped": True},
            {"sentence_score": 1.0, "quality": 0.0, "is_skipped": True},
        ]
        res = engine.calculate_score_adjustment(2.00, items)
        assert res["delta"] < 0
        assert res["new_score"] < 2.00
