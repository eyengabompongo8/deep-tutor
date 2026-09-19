import time
import pytest
import numpy as np
from unittest.mock import MagicMock
from backend.eager_processor import EagerPrecomputeManager, get_aligned_targets


class TestEagerProcessor:
    @pytest.fixture(autouse=True)
    def setup_processor(self):
        self.mock_engine = MagicMock()
        
        src_words = ["hola", "mundo"]
        ref_words = ["bonjour", "le", "monde"]
        # Matrix shape: (num_target_words, num_source_words) -> (3, 2)
        matrix = np.array([
            [0.9, 0.0],
            [0.1, 0.0],
            [0.0, 0.95]
        ])
        self.mock_engine.evaluate_nmt_and_map.return_value = (
            5.0,
            (src_words, ref_words, matrix),
            [],
            []
        )
        
        self.mock_engine.evaluate_fluency.return_value = (4.2, [], [])
        self.mock_engine.encode_semantics.return_value = np.array([0.1, 0.2, 0.3])
        
        self.mock_engine.evaluate_student_against_bundle.return_value = {
            "is_french": True,
            "lid_prob": 1.0,
            "s_sem": 95.0,
            "s_nmt": 90.0,
            "s_fluency": 92.0,
            "mapping_data": (src_words, ref_words, matrix),
            "mapping_data_ref": (src_words, ref_words, matrix),
            "nmt_ppl": 5.2,
            "fluency_ppl": 4.5,
            "raw_cos": 0.95,
            "syntactical_errors": [],
            "nmt_token_breakdown": [],
            "nmt_word_breakdown": [],
            "fluency_token_breakdown": [],
            "fluency_word_breakdown": []
        }
        
        self.manager = EagerPrecomputeManager(
            get_engine_fn=lambda: self.mock_engine,
            data_manager=None,
            max_cache_size=10
        )

    def test_get_aligned_targets(self):
        matrix = np.array([
            [0.9, 0.0],
            [0.1, 0.0],
            [0.0, 0.95]
        ])
        aligned = get_aligned_targets([0], matrix, 3)
        assert aligned == [0]
        
        aligned_world = get_aligned_targets([1], matrix, 3)
        assert aligned_world == [2]

    def test_enqueue_and_get_hint(self):
        batch = [{
            "id": 101,
            "es": "hola mundo",
            "fr": "bonjour le monde"
        }]
        
        self.manager.enqueue_batch(batch)
        
        hint = self.manager.get_hint(
            source_text="hola mundo",
            reference_text="bonjour le monde",
            selected_indices=[0]
        )
        
        assert "src_words" in hint
        assert "tgt_words" in hint
        assert hint["src_words"] == ["hola", "mundo"]
        assert hint["tgt_words"] == ["bonjour", "le", "monde"]
        assert hint["mapped_target_words"] == ["bonjour"]

    def test_instant_cached_hint(self):
        # Prime cache
        _ = self.manager.get_or_compute_bundle(
            source_text="hola",
            reference_text="bonjour"
        )
        
        start = time.perf_counter()
        hint = self.manager.get_hint(
            source_text="hola",
            reference_text="bonjour",
            selected_indices=[]
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        
        # Sub-millisecond lookup from memory
        assert elapsed_ms < 50.0
        assert hint["src_words"] == ["hola", "mundo"]

    def test_evaluate_student_reusing_bundle(self):
        res = self.manager.evaluate_student(
            source_text="hola mundo",
            student_text="bonjour le monde",
            reference_text="bonjour le monde",
            sentence_id=101
        )
        
        assert res["is_french"] is True
        assert res["s_sem"] == 95.0
        self.mock_engine.evaluate_student_against_bundle.assert_called_once()
