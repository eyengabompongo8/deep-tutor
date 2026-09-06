import pytest
from models import MLEngine


@pytest.mark.ml
class TestMLEngine:
    @pytest.fixture(scope="class")
    def ml_engine(self):
        return MLEngine()

    def test_semantic_score_discrimination(self, ml_engine):
        src = "Tengo que irme ahora."
        ref = "Je dois partir maintenant."
        wrong_student = "Je mange de poisson."
        correct_student = "Je dois partir maintenant."

        res_wrong = ml_engine.compute_final_score(src, wrong_student, ref)
        assert res_wrong["s_sem"] < 50.0, f"Expected <50.0 for wrong translation, got {res_wrong['s_sem']}"

        res_correct = ml_engine.compute_final_score(src, correct_student, ref)
        assert res_correct["s_sem"] > 85.0, f"Expected >85.0 for correct translation, got {res_correct['s_sem']}"

    def test_word_mapping_and_subtoken_cleanup(self, ml_engine):
        src = "Tengo que irme ahora."
        ref = "Je dois partir maintenant."
        student = "Je dois partir maintenant."

        res = ml_engine.compute_final_score(src, student, ref)
        assert "mapping_data" in res
        src_words, tgt_words, word_attn = res["mapping_data"]

        assert "Tengo" in src_words and "irme" in src_words, "Subtokens not cleaned into real words!"
        assert "__es__" not in src_words and "</s>" not in src_words, "Special subtokens leaked into word selection!"
        assert word_attn is not None
