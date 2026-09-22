import pytest
from backend.ml.engine import MLEngine


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

    def test_nmt_french_target_decoding(self, ml_engine):
        src = "Él es sólo un soñador."
        ref = "Ce n'est qu'un rêveur."
        student = "Il est seulement un rêveur."

        res = ml_engine.compute_final_score(src, student, ref)
        nmt_word_bd = res.get("nmt_word_breakdown", [])
        
        # Verify that French words like 'Il' are not flagged with Spanish 'É' predictions
        for item in nmt_word_bd:
            if item["word"] == "Il":
                assert "É" not in item["top_prediction"], f"Spanish prediction leaked into French evaluation: {item}"
                assert not item["is_error"], "Valid French subject pronoun 'Il' incorrectly flagged as error"
            if item["word"] == "rêveur":
                assert not item["is_error"], "Reference word 'rêveur' must not be flagged as error"

    def test_unsound_word_expected_suggestion(self, ml_engine):
        src = "Tengo un bolígrafo."
        ref = "J'ai un stylo."
        student = "un style est ce que j'ai."

        res = ml_engine.compute_final_score(src, student, ref)
        nmt_word_bd = res.get("nmt_word_breakdown", [])

        # The unsound word 'style' (typed instead of 'stylo' for 'bolígrafo')
        # should have expected reference suggestion 'stylo', never 'bul'
        found_style = False
        for item in nmt_word_bd:
            if item["word"].lower() == "style":
                found_style = True
                assert item["top_prediction"] == "stylo", f"Expected suggestion 'stylo' but got '{item['top_prediction']}'"
                assert item["is_error"], "Unsound word 'style' should be flagged as error"
            elif item["word"].lower() in {"un", "est", "ce", "que", "j'", "ai"}:
                # Functional or reference words must not be flagged as unexpected words
                assert not item["is_error"], f"Grammar/function word '{item['word']}' should not be flagged as error"

        assert found_style, "'style' not found in nmt_word_breakdown"

    def test_unsound_word_velos_for_voitures(self, ml_engine):
        src = "Tengo dos automóviles."
        ref = "J'ai deux voitures."
        student = "J'ai deux vélos."

        res = ml_engine.compute_final_score(src, student, ref)
        nmt_word_bd = res.get("nmt_word_breakdown", [])

        found_velos = False
        for item in nmt_word_bd:
            clean_w = item["word"].lower().strip(".,!?:;'\"-")
            if clean_w == "vélos":
                found_velos = True
                assert item["is_error"], "Unsound word 'vélos' should be flagged as error"
                assert "voitures" in item["top_prediction"].lower(), f"Expected 'voitures', got '{item['top_prediction']}'"
            elif clean_w in {"j'", "ai", "deux"}:
                assert not item["is_error"], f"Correct word '{item['word']}' should not be flagged"

        assert found_velos, "'vélos' not found in nmt_word_breakdown"


