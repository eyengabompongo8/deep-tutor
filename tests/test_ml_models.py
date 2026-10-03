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

    def test_check_language_validation_and_rejection(self, ml_engine):
        # Empty or whitespace
        is_fr, prob = ml_engine.check_language("")
        assert not is_fr
        assert prob == 0.0

        is_fr, prob = ml_engine.check_language("   \n  ")
        assert not is_fr
        assert prob == 0.0

        # Valid French short sentences
        for sentence in ["Merci", "C'est bon", "Bonjour", "Je suis ici", "un deux trois"]:
            is_fr, prob = ml_engine.check_language(sentence)
            assert is_fr, f"Expected '{sentence}' to be recognized as French, got {is_fr} (prob={prob})"
            assert prob >= 0.40

        # Non-French sentences that previously bypassed LID because of overlapping words
        # (e.g. English 'on', Spanish 'en' / 'la' / 'a')
        non_french_samples = [
            "He is on the bus.",
            "El libro está en la mesa.",
            "The dog is brown.",
            "Hola que tal amigo.",
        ]
        for sentence in non_french_samples:
            is_fr, prob = ml_engine.check_language(sentence)
            assert not is_fr, f"Expected non-French '{sentence}' to be rejected, but got is_fr={is_fr} (prob={prob})"


