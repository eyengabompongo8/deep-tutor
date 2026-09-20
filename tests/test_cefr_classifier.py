import pytest
import torch
from backend.model_registry import registry, UniversalCEFRModel, CEFRClassifierModelWrapper
from backend.vocab_engine import VocabEngine


class TestCEFRClassifierUnit:
    """Fast unit tests using mocks / wrapper logic."""
    def test_registry_registration(self):
        assert "universal_cefr" in registry._cefr_models
        assert issubclass(registry._cefr_models["universal_cefr"], CEFRClassifierModelWrapper)

    def test_wrapper_prediction_structure(self, mocker):
        # Mock tokenizer and model
        mock_tokenizer = mocker.MagicMock()
        mock_tokenizer.return_value = mocker.MagicMock(to=mocker.MagicMock(return_value={"input_ids": torch.tensor([[1, 2, 3]])}))
        
        mock_model = mocker.MagicMock()
        mock_model.to.return_value = mock_model
        # Logits for 6 classes: [10.0, 1.0, 0.0, 0.0, 0.0, 0.0] -> high probability for A1 (index 0)
        mock_model.return_value.logits = torch.tensor([[10.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        mock_model.config.id2label = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "C1", 5: "C2"}

        mocker.patch("backend.model_registry.AutoTokenizer.from_pretrained", return_value=mock_tokenizer)
        mocker.patch("backend.model_registry.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model)

        wrapper = UniversalCEFRModel(device="cpu", dtype=torch.float32)
        res = wrapper.predict(["Bonjour tout le monde."])

        assert len(res) == 1
        assert res[0]["level"] == "A1"
        assert 1.0 <= res[0]["score"] <= 1.5
        assert "probabilities" in res[0]
        assert "A1" in res[0]["probabilities"]
        assert res[0]["probabilities"]["A1"] > 0.90

    def test_vocab_engine_ml_and_hybrid_scoring_mocked(self, mocker):
        engine = VocabEngine()
        mocker.patch.object(
            engine,
            "score_sentence_ml",
            return_value=("B1", 3.25, {"A1": 0.05, "A2": 0.15, "B1": 0.70, "B2": 0.08, "C1": 0.01, "C2": 0.01})
        )

        ml_lvl, ml_sc, lex_sc, breakdown = engine.score_sentence_hybrid("Bienvenue dans notre maison.")
        assert ml_lvl == "B1"
        assert ml_sc == 3.25
        assert isinstance(lex_sc, float)
        assert isinstance(breakdown, list)


@pytest.mark.ml
class TestCEFRClassifierLive:
    """Live ML tests executing actual model weights."""
    @pytest.fixture(scope="class")
    def cefr_model(self):
        return registry.get_cefr("universal_cefr", device="cpu", dtype=torch.float32)

    def test_a1_sentence_classification(self, cefr_model):
        results = cefr_model.predict(["Bonjour, comment tu t'appelles ?", "Où est le restaurant ?"])
        assert len(results) == 2
        for r in results:
            assert r["level"] in {"A1", "A2"}
            assert r["score"] < 2.5

    def test_c1_c2_sentence_classification(self, cefr_model):
        complex_text = "L'herméneutique post-kantienne récuse l'ontologie substantielle au profit d'une dialectique transcendantale."
        results = cefr_model.predict([complex_text])
        assert len(results) == 1
        assert results[0]["level"] in {"B2", "C1", "C2"}
        assert results[0]["score"] >= 3.5

    def test_vocab_engine_batch_integration(self):
        engine = VocabEngine()
        texts = [
            "Bonjour !",
            "Je voudrais acheter une baguette.",
            "Bien que les circonstances fussent défavorables, il persévéra avec véhémence."
        ]
        results = engine.score_sentences_ml_batch(texts, batch_size=2)
        assert len(results) == 3
        assert results[0]["level"] == "A1"
        assert results[2]["score"] > results[0]["score"]
