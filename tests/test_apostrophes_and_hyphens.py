import pytest
from backend.nlp.text_utils import (
    normalize_french_typography,
    autocorrect_text,
)
from backend.ml.token_utils import aggregate_sentencepiece_tokens_to_words
from backend.scoring.transcription import evaluate_transcription
from backend.nlp.vocab_engine import VocabEngine, get_vocab_engine


class TestFrenchTypographyAndPunctuation:
    def test_normalize_quotes_and_apostrophes(self):
        # Curly apostrophes, left/right single quotes, backticks, primes
        raw = "C’est l‘homme d`un ami et l´enfant d′hier"
        normalized = normalize_french_typography(raw)
        assert normalized == "C'est l'homme d'un ami et l'enfant d'hier"

    def test_normalize_dashes_and_hyphens(self):
        # Non-breaking hyphen, en-dash, em-dash, minus sign
        raw = "peut‑être va–t—il venir − aujourd'hui"
        normalized = normalize_french_typography(raw)
        assert normalized == "peut-être va-t-il venir - aujourd'hui"

    def test_normalize_non_breaking_spaces(self):
        raw = "Il\u00a0le\u202fvoyait\u2009!"
        normalized = normalize_french_typography(raw)
        assert normalized == "Il le voyait !"

    def test_high_punctuation_spacing_symmetric(self):
        assert normalize_french_typography("Il le voyait!") == "Il le voyait !"
        assert normalize_french_typography("Comment allez-vous?") == "Comment allez-vous ?"
        assert normalize_french_typography("Attention: danger;") == "Attention : danger ;"
        # Multiple spaces before punctuation collapsed
        assert normalize_french_typography("Vraiment   !") == "Vraiment !"

    def test_autocorrect_symmetric_punctuation_and_quotes(self):
        stu = "C’est magnifique !"
        ref = "C'est magnifique!"
        corrected, msgs = autocorrect_text(stu, ref)
        assert corrected == "C'est magnifique !"
        assert normalize_french_typography(ref) == "C'est magnifique !"


class TestSentencePieceAggregationApostrophesAndHyphens:
    def test_token_aggregation_without_artificial_splits(self):
        # Simulate SentencePiece tokens from MarianMT for: "C'est aujourd'hui l'homme"
        tokens = ['\u2581C', "'", 'est', '\u2581a', 'uj', 'our', 'd', "'", 'hu', 'i', '\u2581l', "'", 'hom', 'me']
        words, groups = aggregate_sentencepiece_tokens_to_words(tokens)
        assert words == ["C'est", "aujourd'hui", "l'homme"]
        assert len(groups) == 3

    def test_token_aggregation_hyphenated_words(self):
        # Simulate SentencePiece tokens for: "Peut-être va-t-il venir ?"
        tokens = ['\u2581Pe', 'ut', '-', 'ê', 'tre', '\u2581va', '-', 't', '-', 'il', '\u2581venir', '\u2581?']
        words, groups = aggregate_sentencepiece_tokens_to_words(tokens)
        assert words == ["Peut-être", "va-t-il", "venir", "?"]
        assert len(groups) == 4


class TestTranscriptionPunctuationTolerance:
    def test_transcription_exact_match_with_apostrophe_and_hyphen(self):
        ref = "C'est aujourd'hui peut-être son tour !"
        stu = "c’est aujourd’hui peut‑être son tour"
        res = evaluate_transcription(stu, ref)
        assert res["is_exact"] is True
        assert res["quality"] >= 0.98
        assert all(d["type"] == "exact" for d in res["diff"])

    def test_transcription_preserves_hyphenated_compound_words(self):
        ref = "Rendez-vous là-bas avec grand-mère."
        stu = "Rendez-vous là-bas avec grand-mère."
        res = evaluate_transcription(stu, ref)
        diff_words = [d["word"] for d in res["diff"]]
        assert "Rendez-vous" in diff_words
        assert "là-bas" in diff_words
        assert "grand-mère" in diff_words


class TestVocabularyElisionMatching:
    def test_extract_sentence_token_pairs_elisions(self):
        engine = get_vocab_engine()
        pairs = engine.extract_sentence_token_pairs(
            "Qu'il vienne jusqu'à la maison d'accord.",
            ["accord", "maison"]
        )
        assert any(p["lemma"] == "accord" for p in pairs)
        assert any(p["lemma"] == "maison" for p in pairs)
