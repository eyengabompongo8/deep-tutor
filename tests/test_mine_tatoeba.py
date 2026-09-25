import pytest
from unittest.mock import patch, MagicMock

from backend.ingestion.pipeline.mine_tatoeba import (
    clean_sentence,
    is_valid_sentence,
    get_sentence_count,
    mine_tatoeba_parallel,
    mine_tatoeba_monolingual,
    run_mine_tatoeba,
)
from backend.ingestion.pipeline.common import clean_quotes_and_symbols, format_sentence, get_spacy_nlp


def test_clean_quotes_and_symbols():
    # Quotes and brackets
    assert clean_quotes_and_symbols("« Bonjour le monde ! »") == "Bonjour le monde !"
    assert clean_quotes_and_symbols("<< C'est magnifique ! >>") == "C'est magnifique !"
    assert clean_quotes_and_symbols('“Je vais bien.”') == "Je vais bien."
    assert clean_quotes_and_symbols('«<Salut tout le monde>»') == "Salut tout le monde"


def test_format_sentence_with_tatoeba_artifacts():
    raw = "  « Bonjour ! Comment vas-tu ? »  "
    cleaned = format_sentence(raw)
    assert "«" not in cleaned
    assert "»" not in cleaned
    assert cleaned == "Bonjour ! Comment vas-tu ?"


def test_multi_sentence_parallel_discard():
    mock_engine = MagicMock()
    mock_engine.score_sentences_ml_batch.return_value = [{"level": "A1", "score": 1.0}]
    mock_engine.score_sentence.return_value = ("A1", 0.5, {})

    # Mock file reading returning a pair with 2 sentences in French vs 1 in Spanish
    mock_pairs = [
        # Multi-sentence French -> should be discarded
        ("Hola. ¿Cómo estás?", "Bonjour. Comment vas-tu ?"),
        # Clean single sentence -> should be kept
        ("Buenos días a todos.", "Bonjour à tous.")
    ]

    # Preload spacy to prevent builtins.open interception during spacy load
    get_spacy_nlp()

    with patch("os.path.exists", return_value=True), \
         patch("os.makedirs") as mock_makedirs, \
         patch("backend.ingestion.pipeline.mine_tatoeba.open") as mock_open, \
         patch("backend.ingestion.pipeline.mine_tatoeba.VocabEngine", return_value=mock_engine):
        
        # Mock file lines
        cm_es = MagicMock()
        cm_es.__enter__.return_value.readlines.return_value = [p[0] + "\n" for p in mock_pairs]
        cm_fr = MagicMock()
        cm_fr.__enter__.return_value.readlines.return_value = [p[1] + "\n" for p in mock_pairs]
        
        mock_open.side_effect = [cm_es, cm_fr]

        records = mine_tatoeba_parallel()

        # Only the clean single sentence should remain
        assert len(records) == 1
        assert records[0]["fr"] == "Bonjour à tous."
        assert records[0]["es"] == "Buenos días a todos."
        assert records[0]["source"] == "tatoeba_parallel"


@patch("backend.ingestion.pipeline.mine_tatoeba.process_and_score_sentences")
@patch("backend.ingestion.pipeline.mine_tatoeba.deduplicate_against_db")
@patch("backend.ingestion.pipeline.mine_tatoeba._download_file")
@patch("bz2.open")
def test_monolingual_multi_sentence_split(
    mock_bz2_open,
    mock_download,
    mock_dedup,
    mock_process_and_score
):
    # A single record containing two sentences
    file_content = "1\tfra\tBonjour tout le monde. Il fait très beau aujourd'hui.\n"
    mock_bz2_open.return_value.__enter__.return_value = file_content.splitlines(keepends=True)
    
    mock_dedup.side_effect = lambda sents: sents
    mock_process_and_score.return_value = [
        {"fr": "Bonjour tout le monde.", "es": "Hola a todos.", "level": "A1", "score": 1.0, "score_lexical": 0.5, "score_ml": 1.0, "source": "tatoeba_monolingual"},
        {"fr": "Il fait très beau aujourd'hui.", "es": "Hace muy buen tiempo hoy.", "level": "A2", "score": 1.5, "score_lexical": 0.8, "score_ml": 1.5, "source": "tatoeba_monolingual"}
    ]

    with patch("os.path.exists", return_value=True):
        records = mine_tatoeba_monolingual()

    assert len(records) == 2
    assert records[0]["fr"] == "Bonjour tout le monde."
    assert records[1]["fr"] == "Il fait très beau aujourd'hui."
    assert records[0]["source"] == "tatoeba_monolingual"


@patch("backend.ingestion.pipeline.mine_tatoeba.mine_tatoeba_parallel")
@patch("backend.ingestion.pipeline.mine_tatoeba.mine_tatoeba_monolingual")
def test_run_mine_tatoeba_modes(mock_mono, mock_par):
    mock_par.return_value = [{"fr": "par_fr", "source": "tatoeba_parallel"}]
    mock_mono.return_value = [{"fr": "mono_fr", "source": "tatoeba_monolingual"}]

    # Test parallel mode
    res_par = run_mine_tatoeba(mode="parallel")
    assert len(res_par) == 1
    assert res_par[0]["source"] == "tatoeba_parallel"
    mock_par.assert_called_once()
    mock_mono.assert_not_called()

    mock_par.reset_mock()
    mock_mono.reset_mock()

    # Test monolingual mode
    res_mono = run_mine_tatoeba(mode="monolingual")
    assert len(res_mono) == 1
    assert res_mono[0]["source"] == "tatoeba_monolingual"
    mock_mono.assert_called_once()
    mock_par.assert_not_called()

    mock_par.reset_mock()
    mock_mono.reset_mock()

    # Test both mode
    res_both = run_mine_tatoeba(mode="both")
    assert len(res_both) == 2
    mock_par.assert_called_once()
    mock_mono.assert_called_once()
