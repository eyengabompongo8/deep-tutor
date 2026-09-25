import pytest
from unittest.mock import patch, MagicMock

from backend.ingestion.pipeline.mine_vekkt_cefr import (
    extract_sentences_from_vekkt_cefr,
    run_mine_vekkt_cefr,
)


@pytest.fixture
def mock_vekkt_dataset():
    row1 = {"sentence": "C'est une phrase très simple.", "difficulty": "A1"}
    row2 = {"sentence": "Une autre phrase avec plus de mots.", "difficulty": "B1"}
    row3 = {"sentence": "Phrase de niveau avancé non sélectionnée.", "difficulty": "C1"}
    row4 = {"sentence": "Non.", "difficulty": "A1"}  # Too short (< 3 words)
    row5 = {"sentence": "Il y a 123 chiffres ici.", "difficulty": "A2"}  # Contains numbers
    return [row1, row2, row3, row4, row5]


@patch("backend.ingestion.pipeline.mine_vekkt_cefr.load_dataset")
@patch("backend.ingestion.pipeline.mine_vekkt_cefr.concatenate_datasets")
def test_extract_sentences_from_vekkt_cefr(mock_concatenate, mock_load, mock_vekkt_dataset):
    mock_load.return_value = {"train": mock_vekkt_dataset, "validation": []}
    mock_concatenate.return_value = mock_vekkt_dataset

    # Request only A1, B1
    extracted = extract_sentences_from_vekkt_cefr(target_levels=["A1", "B1"])

    assert len(extracted) == 2
    assert "C'est une phrase très simple." in extracted
    assert "Une autre phrase avec plus de mots." in extracted
    assert "Phrase de niveau avancé non sélectionnée." not in extracted


@patch("backend.ingestion.pipeline.mine_vekkt_cefr.extract_sentences_from_vekkt_cefr")
@patch("backend.ingestion.pipeline.mine_vekkt_cefr.deduplicate_against_db")
@patch("backend.ingestion.pipeline.mine_vekkt_cefr.process_and_score_sentences")
def test_run_mine_vekkt_cefr(mock_process, mock_dedup, mock_extract):
    mock_extract.return_value = ["C'est une phrase très simple."]
    mock_dedup.return_value = ["C'est une phrase très simple."]
    mock_process.return_value = [
        {
            "fr": "C'est une phrase très simple.",
            "es": "Es una frase muy simple.",
            "level": "A1",
            "score": 1.2,
            "score_lexical": 1.0,
            "score_ml": 1.2,
            "source": "hf_vekkt_cefr",
            "metadata": {"dataset": "vekkt/french_CEFR"},
        }
    ]

    records = run_mine_vekkt_cefr(target_levels=["A1"])

    assert len(records) == 1
    assert records[0]["fr"] == "C'est une phrase très simple."
    assert records[0]["source"] == "hf_vekkt_cefr"
    mock_dedup.assert_called_once_with(["C'est une phrase très simple."])
    mock_process.assert_called_once_with(
        ["C'est une phrase très simple."],
        source_name="hf_vekkt_cefr",
        source_metadata={"dataset": "vekkt/french_CEFR"},
    )
