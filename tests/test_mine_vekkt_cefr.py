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
    row3 = {"sentence": "Phrase de niveau avancé sélectionnée.", "difficulty": "C1"}
    row4 = {"sentence": "Phrase littéraire d'une complexité extrême.", "difficulty": "C2"}
    row5 = {"sentence": "Non.", "difficulty": "A1"}  # Too short (< 3 words)
    row6 = {"sentence": "Il y a 123 chiffres ici.", "difficulty": "A2"}  # Contains numbers
    return [row1, row2, row3, row4, row5, row6]


@patch("backend.ingestion.pipeline.mine_vekkt_cefr.load_dataset")
@patch("backend.ingestion.pipeline.mine_vekkt_cefr.concatenate_datasets")
def test_extract_sentences_from_vekkt_cefr(mock_concatenate, mock_load, mock_vekkt_dataset):
    mock_load.return_value = {"train": mock_vekkt_dataset, "validation": []}
    mock_concatenate.return_value = mock_vekkt_dataset

    # Extracts all valid sentences across all levels (A1-C2)
    extracted = extract_sentences_from_vekkt_cefr()

    assert len(extracted) == 4
    assert "C'est une phrase très simple." in extracted
    assert "Une autre phrase avec plus de mots." in extracted
    assert "Phrase de niveau avancé sélectionnée." in extracted
    assert "Phrase littéraire d'une complexité extrême." in extracted


@patch("backend.ingestion.pipeline.mine_vekkt_cefr.extract_sentences_from_vekkt_cefr")
def test_run_mine_vekkt_cefr(mock_extract):
    mock_extract.return_value = ["C'est une phrase très simple."]

    records = run_mine_vekkt_cefr(target_levels=["A1"], limit=5)

    mock_extract.assert_called_once_with(target_levels=["A1"], limit=5)
    assert records == [
        {
            "fr": "C'est une phrase très simple.",
            "es": None,
            "source": "hf_vekkt_cefr",
            "metadata": {"dataset": "vekkt/french_CEFR"},
        }
    ]


@patch("backend.ingestion.pipeline.mine_vekkt_cefr.extract_sentences_from_vekkt_cefr")
def test_run_mine_vekkt_cefr_empty(mock_extract):
    mock_extract.return_value = []

    records = run_mine_vekkt_cefr()

    mock_extract.assert_called_once_with(target_levels=None, limit=None)
    assert records == []
