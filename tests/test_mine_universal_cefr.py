import pytest
from unittest.mock import patch, MagicMock

from backend.ingestion.pipeline.mine_universal_cefr import run_mine_universal_cefr

@pytest.fixture
def mock_dataset():
    # Mock row matches expected structure from HuggingFace dataset
    row1 = {"text": "C'est une phrase très simple."}
    row2 = {"text": "Une autre phrase avec plus de mots."}
    row3 = {"text": "This is not french but valid by length and punctuation."} # Valid structure
    row4 = {"text": "A"} # Too short, invalid
    row5 = {"text": "Une autre phrase avec plus de mots."} # In-batch duplicate
    
    mock_split = [row1, row2, row3, row4, row5]
    return mock_split

@patch("backend.ingestion.pipeline.mine_universal_cefr.load_dataset")
@patch("backend.ingestion.pipeline.mine_universal_cefr.concatenate_datasets")
def test_run_mine_universal_cefr(
    mock_concatenate_datasets,
    mock_load_dataset,
    mock_dataset
):
    mock_load_dataset.return_value = {"train": mock_dataset, "test": []}
    mock_concatenate_datasets.return_value = mock_dataset
    
    results = run_mine_universal_cefr()
    
    assert mock_load_dataset.called
    assert mock_concatenate_datasets.called
    
    # row4 is too short ("A"), row5 is a duplicate of row2
    # Valid unique rows: row1, row2, row3
    assert len(results) == 3
    assert results[0]["fr"] == "C'est une phrase très simple."
    assert results[0]["es"] is None
    assert results[0]["source"] == "hf_universal_cefr"
    assert results[0]["metadata"] == {}
    
    assert results[2]["fr"] == "This is not french but valid by length and punctuation."
    assert results[2]["es"] is None
    assert results[2]["metadata"] == {}


@patch("backend.ingestion.pipeline.mine_universal_cefr.load_dataset")
@patch("backend.ingestion.pipeline.mine_universal_cefr.concatenate_datasets")
def test_run_mine_universal_cefr_with_limit(
    mock_concatenate_datasets,
    mock_load_dataset,
    mock_dataset
):
    mock_load_dataset.return_value = {"train": mock_dataset}
    mock_concatenate_datasets.return_value = mock_dataset
    
    results = run_mine_universal_cefr(limit=1)
    
    assert len(results) == 1
    assert results[0]["fr"] == "C'est une phrase très simple."
    assert results[0]["es"] is None
    assert results[0]["source"] == "hf_universal_cefr"

