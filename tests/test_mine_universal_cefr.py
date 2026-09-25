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
    row5 = {"text": "Duplicate sentence."} # Duplicate
    
    mock_split = [row1, row2, row3, row4, row5]
    return mock_split

@patch("backend.ingestion.pipeline.mine_universal_cefr.load_dataset")
@patch("backend.ingestion.pipeline.mine_universal_cefr.concatenate_datasets")
@patch("backend.ingestion.pipeline.mine_universal_cefr.DataManager")
@patch("backend.ingestion.pipeline.mine_universal_cefr.VocabEngine")
@patch("backend.ingestion.pipeline.mine_universal_cefr.translate_batch")
def test_run_mine_universal_cefr(
    mock_translate_batch,
    mock_vocab_engine_class,
    mock_data_manager_class,
    mock_concatenate_datasets,
    mock_load_dataset,
    mock_dataset
):
    # Setup mock Dataset
    mock_load_dataset.return_value = {"train": mock_dataset, "test": []}
    mock_concatenate_datasets.return_value = mock_dataset
    
    # Setup mock DataManager
    mock_dm_instance = MagicMock()
    # "duplicate sentence." already in db
    mock_dm_instance.get_all_sentences.return_value = [{"fr": "Duplicate sentence."}]
    mock_data_manager_class.return_value = mock_dm_instance
    
    # Setup mock VocabEngine
    mock_engine_instance = MagicMock()
    mock_engine_instance.score_sentences_ml_batch.return_value = [
        {"level": "A1", "score": 1.2},
        {"level": "A2", "score": 2.1},
        {"level": "B1", "score": 3.0}
    ]
    mock_engine_instance.score_sentence.side_effect = [
        ("A1", 1.0, {}),
        ("A2", 2.0, {}),
        ("B1", 3.0, {})
    ]
    mock_vocab_engine_class.return_value = mock_engine_instance
    
    # Setup mock translation
    mock_translate_batch.return_value = [
        "Es una frase muy simple.",
        "Otra frase con mas palabras.",
        "Esto no es frances."
    ]
    
    # Run pipeline
    results = run_mine_universal_cefr()
    
    # Assertions
    assert mock_load_dataset.called
    assert mock_concatenate_datasets.called
    assert mock_translate_batch.called
    
    # Expected results
    assert len(results) == 3
    assert results[0]["fr"] == "C'est une phrase très simple."
    assert results[0]["es"] == "Es una frase muy simple."
    assert results[0]["level"] == "A1"
    assert results[0]["score"] == 1.2
    assert results[0]["score_lexical"] == 1.0
    assert results[0]["source"] == "hf_universal_cefr"
    
    assert results[2]["fr"] == "This is not french but valid by length and punctuation."


@patch("backend.ingestion.pipeline.mine_universal_cefr.load_dataset")
@patch("backend.ingestion.pipeline.mine_universal_cefr.concatenate_datasets")
@patch("backend.ingestion.pipeline.mine_universal_cefr.DataManager")
@patch("backend.ingestion.pipeline.mine_universal_cefr.VocabEngine")
@patch("backend.ingestion.pipeline.mine_universal_cefr.translate_batch")
def test_run_mine_universal_cefr_with_limit(
    mock_translate_batch,
    mock_vocab_engine_class,
    mock_data_manager_class,
    mock_concatenate_datasets,
    mock_load_dataset,
    mock_dataset
):
    mock_load_dataset.return_value = {"train": mock_dataset}
    mock_concatenate_datasets.return_value = mock_dataset
    
    mock_dm_instance = MagicMock()
    mock_dm_instance.get_all_sentences.return_value = []
    mock_data_manager_class.return_value = mock_dm_instance
    
    mock_engine_instance = MagicMock()
    mock_engine_instance.score_sentences_ml_batch.return_value = [
        {"level": "A1", "score": 1.2}
    ]
    mock_engine_instance.score_sentence.side_effect = [
        ("A1", 1.0, {})
    ]
    mock_vocab_engine_class.return_value = mock_engine_instance
    
    mock_translate_batch.return_value = ["Es una frase muy simple."]
    
    results = run_mine_universal_cefr(limit=1)
    
    assert len(results) == 1
    assert results[0]["fr"] == "C'est une phrase très simple."
