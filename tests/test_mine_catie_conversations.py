import pytest
from unittest.mock import patch, MagicMock

from backend.ingestion.pipeline.mine_catie_conversations import (
    extract_sentences_from_conversations,
    run_mine_catie_conversations,
)


@pytest.fixture
def mock_conversations_dataset():
    # Multi-turn conversation with multi-sentence message
    conv1 = {
        "messages": [
            {"role": "user", "content": "Salut !"},  # generic short greeting -> dropped
            {
                "role": "assistant",
                "content": "Bonjour ! Comment puis-je vous aider aujourd'hui ? J'aime beaucoup cette ville.",
            },  # 1st is generic greeting, 2nd is assistant greeting, 3rd is normal sentence
            {
                "role": "user",
                "content": "Je vais au centre commercial ce week-end. Avez-vous des conseils ?",
            },
            {
                "role": "assistant",
                "content": "Buvez dix onces d'eau.",  # non-metric anomaly -> dropped
            },
        ]
    }
    return [conv1]


@patch("backend.ingestion.pipeline.mine_catie_conversations.load_dataset")
def test_extract_sentences_from_conversations(mock_load, mock_conversations_dataset):
    mock_load.return_value = mock_conversations_dataset

    extracted = extract_sentences_from_conversations(drop_generic_greetings=True)

    # "Salut !" and boilerplate greetings should be dropped
    assert "Salut !" not in extracted
    assert "Bonjour ! Comment puis-je vous aider aujourd'hui ?" not in extracted
    assert "Buvez dix onces d'eau." not in extracted

    # Clean sentences from turns should be extracted and split
    assert "J'aime beaucoup cette ville." in extracted
    assert "Je vais au centre commercial ce week-end." in extracted
    assert "Avez-vous des conseils ?" in extracted


@patch("backend.ingestion.pipeline.mine_catie_conversations.extract_sentences_from_conversations")
def test_run_mine_catie_conversations(mock_extract):
    mock_extract.return_value = ["J'aime beaucoup cette ville."]

    records = run_mine_catie_conversations(limit=1)

    assert len(records) == 1
    assert records[0] == {
        "fr": "J'aime beaucoup cette ville.",
        "es": None,
        "source": "hf_catie_conversations",
        "metadata": {"dataset": "CATIE-AQ/everyday-conversations-llama3.1-2k-in-french"},
    }
    mock_extract.assert_called_once_with(limit=1)
