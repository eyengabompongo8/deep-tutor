import pytest
from unittest.mock import patch
import sys

from backend.ingestion.orchestrator import main


def test_orchestrator_help_displays_mine_universal_cefr(capsys):
    with patch.object(sys, "argv", ["orchestrator.py", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--mine-universal-cefr" in captured.out
        assert "--universal-cefr-limit" in captured.out
        assert "--mine-hf" not in captured.out


def test_orchestrator_rejects_mine_hf(capsys):
    with patch.object(sys, "argv", ["orchestrator.py", "--mine-hf"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "unrecognized arguments: --mine-hf" in captured.err


@patch("backend.ingestion.orchestrator.run_mine_universal_cefr")
def test_orchestrator_runs_mine_universal_cefr(mock_run_cefr):
    mock_run_cefr.return_value = []
    with patch.object(sys, "argv", ["orchestrator.py", "--mine-universal-cefr", "--universal-cefr-limit", "42"]):
        main()
        mock_run_cefr.assert_called_once_with(limit=42)


@patch("backend.ingestion.orchestrator.run_ingest_and_link")
@patch("backend.ingestion.pipeline.preprocess_and_score_candidates")
@patch("backend.ingestion.pipeline.deduplicate_candidate_records")
@patch("backend.ingestion.orchestrator.run_mine_universal_cefr")
@patch("backend.ingestion.orchestrator.run_mine_tatoeba")
def test_orchestrator_mines_joins_deduplicates_scores_translates_and_ingests(
    mock_mine_tatoeba,
    mock_mine_universal_cefr,
    mock_dedup,
    mock_preprocess,
    mock_ingest
):
    # Tatoeba returns pairs with Spanish translations
    mock_mine_tatoeba.return_value = [
        {"fr": "Bonjour le monde.", "es": "Hola mundo.", "source": "tatoeba_parallel"}
    ]
    # Universal CEFR returns candidates without Spanish translations
    mock_mine_universal_cefr.return_value = [
        {"fr": "Je vais au supermarché.", "es": None, "source": "hf_universal_cefr"}
    ]

    mock_dedup.side_effect = lambda candidates: candidates
    mock_preprocess.return_value = [
        {
            "fr": "Bonjour le monde.",
            "es": "Hola mundo.",
            "level": "A1",
            "score": 1.0,
            "score_ml": 1.0,
            "source": "tatoeba_parallel",
        },
        {
            "fr": "Je vais au supermarché.",
            "es": "Voy al supermercado.",
            "level": "A1",
            "score": 1.2,
            "score_ml": 1.2,
            "source": "hf_universal_cefr",
        },
    ]
    mock_ingest.return_value = True

    with patch.object(sys, "argv", ["orchestrator.py", "--mine-tatoeba", "--mine-universal-cefr", "--ingest"]):
        main()

    # Both miners were called
    mock_mine_tatoeba.assert_called_once()
    mock_mine_universal_cefr.assert_called_once()

    # Deduplication was called with the combined candidate list (2 items)
    mock_dedup.assert_called_once()
    dedup_args = mock_dedup.call_args[0][0]
    assert len(dedup_args) == 2
    assert dedup_args[0]["source"] == "tatoeba_parallel"
    assert dedup_args[1]["source"] == "hf_universal_cefr"

    # Preprocessing (scoring and selective translation) was called on deduplicated candidates
    mock_preprocess.assert_called_once_with(dedup_args)

    # Ingestion was called on the final scored records
    mock_ingest.assert_called_once_with(mock_preprocess.return_value)

