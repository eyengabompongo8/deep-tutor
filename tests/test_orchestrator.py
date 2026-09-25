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
