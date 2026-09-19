import os
import tempfile
from unittest.mock import MagicMock
import pytest

# Set a fallback temporary DB for any top-level module imports before tests run
_SESSION_TEMP_DB = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
_SESSION_TEMP_DB.close()
os.environ.setdefault("FRENCHTUTOR_DB_PATH", _SESSION_TEMP_DB.name)

# Canonical seed sentences used across tests
SEED_SENTENCES = [
    {"es": "Hola mundo", "fr": "Bonjour le monde", "level": "A1", "score_lexical": 1.0, "score_ml": 85.0},
    {"es": "El gato duerme", "fr": "Le chat dort", "level": "A1", "score_lexical": 1.2, "score_ml": 88.0},
    {"es": "Tengo que irme", "fr": "Je dois partir", "level": "A2", "score_lexical": 2.2, "score_ml": 90.0},
    {"es": "El restaurante está abierto", "fr": "Le restaurant est ouvert", "level": "B1", "score_lexical": 3.2, "score_ml": 92.0},
]


@pytest.fixture(scope="session", autouse=True)
def cleanup_session_db():
    """Clean up the session fallback database when the test run ends."""
    yield
    if os.path.exists(_SESSION_TEMP_DB.name):
        try:
            os.remove(_SESSION_TEMP_DB.name)
        except OSError:
            pass


@pytest.fixture
def temp_db_path():
    """Provide a fresh, isolated temporary SQLite database file for a test."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_file.close()
    yield temp_file.name
    if os.path.exists(temp_file.name):
        try:
            os.remove(temp_file.name)
        except OSError:
            pass


@pytest.fixture
def mock_tts(tmp_path):
    """Provide a MagicMock TTS engine so no real network requests are made."""
    fake_audio_path = tmp_path / "test_audio.wav"
    fake_audio_path.write_bytes(b"RIFF" + b"\x00" * 44)

    mock = MagicMock()
    mock.synthesize.return_value = str(fake_audio_path)
    mock.get_audio_path.return_value = str(fake_audio_path)
    mock.data_manager = None
    return mock


@pytest.fixture
def isolated_data_manager(temp_db_path):
    """Provide a DataManager attached to an isolated database with initial seed data."""
    from backend.data_manager import DataManager
    dm = DataManager(db_path=temp_db_path)
    dm.bulk_insert_sentences(SEED_SENTENCES)
    return dm


@pytest.fixture
def test_client(isolated_data_manager, mock_tts, monkeypatch):
    """Provide a FastAPI TestClient wired to the isolated data manager, scoring engine, and mock TTS."""
    from fastapi.testclient import TestClient
    from backend import server
    from backend.scoring_engine import ScoringEngine
    
    engine = ScoringEngine(data_manager=isolated_data_manager)
    monkeypatch.setattr(server, "data_manager", isolated_data_manager)
    monkeypatch.setattr(server, "scoring_engine", engine)
    monkeypatch.setattr(server, "tts_engine", mock_tts)
    
    return TestClient(server.app)
