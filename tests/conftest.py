import os
import sys
import tempfile
import pytest

# Ensure repository root and backend directory are on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Set a fallback temporary DB for any top-level module imports before tests run
_SESSION_TEMP_DB = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
_SESSION_TEMP_DB.close()
os.environ.setdefault("FRENCHTUTOR_DB_PATH", _SESSION_TEMP_DB.name)


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
def isolated_data_manager(temp_db_path):
    """Provide a DataManager attached to an isolated database with initial seed data."""
    from data_manager import DataManager
    dm = DataManager(db_path=temp_db_path)
    
    # Seed standard test sentences
    dm.bulk_insert_sentences([
        {"es": "Hola mundo", "fr": "Bonjour le monde", "level": "A1", "score_lexical": 1.0, "score_ml": 85.0},
        {"es": "El gato duerme", "fr": "Le chat dort", "level": "A1", "score_lexical": 1.2, "score_ml": 88.0},
        {"es": "Tengo que irme", "fr": "Je dois partir", "level": "A2", "score_lexical": 2.2, "score_ml": 90.0},
        {"es": "El restaurante está abierto", "fr": "Le restaurant est ouvert", "level": "B1", "score_lexical": 3.2, "score_ml": 92.0},
    ])
    return dm


@pytest.fixture
def test_client(isolated_data_manager, monkeypatch):
    """Provide a FastAPI TestClient wired to the isolated data manager and scoring engine."""
    from fastapi.testclient import TestClient
    import server
    from scoring_engine import ScoringEngine
    
    engine = ScoringEngine(data_manager=isolated_data_manager)
    monkeypatch.setattr(server, "data_manager", isolated_data_manager)
    monkeypatch.setattr(server, "scoring_engine", engine)
    
    return TestClient(server.app)
