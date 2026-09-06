import os
import sys

# Ensure backend directory is in sys.path for direct module resolution
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from .data_manager import DataManager
from .scoring_engine import ScoringEngine, score_to_level
from .tts_engine import TTSEngine, get_tts_engine
from .vocab_engine import VocabEngine

__all__ = [
    "DataManager",
    "ScoringEngine",
    "score_to_level",
    "TTSEngine",
    "get_tts_engine",
    "VocabEngine",
]
