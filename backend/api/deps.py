from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.db import DataManager
    from backend.scoring.engine import ScoringEngine
    from backend.nlp.tts_engine import TTSEngine
    from backend.ml.precompute import EagerPrecomputeManager

data_manager: Optional["DataManager"] = None
scoring_engine: Optional["ScoringEngine"] = None
tts_engine: Optional["TTSEngine"] = None
eager_processor: Optional["EagerPrecomputeManager"] = None
