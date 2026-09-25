import os
from typing import Optional
from backend.ingestion.pipeline.lexique_classifier import CORE_A1_LEMMAS
from backend.ingestion.pipeline.seed_lemmas import run_seed_lemmas


def run_seed_lexemas(db_path: Optional[str] = None) -> bool:
    """
    Backward-compatible wrapper delegating to run_seed_lemmas.
    """
    return run_seed_lemmas(db_path=db_path)


if __name__ == "__main__":
    run_seed_lexemas()
