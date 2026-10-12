from typing import List, Tuple

LEVEL_RANGES: List[Tuple[float, float, str]] = [
    (1.0, 2.0, 'A1'),
    (2.0, 3.0, 'A2'),
    (3.0, 4.0, 'B1'),
    (4.0, 5.0, 'B2'),
    (5.0, 6.0, 'C1'),
    (6.0, 6.01, 'C2'),
]


def score_to_level(score: float) -> str:
    """Converts a continuous numerical score (1.00 - 6.00) to its CEFR level string."""
    score = max(1.0, min(6.0, score))
    for low, high, level in LEVEL_RANGES:
        if low <= score < high:
            return level
    return 'C2'
