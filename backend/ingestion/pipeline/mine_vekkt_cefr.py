import os
from typing import List, Dict, Any, Optional
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from backend.ingestion.pipeline.common import (
    format_sentence,
    is_valid_sentence,
)


def extract_sentences_from_vekkt_cefr(
    limit: Optional[int] = None
) -> List[str]:
    """
    Loads vekkt/french_CEFR dataset and extracts candidate sentences
    across all CEFR difficulty levels (A1-C2).
    """
    target_levels_set = {"A1", "A2", "B1", "B2", "C1", "C2"}

    print("Loading vekkt/french_CEFR dataset from HuggingFace...")
    dataset_dict = load_dataset("vekkt/french_CEFR")

    # Only use train and validation (test has unlabelled difficulty)
    splits_to_use = [split for split in ["train", "validation"] if split in dataset_dict]
    dataset = concatenate_datasets([dataset_dict[s] for s in splits_to_use])

    print(f"Total rows in train/validation splits: {len(dataset)}")

    candidates = []
    seen = set()

    for row in tqdm(dataset, desc="Extracting vekkt/french_CEFR sentences"):
        difficulty = row.get("difficulty")
        if target_levels_set and difficulty not in target_levels_set:
            continue

        raw_sent = row.get("sentence", "")
        if not raw_sent:
            continue

        clean_sent = format_sentence(raw_sent)
        if not is_valid_sentence(clean_sent):
            continue

        sent_lower = clean_sent.lower()
        if sent_lower in seen:
            continue

        seen.add(sent_lower)
        candidates.append(clean_sent)

        if limit is not None and len(candidates) >= limit:
            break

    return candidates


def run_mine_vekkt_cefr(
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Mines candidate sentences from vekkt/french_CEFR:
    1. Extracts validated sentences matching target CEFR levels.
    2. Returns candidate records formatted for the ingestion pipeline.
    """
    print("\n--- Mining HuggingFace Dataset (vekkt/french_CEFR) ---")
    candidates = extract_sentences_from_vekkt_cefr(limit=limit)
    print(f"Found {len(candidates)} candidate sentences from vekkt/french_CEFR.")

    return [
        {
            "fr": sent,
            "es": None,
            "source": "hf_vekkt_cefr",
            "metadata": {"dataset": "vekkt/french_CEFR"},
        }
        for sent in candidates
    ]


if __name__ == "__main__":
    run_mine_vekkt_cefr()
