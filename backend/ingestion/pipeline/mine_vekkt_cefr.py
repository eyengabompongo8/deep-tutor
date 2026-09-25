import os
from typing import List, Dict, Any, Optional
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from backend.ingestion.pipeline.common import (
    format_sentence,
    is_valid_sentence,
    deduplicate_against_db,
    process_and_score_sentences,
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
    Mines sentences from vekkt/french_CEFR:
    1. Extracts validated sentences across all CEFR levels (A1-C2).
    2. Deduplicates against existing database sentences.
    3. Evaluates CEFR ML levels, lexical scores, and Spanish translations.
    """
    print("\n--- Mining HuggingFace Dataset (vekkt/french_CEFR) ---")
    candidates = extract_sentences_from_vekkt_cefr(limit=limit)
    print(f"Found {len(candidates)} candidate sentences from vekkt/french_CEFR.")

    if not candidates:
        print("No candidate sentences extracted.")
        return []

    unique_sentences = deduplicate_against_db(candidates)
    print(f"After deduplication against database, {len(unique_sentences)} unique sentences remain.")

    if not unique_sentences:
        print("No new sentences to process.")
        return []

    records = process_and_score_sentences(
        unique_sentences,
        source_name="hf_vekkt_cefr",
        source_metadata={"dataset": "vekkt/french_CEFR"}
    )

    print(f"Successfully mined and prepared {len(records)} sentences from vekkt/french_CEFR.")
    return records


if __name__ == "__main__":
    run_mine_vekkt_cefr()
