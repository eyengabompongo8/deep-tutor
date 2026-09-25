import os
import re
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm

from backend.ingestion.pipeline.common import (
    get_spacy_nlp,
    format_sentence,
    is_valid_sentence,
    deduplicate_against_db,
    process_and_score_sentences,
)


def extract_sentences_from_conversations(
    limit: Optional[int] = None,
    drop_generic_greetings: bool = True
) -> List[str]:
    """
    Loads CATIE-AQ/everyday-conversations-llama3.1-2k-in-french and unpacks
    multi-turn dialogues into clean, individual spoken French sentences.
    """
    nlp = get_spacy_nlp()

    print("Loading CATIE-AQ/everyday-conversations dataset from HuggingFace...")
    dataset = load_dataset("CATIE-AQ/everyday-conversations-llama3.1-2k-in-french", split="train")

    print(f"Total conversations in dataset: {len(dataset)}")

    # Generic short greetings or assistant boilerplate that are too repetitive or unnatural
    generic_greetings = {
        "bonjour !",
        "bonjour",
        "salut !",
        "salut",
        "coucou !",
        "coucou",
        "bonsoir !",
        "bonsoir",
        "yo !",
        "yo",
        "bonjour ! comment puis-je vous aider aujourd'hui ?",
        "bonjour ! comment puis-je t'aider aujourd'hui ?",
    }

    candidates = []
    seen = set()

    for row in tqdm(dataset, desc="Extracting conversation sentences"):
        messages = row.get("messages", [])
        if not messages:
            continue

        for msg in messages:
            content = msg.get("content", "").strip()
            if not content:
                continue

            # Quick check: ignore single-word turns
            if len(content.split()) < 3:
                continue

            if drop_generic_greetings and content.lower() in generic_greetings:
                continue

            # Filter non-metric units or English artifacts if present
            if re.search(r'\b(onces?|miles?|gallons?|livres?\s+d\'eau)\b', content, re.IGNORECASE):
                continue

            # Parse conversational turn with spaCy to split multiple sentences
            doc = nlp(content)
            for sent in doc.sents:
                raw_sent = sent.text.strip()
                if not raw_sent:
                    continue

                if drop_generic_greetings and raw_sent.lower() in generic_greetings:
                    continue

                clean_sent = format_sentence(raw_sent)
                if not is_valid_sentence(clean_sent, min_words=3, max_words=30):
                    continue

                sent_lower = clean_sent.lower()
                if sent_lower in seen:
                    continue

                seen.add(sent_lower)
                candidates.append(clean_sent)

                if limit is not None and len(candidates) >= limit:
                    return candidates

    return candidates


def run_mine_catie_conversations(
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Mines everyday spoken sentences from CATIE-AQ conversations:
    1. Unpacks conversational turns and segments them with spaCy.
    2. Filters out conversational boilerplate and non-metric artifacts.
    3. Deduplicates against existing database sentences using MinHash LSH.
    4. Evaluates CEFR ML levels, lexical scores, and Spanish translations.
    """
    print("\n--- Mining Spoken French (CATIE-AQ Everyday Conversations) ---")
    candidates = extract_sentences_from_conversations(limit=limit)
    print(f"Found {len(candidates)} candidate sentences from everyday conversations.")

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
        source_name="hf_catie_conversations",
        source_metadata={"dataset": "CATIE-AQ/everyday-conversations-llama3.1-2k-in-french"}
    )

    print(f"Successfully mined and prepared {len(records)} sentences from everyday conversations.")
    return records


if __name__ == "__main__":
    run_mine_catie_conversations()
