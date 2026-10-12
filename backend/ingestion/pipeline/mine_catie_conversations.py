import os
import re
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm

from backend.ingestion.pipeline.common import (
    get_spacy_nlp,
    format_sentence,
    is_valid_sentence,
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
    Extracts everyday spoken sentences from CATIE-AQ conversations and returns
    candidate records without scoring or translation.
    """
    print("\n--- Mining Spoken French (CATIE-AQ Everyday Conversations) ---")
    candidates = extract_sentences_from_conversations(limit=limit)
    print(f"Found {len(candidates)} candidate sentences from everyday conversations.")

    records = [
        {
            "fr": sent,
            "es": None,
            "source": "hf_catie_conversations",
            "metadata": {"dataset": "CATIE-AQ/everyday-conversations-llama3.1-2k-in-french"},
        }
        for sent in candidates
    ]

    print(f"Successfully extracted {len(records)} candidate records from everyday conversations.")
    return records


if __name__ == "__main__":
    run_mine_catie_conversations()
