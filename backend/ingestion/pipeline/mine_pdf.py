import os
from typing import List, Dict, Any

from backend.nlp.vocab_engine import VocabEngine
from backend.ingestion.pipeline.common import (
    get_spacy_nlp,
    format_sentence,
    is_valid_sentence,
    get_lemma_minhash,
    deduplicate_against_db,
    extract_sentences_from_pdf,
    translate_batch,
    process_and_score_sentences,
)

__all__ = [
    "get_spacy_nlp",
    "format_sentence",
    "is_valid_sentence",
    "get_lemma_minhash",
    "deduplicate_against_db",
    "extract_sentences_from_pdf",
    "translate_batch",
    "process_and_score_sentences",
    "run_mine_pdf",
]


def run_mine_pdf(pdf_path: str, interactive: bool = True) -> List[Dict[str, Any]]:
    """Mines and scores sentences from a PDF file, returning standard sentence records."""
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return []

    print(f"Extracting sentences from {pdf_path}...")
    new_sentences = extract_sentences_from_pdf(pdf_path)

    print(f"Extracted {len(new_sentences)} candidate sentences from PDF.")
    unique_new_sentences = deduplicate_against_db(new_sentences)
    print(f"After deduplication, {len(unique_new_sentences)} unique sentences remain.")

    if not unique_new_sentences:
        print("No new unique sentences found.")
        return []

    engine = VocabEngine()

    if interactive:
        print("\nCalculating CEFR Levels using ML Classifier to group candidates...")
        level_groups = {'A1': [], 'A2': [], 'B1': [], 'B2': [], 'C1': [], 'C2': []}
        ml_results = engine.score_sentences_ml_batch(unique_new_sentences, batch_size=64)
        for sent, ml_res in zip(unique_new_sentences, ml_results):
            lvl = ml_res["level"]
            score_ml = ml_res["score"]
            if lvl in level_groups:
                level_groups[lvl].append((sent, score_ml))

        print("\nCEFR Level Distribution of Extracted Sentences:")
        for lvl, sents in level_groups.items():
            print(f"  {lvl}: {len(sents)}")

        selected_sentences = []
        for lvl, sents in level_groups.items():
            if not sents:
                continue
            while True:
                ans = input(f"How many sentences of level {lvl} to mine? (number or 'all', max {len(sents)}): ").strip().lower()
                if ans == 'all':
                    selected_sentences.extend([s[0] for s in sents])
                    break
                elif ans.isdigit():
                    count = min(int(ans), len(sents))
                    selected_sentences.extend([s[0] for s in sents[:count]])
                    break
                else:
                    print("Invalid input. Please enter a number or 'all'.")

        sentences_to_process = selected_sentences
    else:
        sentences_to_process = unique_new_sentences

    if not sentences_to_process:
        print("No sentences selected.")
        return []

    filename = os.path.basename(pdf_path)
    records = process_and_score_sentences(
        sentences_to_process,
        source_name="pdf",
        source_metadata={"source_file": filename},
        engine=engine
    )

    print(f"Successfully mined {len(records)} sentences from PDF.")
    return records


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_mine_pdf(sys.argv[1])
    else:
        print("Usage: python mine_pdf.py <path_to_pdf>")
