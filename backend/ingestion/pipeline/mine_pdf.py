import os
from typing import List, Dict, Any

from backend.ingestion.pipeline.common import (
    get_spacy_nlp,
    format_sentence,
    is_valid_sentence,
    extract_sentences_from_pdf,
)

__all__ = [
    "get_spacy_nlp",
    "format_sentence",
    "is_valid_sentence",
    "extract_sentences_from_pdf",
    "run_mine_pdf",
]


def run_mine_pdf(pdf_path: str, interactive: bool = False) -> List[Dict[str, Any]]:
    """Extracts valid candidate sentences from a PDF file, returning candidate records."""
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return []

    print(f"Extracting sentences from {pdf_path}...")
    new_sentences = extract_sentences_from_pdf(pdf_path)

    print(f"Extracted {len(new_sentences)} candidate sentences from PDF.")
    filename = os.path.basename(pdf_path)
    records = [
        {
            "fr": sent,
            "es": None,
            "source": "pdf",
            "metadata": {"source_file": filename}
        }
        for sent in new_sentences
    ]

    if interactive:
        print(f"\nExtracted {len(records)} candidate sentences:")
        for idx, rec in enumerate(records[:20], 1):
            print(f"  {idx}. {rec['fr']}")
        if len(records) > 20:
            print(f"  ... and {len(records) - 20} more sentences.")

    print(f"Successfully extracted {len(records)} candidate records from PDF.")
    return records


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_mine_pdf(sys.argv[1])
    else:
        print("Usage: python mine_pdf.py <path_to_pdf>")
