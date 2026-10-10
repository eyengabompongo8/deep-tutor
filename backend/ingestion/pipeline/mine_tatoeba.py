import os
import re
import bz2
import urllib.request
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from backend.db.connection import ROOT_DIR
from backend.ingestion.pipeline.common import (
    get_spacy_nlp,
    format_sentence,
    clean_quotes_and_symbols,
    is_valid_sentence as common_is_valid_sentence,
)

# Backwards compatibility helpers
def clean_sentence(text: str) -> str:
    """Cleans up internal spacing and removes quotes and symbol artifacts."""
    return format_sentence(text)


def is_valid_sentence(text: str, min_words: int = 3, max_words: int = 25) -> bool:
    """Compatibility wrapper checking validity with word-count limits."""
    return common_is_valid_sentence(text, min_words=min_words, max_words=max_words)


def _download_file(url: str, dest_path: str, desc: str):
    """Downloads a remote file with progress reporting and user-agent headers."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return

    print(f"Downloading {desc} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "DeepTutor-Ingestion/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f_out:
        total_size = int(resp.headers.get("Content-Length", 0))
        block_size = 1024 * 64
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                f_out.write(chunk)
                pbar.update(len(chunk))


def ensure_tatoeba_data_files() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Ensures Tatoeba data files exist locally in data/raw/ or data/raw/tatoeba/.
    Returns (legacy_es_path, legacy_fr_path, fra_sentences_path).
    """
    raw_dir = os.path.join(ROOT_DIR, "data", "raw")
    tatoeba_dir = os.path.join(raw_dir, "tatoeba")
    os.makedirs(tatoeba_dir, exist_ok=True)

    # 1. Check legacy parallel files
    legacy_es = os.path.join(raw_dir, "Tatoeba.es-fr.es")
    legacy_fr = os.path.join(raw_dir, "Tatoeba.es-fr.fr")

    # 2. Monolingual French sentences file
    fra_bz2 = os.path.join(tatoeba_dir, "fra_sentences.tsv.bz2")
    fra_tsv = os.path.join(tatoeba_dir, "fra_sentences.tsv")

    return legacy_es, legacy_fr, fra_bz2, fra_tsv


def get_sentence_count(text: str, nlp=None) -> int:
    """Counts sentences in text using spaCy sentence segmentation."""
    if nlp is None:
        nlp = get_spacy_nlp()
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    return len(sents)


def mine_tatoeba_parallel(
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Mines direct French-Spanish sentence pairs.
    If a record contains multiple sentences, or if French and Spanish sentence
    counts differ, it is discarded to prevent alignment errors.
    Returns candidate records without ML scoring, lexical scoring, or translation.
    """
    print("\n--- Mining Bilingual Parallel Sentences (Tatoeba fra-spa) ---")
    raw_dir = os.path.join(ROOT_DIR, "data", "raw")
    es_file = os.path.join(raw_dir, "Tatoeba.es-fr.es")
    fr_file = os.path.join(raw_dir, "Tatoeba.es-fr.fr")

    tatoeba_dir = os.path.join(raw_dir, "tatoeba")
    os.makedirs(tatoeba_dir, exist_ok=True)

    # If legacy files not present, download fra-spa_links and sentences
    if not (os.path.exists(es_file) and os.path.exists(fr_file)):
        print("Legacy Tatoeba.es-fr files not found. Attempting Tatoeba exports download...")
        links_bz2 = os.path.join(tatoeba_dir, "fra-spa_links.tsv.bz2")
        fra_bz2 = os.path.join(tatoeba_dir, "fra_sentences.tsv.bz2")
        spa_bz2 = os.path.join(tatoeba_dir, "spa_sentences.tsv.bz2")

        _download_file(
            "https://downloads.tatoeba.org/exports/per_language/fra/fra-spa_links.tsv.bz2",
            links_bz2,
            "fra-spa links"
        )
        _download_file(
            "https://downloads.tatoeba.org/exports/per_language/fra/fra_sentences.tsv.bz2",
            fra_bz2,
            "French sentences"
        )
        _download_file(
            "https://downloads.tatoeba.org/exports/per_language/spa/spa_sentences.tsv.bz2",
            spa_bz2,
            "Spanish sentences"
        )

        # Load sentence mappings
        print("Loading French sentences index...")
        fr_map = {}
        with bz2.open(fra_bz2, "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    fr_map[parts[0]] = parts[2]

        print("Loading Spanish sentences index...")
        es_map = {}
        with bz2.open(spa_bz2, "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    es_map[parts[0]] = parts[2]

        print("Linking French-Spanish pairs from links TSV...")
        pairs = []
        with bz2.open(links_bz2, "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    fr_id, es_id = parts[0], parts[1]
                    if fr_id in fr_map and es_id in es_map:
                        pairs.append((es_map[es_id], fr_map[fr_id]))
        print(f"Total linked Tatoeba pairs: {len(pairs)}")
    else:
        with open(es_file, "r", encoding="utf-8") as fes, open(fr_file, "r", encoding="utf-8") as ffr:
            es_lines = fes.readlines()
            fr_lines = ffr.readlines()
        pairs = list(zip([e.strip() for e in es_lines], [f.strip() for f in fr_lines]))
        print(f"Total legacy Tatoeba pairs: {len(pairs)}")

    if not pairs:
        print("No parallel pairs found to process.")
        return []

    nlp = get_spacy_nlp()
    print("Filtering and cleaning candidate parallel pairs...")
    candidates = []
    seen_fr = set()

    for es_raw, fr_raw in tqdm(pairs, desc="Validating parallel pairs"):
        fr_clean = format_sentence(fr_raw)
        es_clean = clean_quotes_and_symbols(es_raw).strip()

        # Validation checks
        if not is_valid_sentence(fr_clean) or not es_clean:
            continue

        # Single-sentence enforcement:
        # Discard multi-sentence pairs if alignment cannot be guaranteed
        doc_fr = nlp(fr_clean)
        sents_fr = [s.text.strip() for s in doc_fr.sents if s.text.strip()]
        if len(sents_fr) != 1:
            continue

        doc_es = nlp(es_clean)
        sents_es = [s.text.strip() for s in doc_es.sents if s.text.strip()]
        if len(sents_es) != 1:
            continue

        fr_lower = fr_clean.lower()
        if fr_lower in seen_fr:
            continue
        seen_fr.add(fr_lower)

        candidates.append({
            "fr": fr_clean,
            "es": es_clean,
            "source": "tatoeba_parallel",
            "metadata": {}
        })
        if limit is not None and len(candidates) >= limit:
            break

    print(f"Found {len(candidates)} valid single-sentence parallel candidates.")
    return candidates


def mine_tatoeba_monolingual(
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Mines French sentences from Tatoeba (overall), splits multi-sentence entries
    with spaCy so each valid sentence is retained as a candidate record.
    Returns candidate records without ML scoring, lexical scoring, or translation.
    """
    print("\n--- Mining Monolingual French Sentences (Tatoeba fra) ---")
    tatoeba_dir = os.path.join(ROOT_DIR, "data", "raw", "tatoeba")
    fra_bz2 = os.path.join(tatoeba_dir, "fra_sentences.tsv.bz2")

    _download_file(
        "https://downloads.tatoeba.org/exports/per_language/fra/fra_sentences.tsv.bz2",
        fra_bz2,
        "French monolingual sentences"
    )

    if not os.path.exists(fra_bz2):
        print(f"Error: {fra_bz2} not available.")
        return []

    nlp = get_spacy_nlp()
    print("Extracting and segmenting French sentences...")
    candidates = []
    seen = set()

    with bz2.open(fra_bz2, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading fra_sentences"):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            raw_text = parts[2]
            # Split record into individual sentences using spaCy
            doc = nlp(raw_text)
            for sent in doc.sents:
                clean_sent = format_sentence(sent.text)
                if not is_valid_sentence(clean_sent):
                    continue

                sent_lower = clean_sent.lower()
                if sent_lower in seen:
                    continue
                seen.add(sent_lower)
                candidates.append({
                    "fr": clean_sent,
                    "es": None,
                    "source": "tatoeba_monolingual",
                    "metadata": {"dataset": "tatoeba_fra"}
                })

                if limit is not None and len(candidates) >= limit:
                    break
            if limit is not None and len(candidates) >= limit:
                break

    print(f"Extracted {len(candidates)} valid candidate sentences from monolingual Tatoeba.")
    return candidates


def run_mine_tatoeba(
    mode: str = "both",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Main entry point for mining Tatoeba.
    Returns candidate records in the format:
    [{"fr": str, "es": Optional[str], "source": str, "metadata": dict}]
    Modes:
      - 'parallel': Mines direct French-Spanish pairs. Discards multi-sentence pairs.
      - 'monolingual': Mines French overall, splits multi-sentence entries.
      - 'both': Mines both and aggregates candidate records.
    """
    print(f"\n==================================================")
    print(f"Tatoeba Mining Pipeline (Mode: {mode})")
    print(f"==================================================")

    results = []
    if mode in ("parallel", "both"):
        par_results = mine_tatoeba_parallel(limit=limit)
        results.extend(par_results)

    if mode in ("monolingual", "both"):
        remaining_limit = None
        if limit is not None:
            remaining_limit = max(0, limit - len(results)) if mode == "both" else limit
        mono_results = mine_tatoeba_monolingual(limit=remaining_limit)
        results.extend(mono_results)

    if limit is not None:
        results = results[:limit]

    print(f"\nTotal Tatoeba records harvested across selected mode ({mode}): {len(results)}")
    return results


if __name__ == "__main__":
    import sys
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "both"
    run_mine_tatoeba(mode=mode_arg)
