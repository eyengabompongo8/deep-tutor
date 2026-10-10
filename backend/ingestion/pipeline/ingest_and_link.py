import os
import time
from typing import List, Dict, Any

from backend.db import DataManager
from backend.nlp.vocab_engine import VocabEngine
from backend.db.connection import DB_PATH, ROOT_DIR


def run_clear_db(db_path: str = None) -> bool:
    """Clears all sentences and sentence vocabulary links from the database."""
    if db_path is None:
        db_path = os.environ.get("DEEPTUTOR_DB_PATH") or os.environ.get("FRENCHTUTOR_DB_PATH") or DB_PATH
        
    print(f"\n--- Clearing Sentences and Links from Database ({db_path}) ---")
    dm = DataManager(db_path=db_path)
    with dm._get_connection() as conn:
        has_sl = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sentence_lemmas';").fetchone()
        if has_sl:
            conn.execute("DELETE FROM sentence_lemmas;")
        has_sw = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sentence_words';").fetchone()
        if has_sw:
            conn.execute("DELETE FROM sentence_words;")
        has_slex = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sentence_lexemas';").fetchone()
        if has_slex:
            conn.execute("DELETE FROM sentence_lexemas;")
        conn.execute("DELETE FROM sentences;")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('sentences', 'sentence_lemmas', 'sentence_words', 'sentence_lexemas');")
    print("Database sentences and links cleared successfully.")
    return True


def run_ingest_and_link(sentences: List[Dict[str, Any]], db_path: str = None, batch_size: int = 500) -> bool:
    """
    Ingests a list of standard sentence objects into the database after strictly filtering
    out any sentence that contains words without a lemma in the database.
    """
    if not sentences:
        print("No sentences provided to ingest.")
        return False
        
    if db_path is None:
        db_path = os.environ.get("DEEPTUTOR_DB_PATH") or os.environ.get("FRENCHTUTOR_DB_PATH") or DB_PATH
        
    print(f"\n--- Filtering and Ingesting {len(sentences)} Candidate Sentences ---")
    start_time = time.time()
    
    print("Connecting to database...")
    dm = DataManager(db_path=db_path)
    
    print("Initializing VocabEngine for lemma coverage validation...")
    engine = VocabEngine()
    
    print("Validating sentence lemma coverage (filtering out sentences with missing lemmas)...")
    accepted_sentences, skipped_count = engine.filter_sentences_with_complete_lemmas(
        sentences, dm, batch_size=batch_size, show_progress=True
    )
    print(f"Validation complete: {len(accepted_sentences)} accepted, {skipped_count} skipped.")
    
    if not accepted_sentences:
        print("No sentences satisfied the complete lemma requirement. Ingestion skipped.")
        return True
    
    print(f"Bulk inserting {len(accepted_sentences)} accepted sentences into 'sentences' table...")
    inserted_count = dm.bulk_insert_sentences(accepted_sentences, clear_existing=False)
    print(f"Successfully inserted {inserted_count} sentences.")
    
    print(f"Extracting tokens and generating junction links (batch size: {batch_size})...")
    engine.link_sentence_tokens_to_database(dm, batch_size=batch_size, verbose=True)
    
    with dm._get_connection() as conn:
        has_lemmas = conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name='lemmas';").fetchone()
        lemma_tbl = "lemmas" if has_lemmas else "lexemas"
        lemma_count = conn.execute(f"SELECT COUNT(*) FROM {lemma_tbl};").fetchone()[0]

        has_words = conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name='words';").fetchone()
        word_count = conn.execute("SELECT COUNT(*) FROM words;").fetchone()[0] if has_words else 0

        has_sl = conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name='sentence_lemmas';").fetchone()
        lemma_link_count = conn.execute("SELECT COUNT(*) FROM sentence_lemmas;").fetchone()[0] if has_sl else 0

        has_sw = conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name='sentence_words';").fetchone()
        word_link_count = conn.execute("SELECT COUNT(*) FROM sentence_words;").fetchone()[0] if has_sw else 0

        sent_indexed = conn.execute("SELECT COUNT(DISTINCT sentence_id) FROM sentence_lemmas;").fetchone()[0] if has_sl else 0

    print("\n=== Ingestion & Indexing Summary ===")
    print(f"Candidate Sentences Evaluated: {len(sentences)}")
    print(f"Sentences Accepted & Stored:   {inserted_count}")
    print(f"Sentences Skipped:             {skipped_count}")
    print(f"Total Unique Lemmas:           {lemma_count}")
    print(f"Total Unique Words:            {word_count}")
    print(f"Sentence-Lemma Links:          {lemma_link_count}")
    print(f"Sentence-Word Links:           {word_link_count}")
    print(f"Sentences Indexed:             {sent_indexed}")
    print(f"Total Time:                    {time.time() - start_time:.2f}s")
    print("====================================\n")
    
    return True
