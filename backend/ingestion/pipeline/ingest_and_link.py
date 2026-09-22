import os
import time
from typing import List, Dict, Any

from backend.db import DataManager
from backend.nlp.vocab_engine import VocabEngine
from backend.db.connection import DB_PATH, ROOT_DIR


def run_clear_db(db_path: str = None) -> bool:
    """Clears all sentences and sentence-lexema links from the database."""
    if db_path is None:
        db_path = os.environ.get("FRENCHTUTOR_DB_PATH") or DB_PATH
        
    print(f"\n--- Clearing Sentences and Links from Database ({db_path}) ---")
    dm = DataManager(db_path=db_path)
    with dm._get_connection() as conn:
        conn.execute("DELETE FROM sentence_lexemas;")
        conn.execute("DELETE FROM sentences;")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('sentences', 'sentence_lexemas');")
    print("Database sentences and links cleared successfully.")
    return True


def run_ingest_and_link(sentences: List[Dict[str, Any]], db_path: str = None, batch_size: int = 500) -> bool:
    """
    Ingests a list of standard sentence objects into the database after strictly filtering
    out any sentence that contains words without a lexema in the database.
    """
    if not sentences:
        print("No sentences provided to ingest.")
        return False
        
    if db_path is None:
        db_path = os.environ.get("FRENCHTUTOR_DB_PATH") or DB_PATH
        
    print(f"\n--- Filtering and Ingesting {len(sentences)} Candidate Sentences ---")
    start_time = time.time()
    
    print("Connecting to database...")
    dm = DataManager(db_path=db_path)
    
    print("Initializing VocabEngine for lexema coverage validation...")
    engine = VocabEngine()
    
    print("Validating sentence lexema coverage (filtering out sentences with missing lexemas)...")
    accepted_sentences, skipped_count = engine.filter_sentences_with_complete_lexemas(
        sentences, dm, batch_size=batch_size, show_progress=True
    )
    print(f"Validation complete: {len(accepted_sentences)} accepted, {skipped_count} skipped.")
    
    if not accepted_sentences:
        print("No sentences satisfied the complete lexema requirement. Ingestion skipped.")
        return True
    
    print(f"Bulk inserting {len(accepted_sentences)} accepted sentences into 'sentences' table...")
    inserted_count = dm.bulk_insert_sentences(accepted_sentences, clear_existing=False)
    print(f"Successfully inserted {inserted_count} sentences.")
    
    print(f"Extracting lexemas and generating junction links (batch size: {batch_size})...")
    engine.link_sentence_lexemas_to_database(dm, batch_size=batch_size, verbose=True)
    
    with dm._get_connection() as conn:
        lex_count = conn.execute("SELECT COUNT(*) FROM lexemas;").fetchone()[0]
        link_count = conn.execute("SELECT COUNT(*) FROM sentence_lexemas;").fetchone()[0]
        sent_indexed = conn.execute("SELECT COUNT(DISTINCT sentence_id) FROM sentence_lexemas;").fetchone()[0]

    print("\n=== Ingestion & Indexing Summary ===")
    print(f"Candidate Sentences Evaluated: {len(sentences)}")
    print(f"Sentences Accepted & Stored:   {inserted_count}")
    print(f"Sentences Skipped:             {skipped_count}")
    print(f"Total Unique Lexemas:          {lex_count}")
    print(f"Sentence-Lexema Links:         {link_count}")
    print(f"Sentences Indexed:             {sent_indexed}")
    print(f"Total Time:                    {time.time() - start_time:.2f}s")
    print("====================================\n")
    
    return True
