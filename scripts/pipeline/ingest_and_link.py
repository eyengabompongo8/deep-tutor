import os
import sys
import time
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from backend.data_manager import DataManager
from scripts.vocab_engine import VocabEngine

def run_ingest_and_link(sentences, db_path=None, batch_size=500):
    if not sentences:
        print("No sentences provided to ingest.")
        return False
        
    if db_path is None:
        db_path = os.environ.get("FRENCHTUTOR_DB_PATH") or os.path.join(BASE_DIR, 'data', 'app.db')
        
    print(f"\n--- Ingesting and Linking {len(sentences)} Sentences ---")
    start_time = time.time()
    
    print("Connecting to database...")
    dm = DataManager(db_path=db_path)
    
    print(f"Bulk inserting {len(sentences)} sentences into 'sentences' table...")
    inserted_count = dm.bulk_insert_sentences(sentences, clear_existing=True)
    print(f"Successfully inserted {inserted_count} sentences.")
    
    print("Initializing VocabEngine for semantic linking...")
    engine = VocabEngine()
    
    print(f"Extracting lexemas and generating junction links (batch size: {batch_size})...")
    engine.link_sentence_lexemas_to_database(dm, batch_size=batch_size, verbose=True)
    
    with dm._get_connection() as conn:
        lex_count = conn.execute("SELECT COUNT(*) FROM lexemas;").fetchone()[0]
        link_count = conn.execute("SELECT COUNT(*) FROM sentence_lexemas;").fetchone()[0]
        sent_indexed = conn.execute("SELECT COUNT(DISTINCT sentence_id) FROM sentence_lexemas;").fetchone()[0]

    print("\n=== Ingestion & Indexing Summary ===")
    print(f"Total Sentences Inserted:  {inserted_count}")
    print(f"Total Unique Lexemas:      {lex_count}")
    print(f"Sentence-Lexema Links:     {link_count}")
    print(f"Sentences Indexed:         {sent_indexed}")
    print(f"Total Time:                {time.time() - start_time:.2f}s")
    print("====================================\n")
    
    return True

if __name__ == "__main__":
    in_path = os.path.join(BASE_DIR, 'data', 'mined_sentences.json')
    if not os.path.exists(in_path):
        print(f"Error: {in_path} not found.")
        sys.exit(1)
        
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    run_ingest_and_link(data)
