#!/usr/bin/env python3
"""
Indexes all sentences in data/sentences.db:
- Extracts canonical word roots (lemmas) using spaCy + Lexique383 + FLELex
- Populates the 'lexemas' table
- Populates the 'sentence_lexemas' junction table
- Seeds foundational A1 base vocabulary
"""
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_manager import DataManager
from vocab_engine import VocabEngine


def main():
    print("Initializing DataManager and VocabEngine...")
    start_time = time.time()
    dm = DataManager()
    engine = VocabEngine()
    print(f"Loaded engines in {time.time() - start_time:.2f}s")

    print("\nStarting lexema database indexing...")
    index_start = time.time()
    engine.index_database_lexemas(dm, batch_size=500, verbose=True)
    print(f"Indexing completed in {time.time() - index_start:.2f}s")

    # Diagnostic summary
    with dm._get_connection() as conn:
        lex_count = conn.execute("SELECT COUNT(*) FROM lexemas;").fetchone()[0]
        base_count = conn.execute("SELECT COUNT(*) FROM lexemas WHERE is_base = 1;").fetchone()[0]
        link_count = conn.execute("SELECT COUNT(*) FROM sentence_lexemas;").fetchone()[0]
        sent_indexed = conn.execute("SELECT COUNT(DISTINCT sentence_id) FROM sentence_lexemas;").fetchone()[0]
        user_known = len(dm.get_user_known_lexema_ids('default_user'))

    print("\n=== Indexing Summary ===")
    print(f"Total Unique Lexemas: {lex_count}")
    print(f"Base Foundational Words (unlocked): {base_count}")
    print(f"Sentence-Lexema Links: {link_count}")
    print(f"Sentences Indexed: {sent_indexed}")
    print(f"Default User Initial Known Roots: {user_known}")
    print("========================\n")


if __name__ == "__main__":
    main()
