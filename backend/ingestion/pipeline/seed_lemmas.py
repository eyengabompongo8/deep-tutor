import os
import json
import zipfile
import urllib.request
import io
from typing import Optional

from backend.db import DataManager
from backend.db.connection import DB_PATH, ROOT_DIR
from backend.ingestion.pipeline.lexique_classifier import parse_and_classify_lexique


def run_seed_lemmas(db_path: Optional[str] = None) -> bool:
    """
    Downloads Lexique 4 if necessary, classifies all lemmas and surface words,
    and seeds the database tables `lemmas` and `words`.
    """
    if db_path is None:
        db_path = os.environ.get("DEEPTUTOR_DB_PATH") or os.environ.get("FRENCHTUTOR_DB_PATH") or DB_PATH

    lexique_dir = os.path.join(ROOT_DIR, 'data', 'lexique')
    lexique_file = os.path.join(lexique_dir, 'Lexique4', 'Lexique4.tsv')

    if not os.path.exists(lexique_file):
        print("Downloading Lexique 4 dataset...")
        os.makedirs(lexique_dir, exist_ok=True)
        url = 'http://www.lexique.org/databases/Lexique400/Lexique400.zip'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
                z.extractall(lexique_dir)
        print("Lexique 4 dataset downloaded and extracted.")

    print(f"Parsing and classifying vocabulary from {lexique_file}...")
    lemmas, words = parse_and_classify_lexique(lexique_file)

    if not lemmas:
        print("No lemmas parsed from Lexique 4.")
        return False

    print(f"Parsed {len(lemmas)} unique lemmas and {len(words)} surface words.")

    dm = DataManager(db_path=db_path)

    with dm._get_connection() as conn:
        # Ensure schema tables exist for lemmas and words
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lemmas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                pos TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'A1',
                frequency REAL DEFAULT 0.0,
                is_base BOOLEAN NOT NULL DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(lemma, pos)
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lemmas_pos ON lemmas(pos);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lemmas_level ON lemmas(level);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lemmas_is_base ON lemmas(is_base);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL,
                lemma_id INTEGER NOT NULL,
                pos TEXT NOT NULL,
                frequency REAL DEFAULT 0.0,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (lemma_id) REFERENCES lemmas(id) ON DELETE CASCADE,
                UNIQUE(word, lemma_id, pos)
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_words_pos ON words(pos);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_words_lemma_id ON words(lemma_id);")

        # Insert classified lemmas
        print(f"Seeding {len(lemmas)} lemmas into database...")
        lemma_records = [
            (
                item['lemma'],
                item['pos'],
                item['level'],
                item['frequency'],
                1 if item['is_base'] else 0,
                json.dumps(item['metadata']) if isinstance(item['metadata'], dict) else str(item['metadata'])
            )
            for item in lemmas
        ]
        conn.executemany("""
            INSERT OR IGNORE INTO lemmas (lemma, pos, level, frequency, is_base, metadata)
            VALUES (?, ?, ?, ?, ?, ?);
        """, lemma_records)

        # Query back lemma surrogate IDs
        lemma_id_map = {
            (r['lemma'], r['pos']): r['id']
            for r in conn.execute("SELECT id, lemma, pos FROM lemmas;").fetchall()
        }

        # Insert surface words linking to lemma_id
        print(f"Seeding {len(words)} surface words into database...")
        word_records = []
        for item in words:
            lemma_id = lemma_id_map.get((item['lemma'], item['pos']))
            if lemma_id is not None:
                word_records.append((
                    item['word'],
                    lemma_id,
                    item['pos'],
                    item['frequency'],
                    json.dumps(item['metadata']) if isinstance(item['metadata'], dict) else str(item['metadata'])
                ))

        conn.executemany("""
            INSERT OR IGNORE INTO words (word, lemma_id, pos, frequency, metadata)
            VALUES (?, ?, ?, ?, ?);
        """, word_records)

        # Backward compatibility: populate legacy lexemas table if it exists
        has_lexemas = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lexemas';"
        ).fetchone()
        if has_lexemas:
            conn.executemany("""
                INSERT OR IGNORE INTO lexemas (lemma, pos, level, frequency, is_base, metadata)
                VALUES (?, ?, ?, ?, ?, ?);
            """, lemma_records)

        lemma_count = conn.execute("SELECT COUNT(*) FROM lemmas;").fetchone()[0]
        word_count = conn.execute("SELECT COUNT(*) FROM words;").fetchone()[0]

    print(f"Successfully seeded database. Total lemmas in DB: {lemma_count}, total words: {word_count}")
    return True


if __name__ == "__main__":
    run_seed_lemmas()
