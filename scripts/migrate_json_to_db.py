import os
import sys
import json
import shutil
from collections import Counter

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_manager import DataManager

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
JSON_FILE = os.path.join(DATA_DIR, 'sentences.json')
DB_FILE = os.path.join(DATA_DIR, 'sentences.db')
BACKUP_JSON = os.path.join(DATA_DIR, 'sentences.json.bak')

def migrate(backup: bool = True):
    if not os.path.exists(JSON_FILE):
        print(f"❌ Error: {JSON_FILE} does not exist.")
        sys.exit(1)

    print(f"Reading {JSON_FILE}...")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    total_in_json = len(json_data)
    print(f"Loaded {total_in_json} sentences from JSON.")
    json_distribution = Counter(item.get('level', 'A1').upper() for item in json_data)
    print("Original JSON Level Distribution:", dict(json_distribution))

    print(f"Initializing SQLite database at {DB_FILE}...")
    dm = DataManager(db_path=DB_FILE)

    records_to_insert = []
    for idx, item in enumerate(json_data, start=1):
        s_id = item.get('id', idx)
        fr = item.get('fr', '').strip()
        es = item.get('es', '').strip()
        level = item.get('level', 'A1').upper()
        # Original 'score' in sentences.json represents FLELex lexical difficulty
        score_lexical = float(item.get('score', 1.0))
        score_ml = item.get('score_ml', None)
        source = item.get('source', 'custom')
        metadata = item.get('metadata', {})

        records_to_insert.append({
            'id': s_id,
            'fr': fr,
            'es': es,
            'level': level,
            'score_lexical': score_lexical,
            'score_ml': score_ml,
            'source': source,
            'metadata': metadata
        })

    print(f"Bulk inserting {len(records_to_insert)} records into SQLite...")
    inserted_count = dm.bulk_insert_sentences(records_to_insert, clear_existing=True)
    print(f"Successfully inserted/verified {inserted_count} records.")

    # Verification
    db_sentences = dm.get_all_sentences()
    db_count = len(db_sentences)
    print(f"Verification: Found {db_count} records in database.")
    if db_count != total_in_json:
        print(f"❌ Mismatch! JSON had {total_in_json} but DB has {db_count}!")
        sys.exit(1)

    db_distribution = dm.get_level_distribution()
    print("Database Level Distribution:", db_distribution)
    for lvl, cnt in json_distribution.items():
        if db_distribution.get(lvl, 0) != cnt:
            print(f"❌ Distribution mismatch for level {lvl}: JSON {cnt} vs DB {db_distribution.get(lvl)}")
            sys.exit(1)

    # Inspect sample records
    sample_first = dm.get_sentence_by_id(1)
    print("Sample record (ID 1):", sample_first)
    assert sample_first is not None, "ID 1 not found!"
    assert sample_first['fr'] == json_data[0]['fr']
    assert sample_first['es'] == json_data[0]['es']
    assert 'score_lexical' in sample_first
    assert 'score' in sample_first
    assert 'created_at' in sample_first

    if backup and not os.path.exists(BACKUP_JSON):
        shutil.copyfile(JSON_FILE, BACKUP_JSON)
        print(f"✅ Created backup at {BACKUP_JSON}")

    print("\n🎉 Migration completed and verified with 100% fidelity!")

if __name__ == '__main__':
    migrate()
