import os
import sys
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from backend.data_manager import DataManager

def run_seed_lexemas(db_path=None):
    if db_path is None:
        db_path = os.environ.get("FRENCHTUTOR_DB_PATH") or os.path.join(BASE_DIR, 'data', 'app.db')
        
    flelex_path = os.path.join(BASE_DIR, 'data', 'flelex_beacco.tsv')
    
    if not os.path.exists(flelex_path):
        print(f"Error: Could not find lexema vocabulary data at {flelex_path}")
        return False

    print(f"Loading vocabulary from {flelex_path}...")
    
    core_a1_lemmas = {
        'le', 'la', 'un', 'ce', 'cet', 'cette', 'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
        'me', 'te', 'se', 'moi', 'toi', 'lui', 'y', 'en', 'mon', 'ton', 'son', 'notre', 'votre', 'leur',
        'à', 'de', 'pour', 'dans', 'sur', 'avec', 'sans', 'sous', 'chez', 'vers', 'par', 'entre',
        'et', 'ou', 'mais', 'car', 'si', 'que', 'qui', 'quand', 'où', 'comme', 'donc',
        'être', 'avoir', 'aller', 'faire', 'dire', 'pouvoir', 'vouloir', 'savoir', 'voir', 'devoir', 'venir', 'falloir',
        'ne', 'pas', 'plus', 'très', 'bien', 'oui', 'non', 'merci', 'bonjour', 'bonsoir', 'salut',
        'aujourd\'hui', 'hier', 'demain'
    }

    records = []
    with open(flelex_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            word = row['word'].lower().strip()
            pos = (row.get('tag') or '').upper().strip() or None
            level = (row.get('level') or 'A1').upper().strip()
            try:
                freq = float(row.get('freq_total', 0.0) or 0.0)
            except (ValueError, TypeError):
                freq = 0.0

            is_base = 1 if (
                word in core_a1_lemmas or
                pos in ('DET:ART', 'DET:POS', 'PRO', 'PRP', 'PRP:DET', 'KON') or
                (level == 'A1' and word in ('être', 'avoir', 'aller', 'faire', 'dire', 'pouvoir', 'vouloir', 'voir', 'savoir'))
            ) else 0

            records.append((word, pos, level, freq, is_base, '{}'))
            
    if not records:
        return False
        
    print(f"Preparing to insert {len(records)} lexemas into the database...")
    
    dm = DataManager(db_path=db_path)
    
    with dm._get_connection() as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO lexemas (lemma, pos, level, frequency, is_base, metadata)
            VALUES (?, ?, ?, ?, ?, ?);
        """, records)
        lex_count = conn.execute("SELECT COUNT(*) FROM lexemas;").fetchone()[0]
        
    print(f"Successfully seeded database. Total lexemas in DB: {lex_count}")
    return True

if __name__ == "__main__":
    run_seed_lexemas()
