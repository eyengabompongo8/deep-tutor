import os
import csv
import zipfile
import urllib.request
import io
from backend.db import DataManager
from backend.db.connection import DB_PATH, ROOT_DIR

def run_seed_lexemas(db_path=None):
    if db_path is None:
        db_path = os.environ.get("FRENCHTUTOR_DB_PATH") or DB_PATH
        
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

    print(f"Loading vocabulary from {lexique_file}...")
    
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
    seen = set()
    with open(lexique_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            word = row.get('4_Lemme')
            if not word:
                continue
            word = word.lower().strip()
            
            if word in seen:
                continue
            seen.add(word)
            
            pos = (row.get('5_Cgram') or '').upper().strip() or None
            level = 'A1'  # Default level as requested
            try:
                freq = float(row.get('12_FreqLemme', 0.0) or 0.0)
            except (ValueError, TypeError):
                freq = 0.0

            is_base = 1 if word in core_a1_lemmas else 0

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
