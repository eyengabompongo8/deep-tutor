import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_manager import DataManager

def convert_tatoeba():
    es_file = os.path.join(os.path.dirname(__file__), 'Tatoeba.es-fr.es')
    fr_file = os.path.join(os.path.dirname(__file__), 'Tatoeba.es-fr.fr')
    out_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentences.json')
    
    if not os.path.exists(es_file) or not os.path.exists(fr_file):
        print(f"Tatoeba source files not found at {es_file} / {fr_file}")
        return

    with open(es_file, 'r', encoding='utf-8') as fes, \
         open(fr_file, 'r', encoding='utf-8') as ffr:
        es_lines = fes.readlines()
        fr_lines = ffr.readlines()
        
    sentences = []
    # Tatoeba es-fr has about 40k sentences. 
    # To avoid exploding the processing time of our pipeline, we'll take a large sample of 5,000 sentences
    # Since Tatoeba sentences are usually ordered randomly or by length, we can take the first 5000.
    limit = min(5000, len(es_lines))
    for i in range(limit):
        es = es_lines[i].strip()
        fr = fr_lines[i].strip()
        if es and fr:
            sentences.append({
                "id": len(sentences) + 1,
                "es": es,
                "fr": fr,
                "level": "A1",
                "score_lexical": 1.0,
                "source": "tatoeba"
            })
            
    dm = DataManager()
    inserted = dm.bulk_insert_sentences(sentences, clear_existing=True)

    # Optional JSON backup
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
        
    print(f"Converted {inserted} sentence pairs into database (and {out_file})")

if __name__ == "__main__":
    convert_tatoeba()
