import os
import sys
import re
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from scripts.vocab_engine import VocabEngine

def clean_sentence(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip('"\'«»')
    return text.strip()

def is_valid_sentence(text, min_words=3, max_words=25):
    if not text:
        return False
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if 'http' in text or 'www.' in text or '@' in text or '<' in text:
        return False
    if not (text[0].isupper() or text[0] in ['«', '"', "'", '¿', '¡']):
        return False
    if not text[-1] in ['.', '!', '?', '»', '"', "'", ';']:
        return False
    return True

def run_mine_tatoeba(max_per_level=1200):
    print("\n--- Processing Bilingual Dataset (Tatoeba) ---")
    data_dir = os.path.join(BASE_DIR, 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    es_file = os.path.join(raw_dir, 'Tatoeba.es-fr.es')
    fr_file = os.path.join(raw_dir, 'Tatoeba.es-fr.fr')

    if not os.path.exists(es_file) or not os.path.exists(fr_file):
        print(f"Error: Tatoeba parallel files not found in {raw_dir}")
        return []

    print("Initializing FLELex VocabEngine for scoring...")
    engine = VocabEngine()

    with open(es_file, 'r', encoding='utf-8') as fes, open(fr_file, 'r', encoding='utf-8') as ffr:
        es_lines = fes.readlines()
        fr_lines = ffr.readlines()

    print(f"Total raw Tatoeba pairs: {len(es_lines)}")

    buckets = {'A1': [], 'A2': [], 'B1': [], 'B2': [], 'C1': [], 'C2': []}
    seen_fr = set()

    for es_raw, fr_raw in tqdm(zip(es_lines, fr_lines), total=len(es_lines), desc="Scoring Tatoeba"):
        es = clean_sentence(es_raw)
        fr = clean_sentence(fr_raw)

        if not is_valid_sentence(fr) or not is_valid_sentence(es):
            continue

        if fr.lower() in seen_fr:
            continue
        seen_fr.add(fr.lower())

        level, score, _ = engine.score_sentence(fr)
        if len(buckets[level]) < max_per_level:
            buckets[level].append({
                'es': es,
                'fr': fr,
                'level': level,
                'score': float(score),
                'score_lexical': float(score),
                'source': 'tatoeba'
            })

    total_tatoeba = sum(len(v) for v in buckets.values())
    print(f"\nTatoeba sentences harvested ({total_tatoeba} total):")
    
    level_order = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
    unique_pairs = []
    
    for lvl in sorted(buckets.keys(), key=lambda k: level_order.get(k, 0)):
        print(f"  Level {lvl}: {len(buckets[lvl])}")
        unique_pairs.extend(buckets[lvl])
        
    unique_pairs.sort(key=lambda x: (level_order.get(x['level'], 99), len(x['fr'])))
    
    for idx, item in enumerate(unique_pairs, start=1):
        item['id'] = idx

    return unique_pairs

if __name__ == "__main__":
    import json
    results = run_mine_tatoeba()
    if results:
        out_path = os.path.join(BASE_DIR, 'data', 'mined_sentences.json')
        print(f"\nSaving output to {out_path} for inspection...")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
