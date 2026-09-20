import os
import sys
import re
from typing import List, Dict, Any
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from scripts.vocab_engine import VocabEngine

def clean_sentence(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip('"\'«»')
    return text.strip()

def is_valid_sentence(text: str, min_words: int = 3, max_words: int = 25) -> bool:
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

def run_mine_tatoeba(max_per_level: int = 1200) -> List[Dict[str, Any]]:
    print("\n--- Mining Bilingual Dataset (Tatoeba) ---")
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

    print("Filtering valid candidate sentences...")
    valid_pairs = []
    seen_fr = set()
    for es_raw, fr_raw in zip(es_lines, fr_lines):
        es = clean_sentence(es_raw)
        fr = clean_sentence(fr_raw)
        if not is_valid_sentence(fr) or not is_valid_sentence(es):
            continue
        if fr.lower() in seen_fr:
            continue
        seen_fr.add(fr.lower())
        valid_pairs.append((es, fr))

    print(f"Scoring {len(valid_pairs)} candidate pairs using CEFR ML Model...")
    batch_size = 64
    all_fr_texts = [p[1] for p in valid_pairs]
    ml_results = engine.score_sentences_ml_batch(all_fr_texts, batch_size=batch_size)

    buckets = {'A1': [], 'A2': [], 'B1': [], 'B2': [], 'C1': [], 'C2': []}
    for (es, fr), ml_res in zip(valid_pairs, ml_results):
        level = ml_res["level"]
        score_ml = ml_res["score"]
        _, lex_score, _ = engine.score_sentence(fr)

        if len(buckets[level]) < max_per_level:
            buckets[level].append({
                'es': es,
                'fr': fr,
                'level': level,
                'score': float(score_ml),
                'score_lexical': float(lex_score),
                'score_ml': float(score_ml),
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
    
    return unique_pairs
