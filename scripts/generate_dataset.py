import os
import json
import gzip
import re
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))

from vocab_engine import VocabEngine
from data_manager import DataManager

DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'sentences.json')
DB_FILE = os.path.join(DATA_DIR, 'sentences.db')

def clean_sentence(text):
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove surrounding quotes
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
    # Check if starts with uppercase or quote
    if not (text[0].isupper() or text[0] in ['«', '"', "'", '¿', '¡']):
        return False
    # Check if ends with ending punctuation
    if not text[-1] in ['.', '!', '?', '»', '"', "'", ';']:
        return False
    return True

def process_tatoeba(engine, max_per_level=1000):
    print("\n--- 1. Processing Bilingual Dataset (Tatoeba) ---")
    raw_dir = os.path.join(DATA_DIR, 'raw')
    es_file = os.path.join(raw_dir, 'Tatoeba.es-fr.es')
    fr_file = os.path.join(raw_dir, 'Tatoeba.es-fr.fr')

    if not os.path.exists(es_file) or not os.path.exists(fr_file):
        print("Tatoeba parallel files not found.")
        return []

    with open(es_file, 'r', encoding='utf-8') as fes, \
         open(fr_file, 'r', encoding='utf-8') as ffr:
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
                'score': score,
                'source': 'tatoeba'
            })

    total_tatoeba = sum(len(v) for v in buckets.values())
    print(f"Tatoeba sentences harvested ({total_tatoeba} total):")
    for lvl, items in buckets.items():
        print(f"  {lvl}: {len(items)}")

    result = []
    for lvl in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
        result.extend(buckets[lvl])
    return result

def extract_complex_unilingual_candidates(engine, target_levels=['B1', 'B2', 'C1', 'C2'], quota_per_level=400):
    print("\n--- 2. Extracting Unilingual French Candidates ---")
    raw_dir = os.path.join(DATA_DIR, 'raw')
    corpora = [
        os.path.join(raw_dir, 'globalvoices_fr.txt.gz'),
        os.path.join(raw_dir, 'europarl_fr.txt.gz')
    ]

    candidates = {lvl: [] for lvl in target_levels}
    seen = set()

    for corpus in corpora:
        if not os.path.exists(corpus):
            continue
        print(f"Scanning {os.path.basename(corpus)}...")
        with gzip.open(corpus, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if all(len(candidates[lvl]) >= quota_per_level for lvl in target_levels):
                    break

                text = clean_sentence(line)
                if not is_valid_sentence(text, min_words=6, max_words=22):
                    continue

                # Filter out lines with too many numbers or non-standard characters
                if len(re.findall(r'\d+', text)) > 2:
                    continue
                if text.lower() in seen:
                    continue
                seen.add(text.lower())

                level, score, _ = engine.score_sentence(text)
                if level in candidates and len(candidates[level]) < quota_per_level:
                    candidates[level].append((text, level, score))

    extracted = []
    for lvl in target_levels:
        extracted.extend(candidates[lvl])
        print(f"  {lvl} unilingual candidates: {len(candidates[lvl])}")
    return extracted

def translate_and_build_unilingual(candidates, batch_size=32):
    if not candidates:
        return []

    print(f"\n--- 3. Translating {len(candidates)} Unilingual Sentences with Opus-MT ---")
    model_name = 'Helsinki-NLP/opus-mt-fr-es'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    device = 'cpu'
    model.to(device)

    fr_texts = [c[0] for c in candidates]
    translations = []

    for i in tqdm(range(0, len(fr_texts), batch_size), desc="Translating"):
        batch = fr_texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256)
        translated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(translated_batch)

    results = []
    for (fr, level, score), es_raw in zip(candidates, translations):
        es = clean_sentence(es_raw)
        if es and is_valid_sentence(es, min_words=3, max_words=30):
            results.append({
                'es': es,
                'fr': fr,
                'level': level,
                'score': score,
                'source': 'monolingual_corpus'
            })

    print(f"Successfully generated {len(results)} translated unilingual phrase pairs.")
    return results

def main():
    print("Initializing FLELex VocabEngine...")
    engine = VocabEngine()

    # 1. Harvest Tatoeba pairs across all levels
    tatoeba_pairs = process_tatoeba(engine, max_per_level=1200)

    # 2. Extract & translate authentic complex French sentences for B1-C2
    unilingual_candidates = extract_complex_unilingual_candidates(engine, target_levels=['B1', 'B2', 'C1', 'C2'], quota_per_level=350)
    unilingual_pairs = translate_and_build_unilingual(unilingual_candidates, batch_size=32)

    # 3. Merge, assign unique IDs, and sort
    all_pairs = tatoeba_pairs + unilingual_pairs

    # Deduplicate on French sentence
    unique_pairs = []
    seen = set()
    for item in all_pairs:
        key = item['fr'].lower()
        if key not in seen:
            seen.add(key)
            unique_pairs.append(item)

    # Sort primarily by CEFR level order then score
    level_order = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
    unique_pairs.sort(key=lambda x: (level_order.get(x['level'], 99), len(x['fr'])))

    for idx, item in enumerate(unique_pairs, start=1):
        item['id'] = idx
        if 'score_lexical' not in item and 'score' in item:
            item['score_lexical'] = float(item['score'])

    # Save to SQLite database via DataManager
    os.makedirs(DATA_DIR, exist_ok=True)
    dm = DataManager(db_path=DB_FILE)
    inserted_count = dm.bulk_insert_sentences(unique_pairs, clear_existing=True)

    # Optional JSON backup
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(unique_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n==========================================")
    print(f"🎉 Dataset generation complete!")
    print(f"Total sentences saved to database {DB_FILE}: {inserted_count}")
    print(f"Backup saved to {OUTPUT_FILE}")
    distribution = dm.get_level_distribution()
    for lvl in sorted(distribution.keys(), key=lambda k: level_order.get(k, 0)):
        print(f"  Level {lvl}: {distribution[lvl]} sentences")
    print(f"==========================================")

if __name__ == '__main__':
    main()
