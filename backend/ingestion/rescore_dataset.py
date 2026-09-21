import json
import os
from collections import Counter
from tqdm import tqdm

from backend.vocab_engine import VocabEngine
from backend.data_manager import DataManager, DB_PATH
from backend.db.connection import ROOT_DIR

JSON_BACKUP_PATH = os.path.join(ROOT_DIR, 'data', 'sentences.json')


def rescore_dataset(db_path=DB_PATH, json_backup=True, use_ml=True):
    dm = DataManager(db_path=db_path)
    sentences = dm.get_all_sentences()

    if not sentences:
        print(f"Error: No sentences found in database at {db_path}.")
        return

    print("Loading VocabEngine...")
    engine = VocabEngine()

    print(f"Loaded {len(sentences)} sentence pairs from database.")
    old_levels = Counter(s.get('level', 'Unknown') for s in sentences)
    print("Previous level distribution:", dict(old_levels))

    changed_count = 0
    new_levels = Counter()
    updates = []

    if use_ml:
        print("Computing CEFR predictions using ML model (batch inference)...")
        fr_texts = [s.get('fr', '') for s in sentences]
        ml_results = engine.score_sentences_ml_batch(fr_texts, batch_size=64)

        for item, ml_res in tqdm(zip(sentences, ml_results), total=len(sentences), desc="Processing scored sentences"):
            fr_text = item.get('fr', '')
            old_level = item.get('level')
            new_lvl = ml_res["level"]
            score_ml = float(ml_res["score"])
            _, lex_score, _ = engine.score_sentence(fr_text)

            if new_lvl != old_level:
                changed_count += 1

            item['level'] = new_lvl
            item['score_lexical'] = float(lex_score)
            item['score_ml'] = score_ml
            item['score'] = score_ml
            new_levels[new_lvl] += 1
            updates.append((new_lvl, float(lex_score), score_ml, item['id']))

        with dm._get_connection() as conn:
            conn.executemany(
                "UPDATE sentences SET level = ?, score_lexical = ?, score_ml = ? WHERE id = ?;",
                updates
            )
    else:
        for item in tqdm(sentences, desc="Rescoring sentences lexically"):
            fr_text = item.get('fr', '')
            old_level = item.get('level')
            new_lvl, new_score, _ = engine.score_sentence(fr_text)

            if new_lvl != old_level:
                changed_count += 1

            item['level'] = new_lvl
            item['score_lexical'] = float(new_score)
            item['score'] = float(new_score)
            new_levels[new_lvl] += 1
            updates.append((new_lvl, float(new_score), item['id']))

        with dm._get_connection() as conn:
            conn.executemany(
                "UPDATE sentences SET level = ?, score_lexical = ? WHERE id = ?;",
                updates
            )

    if json_backup:
        try:
            with open(JSON_BACKUP_PATH, 'w', encoding='utf-8') as f:
                json.dump(sentences, f, ensure_ascii=False, indent=2)
            print(f"JSON backup updated at {JSON_BACKUP_PATH}")
        except Exception as e:
            print(f"Warning: Could not write JSON backup: {e}")

    print(f"\n✅ Rescoring complete! {changed_count} of {len(sentences)} sentences updated in database.")
    print("New level distribution:")
    for lvl in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
        print(f"  {lvl}: {new_levels[lvl]}")


if __name__ == "__main__":
    rescore_dataset()
