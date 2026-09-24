import os
from typing import List, Dict, Any
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from backend.db import DataManager
from backend.nlp.vocab_engine import VocabEngine
from backend.ingestion.pipeline.mine_tatoeba import is_valid_sentence, clean_sentence
from backend.ingestion.pipeline.common import translate_batch

def run_mine_hf_dataset() -> List[Dict[str, Any]]:
    print("\n--- Mining HuggingFace Dataset (UniversalCEFR/readme_fr) ---")
    
    print("Loading dataset from HuggingFace...")
    dataset_dict = load_dataset("UniversalCEFR/readme_fr")
    
    # Concatenate all available splits
    all_splits = [dataset_dict[split] for split in dataset_dict.keys()]
    dataset = concatenate_datasets(all_splits)
    
    print(f"Total raw sentences in dataset: {len(dataset)}")
    
    print("Loading existing database sentences for deduplication...")
    dm = DataManager()
    existing_fr_sentences = {s['fr'].lower() for s in dm.get_all_sentences()}
    
    print("Filtering and deduplicating candidate sentences...")
    valid_new_sentences = []
    seen_in_batch = set()
    
    for row in tqdm(dataset, desc="Processing rows"):
        raw_text = row.get("text", "")
        if not raw_text:
            continue
            
        fr = clean_sentence(raw_text)
        
        # We allow up to 25 words (same as Tatoeba)
        if not is_valid_sentence(fr):
            continue
            
        fr_lower = fr.lower()
        if fr_lower in existing_fr_sentences or fr_lower in seen_in_batch:
            continue
            
        seen_in_batch.add(fr_lower)
        valid_new_sentences.append(fr)
        
    print(f"Found {len(valid_new_sentences)} new valid sentences after deduplication.")
    
    if not valid_new_sentences:
        return []
        
    print("\nCalculating CEFR Levels using ML Classifier...")
    engine = VocabEngine()
    ml_results = engine.score_sentences_ml_batch(valid_new_sentences, batch_size=64)
    
    print(f"\nTranslating {len(valid_new_sentences)} sentences to Spanish...")
    translated_texts = translate_batch(valid_new_sentences)
    
    records = []
    for fr, ml_res, es in zip(valid_new_sentences, ml_results, translated_texts):
        lvl = ml_res["level"]
        score_ml = float(ml_res["score"])
        
        # Get lexical score for progressive vocabulary
        _, lex_sc, _ = engine.score_sentence(fr)
        
        records.append({
            "fr": fr,
            "es": es,
            "level": lvl,
            "score": score_ml,
            "score_lexical": float(lex_sc),
            "score_ml": score_ml,
            "source": "hf_readme_fr"
        })
        
    print(f"Successfully mined and prepared {len(records)} sentences from HuggingFace.")
    return records

if __name__ == "__main__":
    run_mine_hf_dataset()
