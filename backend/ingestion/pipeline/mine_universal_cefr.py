from typing import List, Dict, Any, Optional
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from backend.ingestion.pipeline.common import format_sentence, is_valid_sentence


def run_mine_universal_cefr(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    print("\n--- Mining UniversalCEFR Dataset (UniversalCEFR/readme_fr) ---")
    
    print("Loading dataset from HuggingFace...")
    dataset_dict = load_dataset("UniversalCEFR/readme_fr")
    
    # Concatenate all available splits
    all_splits = [dataset_dict[split] for split in dataset_dict.keys()]
    dataset = concatenate_datasets(all_splits)
    
    print(f"Total raw sentences in dataset: {len(dataset)}")
    
    print("Extracting candidate sentences...")
    valid_new_sentences = []
    seen_in_batch = set()
    
    for row in tqdm(dataset, desc="Processing rows"):
        raw_text = row.get("text", "")
        if not raw_text:
            continue
            
        fr = format_sentence(raw_text)

        if not is_valid_sentence(fr):
            continue
            
        fr_lower = fr.lower()
        if fr_lower in seen_in_batch:
            continue
            
        seen_in_batch.add(fr_lower)
        valid_new_sentences.append({
            "fr": fr,
            "es": None,
            "source": "hf_universal_cefr",
            "metadata": {}
        })
        if limit is not None and len(valid_new_sentences) >= limit:
            break
        
    print(f"Extracted {len(valid_new_sentences)} candidate sentences from UniversalCEFR.")
    return valid_new_sentences


if __name__ == "__main__":
    run_mine_universal_cefr()
