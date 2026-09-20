import json
import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

from backend.db.connection import ROOT_DIR


def translate_queue():
    queue_file = os.path.join(ROOT_DIR, 'data', 'translation_queue.json')
    output_file = os.path.join(ROOT_DIR, 'data', 'translated_sentences.json')
    
    if not os.path.exists(queue_file):
        print(f"No queue file found at {queue_file}")
        return
        
    with open(queue_file, 'r', encoding='utf-8') as f:
        sentences_to_translate = json.load(f)
        
    print(f"Loaded {len(sentences_to_translate)} sentences to translate.")
    
    model_name = 'Helsinki-NLP/opus-mt-fr-es'
    print(f"Loading {model_name}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}.")
    
    batch_size = 16
    results = {}
    
    for i in tqdm(range(0, len(sentences_to_translate), batch_size)):
        batch = sentences_to_translate[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Translate
        with torch.no_grad():
            translated = model.generate(**inputs)
            
        # Decode
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        for fr_text, es_text in zip(batch, translated_texts):
            results[fr_text] = es_text
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully translated {len(results)} sentences and saved to {output_file}")


if __name__ == "__main__":
    translate_queue()
