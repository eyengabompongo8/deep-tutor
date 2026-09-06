import json
import os
import torch
import sys

# Append current dir and parent dir to sys.path to import models and data_manager
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import MLEngine
from data_manager import DataManager

class SentenceMiner:
    def __init__(self):
        self.output_file = os.path.join(os.path.dirname(__file__), 'data', 'authentic_pool.json')
        self.engine = MLEngine()
        
    def translate_to_french(self, es_text):
        """Translates authentic Spanish text to French using m2m100."""
        self.engine.nmt_tokenizer.src_lang = "es"
        inputs = self.engine.nmt_tokenizer(es_text, return_tensors="pt").to(self.engine.device)
        forced_bos_token_id = self.engine.nmt_tokenizer.get_lang_id("fr")
        
        with torch.no_grad():
            generated_tokens = self.engine.nmt_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id
            )
        fr_text = self.engine.nmt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return fr_text
        
    def translate_to_spanish(self, fr_text):
        """Translates advanced French text back to Spanish using m2m100."""
        self.engine.nmt_tokenizer.src_lang = "fr"
        inputs = self.engine.nmt_tokenizer(fr_text, return_tensors="pt", max_length=512, truncation=True).to(self.engine.device)
        forced_bos_token_id = self.engine.nmt_tokenizer.get_lang_id("es")
        
        with torch.no_grad():
            generated_tokens = self.engine.nmt_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512
            )
        es_text = self.engine.nmt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return es_text
        
    def process_translation_queue(self, results, start_id):
        queue_file = os.path.join(os.path.dirname(__file__), 'data', 'translation_queue.json')
        if not os.path.exists(queue_file):
            return start_id
            
        with open(queue_file, 'r', encoding='utf-8') as f:
            queue = json.load(f)
            
        print(f"Found translation queue with {len(queue)} French sentences. Translating to Spanish...")
        for i, fr in enumerate(queue):
            if i % 10 == 0:
                print(f"Translating {i}/{len(queue)} advanced sentences...")
            es = self.translate_to_spanish(fr)
            results.append({
                "id": start_id,
                "es": es,
                "fr": fr,
                "is_human_pair": False # French is human (Europarl), Spanish is NMT
            })
            start_id += 1
            
        # Optional: remove or rename the queue file so we don't translate it again next time
        os.rename(queue_file, queue_file + '.processed')
        return start_id

    def mine_sentences(self):
        results = []

        # Load existing sentences from SQLite database
        dm = DataManager()
        existing = dm.get_all_sentences()
        start_id = 1
        for item in existing:
            results.append({
                "id": start_id,
                "es": item['es'],
                "fr": item['fr'],
                "is_human_pair": True
            })
            start_id += 1
                    
        # Process any advanced sentences waiting for translation
        start_id = self.process_translation_queue(results, start_id)
                    
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Saved {len(results)} sentences to {self.output_file}")

if __name__ == "__main__":
    miner = SentenceMiner()
    miner.mine_sentences()
