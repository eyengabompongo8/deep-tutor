import os
import json
import bz2
import spacy
import sys

# Import our curriculum clusterer
sys.path.append(os.path.dirname(__file__))
from curriculum_parser import CurriculumParser
from spacy_clusterer import SpacyClusterer

def mine_tatoeba_sentences():
    corpus_path = os.path.join(os.path.dirname(__file__), 'fra_sentences.tsv.bz2')
    
    print("Loading curriculum rules...")
    parser = CurriculumParser()
    levels = parser.parse_all()
    clusterer = SpacyClusterer(levels)
    
    # Initialize quotas per topic (for ALL levels)
    topic_quotas = {}
    for level_id, level in levels.items():
        for mod in level['modules']:
            for topic in mod['topics']:
                topic_quotas[topic['id']] = 50 # Target 50 sentences per topic
                
    print(f"Hunting sentences to fill {len(topic_quotas)} topics from Tatoeba...")
    
    selected_sentences = set()
    
    with bz2.open(corpus_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                filled_topics = sum(1 for q in topic_quotas.values() if q <= 0)
                print(f"Scanned {i} lines. Filled {filled_topics}/{len(topic_quotas)} topics. Total queued sentences: {len(selected_sentences)}")
                
            if all(q <= 0 for q in topic_quotas.values()):
                print("All topics reached their quota!")
                break
                
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            fr_text = parts[2]
                
            if not fr_text or len(fr_text) < 15 or len(fr_text) > 150:
                continue
                
            topics = clusterer.cluster_sentence(fr_text)
            
            kept = False
            for t in topics:
                if t in topic_quotas and topic_quotas[t] > 0:
                    topic_quotas[t] -= 1
                    kept = True
                    
            if kept:
                selected_sentences.add(fr_text)
                
    queue_file = os.path.join(os.path.dirname(__file__), 'data', 'translation_queue.json')
    os.makedirs(os.path.dirname(queue_file), exist_ok=True)
    with open(queue_file, 'w', encoding='utf-8') as f:
        json.dump(list(selected_sentences), f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccessfully queued {len(selected_sentences)} French sentences to {queue_file}.")
    print("Unfilled topic quotas:")
    for t, q in topic_quotas.items():
        if q > 0:
            print(f"  {t}: {q} left")

if __name__ == "__main__":
    mine_tatoeba_sentences()
