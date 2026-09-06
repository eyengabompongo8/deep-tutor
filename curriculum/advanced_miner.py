import os
import json
import gzip
import urllib.request
import spacy
import sys

# Import our curriculum clusterer
sys.path.append(os.path.dirname(__file__))
from curriculum_parser import CurriculumParser
from spacy_clusterer import SpacyClusterer

def download_corpus():
    url = "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/mono/fr.txt.gz"
    output_path = os.path.join(os.path.dirname(__file__), 'globalvoices_fr.txt.gz')
    if not os.path.exists(output_path):
        print(f"Downloading GlobalVoices French corpus from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print("Download complete.")
    return output_path

def mine_advanced_sentences():
    corpus_path = download_corpus()
    
    print("Loading curriculum rules...")
    parser = CurriculumParser()
    levels = parser.parse_all()
    clusterer = SpacyClusterer(levels)
    
    # Initialize quotas per topic
    topic_quotas = {}
    for level_id, level in levels.items():
        if level_id in ['A1', 'A2']: continue
        for mod in level['modules']:
            for topic in mod['topics']:
                topic_quotas[topic['id']] = 50 # Target 50 sentences per advanced topic
                
    print(f"Hunting sentences to fill {len(topic_quotas)} advanced B1-C2 topics...")
    
    advanced_sentences = set() # using a set to avoid duplicating sentences
    
    with gzip.open(corpus_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                filled_topics = sum(1 for q in topic_quotas.values() if q <= 0)
                print(f"Scanned {i} lines. Filled {filled_topics}/{len(topic_quotas)} topics. Total queued sentences: {len(advanced_sentences)}")
                
            if all(q <= 0 for q in topic_quotas.values()):
                print("All advanced topics reached their quota!")
                break
                
            fr_text = line.strip()
            if not fr_text or len(fr_text) < 20 or len(fr_text) > 150:
                continue
                
            topics = clusterer.cluster_sentence(fr_text)
            
            kept = False
            for t in topics:
                if t in topic_quotas and topic_quotas[t] > 0:
                    topic_quotas[t] -= 1
                    kept = True
                    
            if kept:
                advanced_sentences.add(fr_text)
                
    queue_file = os.path.join(os.path.dirname(__file__), 'data', 'translation_queue.json')
    with open(queue_file, 'w', encoding='utf-8') as f:
        json.dump(list(advanced_sentences), f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccessfully queued {len(advanced_sentences)} advanced French sentences to {queue_file}.")
    print("Next step: Run the translation model to generate Spanish pairs.")

if __name__ == "__main__":
    mine_advanced_sentences()
