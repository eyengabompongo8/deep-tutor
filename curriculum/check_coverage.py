import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from curriculum_parser import CurriculumParser
from spacy_clusterer import SpacyClusterer

def main():
    with open('data/translation_queue.json', 'r') as f:
        queue = json.load(f)
        
    parser = CurriculumParser()
    levels = parser.parse_all()
    clusterer = SpacyClusterer(levels)
    
    # Initialize counts for all B1+ topics
    topic_counts = {}
    for level_id, level in levels.items():
        if level_id in ['A1', 'A2']: continue
        for mod in level['modules']:
            for topic in mod['topics']:
                topic_counts[topic['id']] = 0
                
    # Cluster queued sentences
    for fr_text in queue:
        topics = clusterer.cluster_sentence(fr_text)
        for t in topics:
            if t in topic_counts:
                topic_counts[t] += 1
                
    # Report
    empty = 0
    filled = 0
    for topic, count in topic_counts.items():
        if count == 0:
            empty += 1
            print(f"EMPTY: {topic}")
        else:
            filled += 1
            
    print(f"\nOut of {len(topic_counts)} advanced topics (B1-C2):")
    print(f"Filled: {filled}")
    print(f"Empty: {empty}")

if __name__ == "__main__":
    main()
