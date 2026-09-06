import os
import sys
import glob

sys.path.append(os.path.dirname(__file__))
from curriculum_parser import CurriculumParser
from spacy_clusterer import SpacyClusterer

def main():
    parser = CurriculumParser()
    levels = parser.parse_all()
    
    print("Loading spaCy and compiling patterns...")
    clusterer = SpacyClusterer(levels)
    
    total_patterns = 0
    failed_patterns = []
    
    # Read files manually to get tables
    curriculum_dir = os.path.join(os.path.dirname(__file__), 'curriculum')
    files = glob.glob(os.path.join(curriculum_dir, '*.md'))
    
    # Map topic -> sentences
    topic_sentences = {}
    
    for filepath in files:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        current_topic = None
        in_table = False
        
        for line in lines:
            if line.startswith("### Topic ID:"):
                current_topic = line.split(":", 1)[1].strip()
                topic_sentences[current_topic] = []
            elif line.startswith("#### Micro-topic"):
                # E.g. #### Micro-topic A2.1.1: 
                parts = line.split(":")
                current_topic = parts[0].replace("#### Micro-topic", "").strip()
                topic_sentences[current_topic] = []
                
            if current_topic:
                if '| French Sentence |' in line:
                    in_table = True
                    continue
                if in_table and (line.startswith('|---') or line.startswith('| :---') or line.startswith('|:---')):
                    continue
                if in_table and line.startswith('|'):
                    parts = line.split('|')
                    if len(parts) >= 2:
                        fr_sentence = parts[1].strip()
                        # Clean markdown asterisks/formatting
                        fr_sentence = fr_sentence.strip('*_` \t"\'')
                        if fr_sentence and not fr_sentence.startswith('---') and not fr_sentence.startswith(':---') and fr_sentence != 'French Sentence':
                            topic_sentences[current_topic].append(fr_sentence)
                elif in_table and not line.strip():
                    in_table = False

    for level_id, level in levels.items():
        for mod in level['modules']:
            for topic in mod['topics']:
                if not topic.get('spacy_pattern'):
                    continue
                    
                total_patterns += 1
                topic_id = topic['id']
                sentences = topic_sentences.get(topic_id, [])
                
                if not sentences:
                    print(f"[{topic_id}] WARNING: No canonical sentences found to test.")
                    continue
                    
                matched_sentences = 0
                failed_sentences = []
                for sent in sentences:
                    matches = clusterer.cluster_sentence(sent)
                    if topic_id in matches:
                        matched_sentences += 1
                    else:
                        failed_sentences.append(sent)
                        
                if matched_sentences == 0:
                    failed_patterns.append((topic_id, sentences))
                    print(f"[{topic_id}] FAIL: Pattern failed to match ANY of its own canonical sentences!")
                    for s in sentences:
                        print(f"  - {s}")
                elif failed_sentences:
                    print(f"[{topic_id}] PARTIAL FAIL: Missed {len(failed_sentences)} sentences:")
                    for s in failed_sentences:
                        print(f"  - {s}")
                else:
                    print(f"[{topic_id}] PASS (Matched all {len(sentences)})")
                    
    print("\n--- SUMMARY ---")
    print(f"Total Patterns Tested: {total_patterns}")
    print(f"Failed Patterns (0 matches): {len(failed_patterns)}")
    
    if failed_patterns:
        print("\nFailed Topic IDs:")
        for topic_id, _ in failed_patterns:
            print(f" - {topic_id}")

if __name__ == "__main__":
    main()
