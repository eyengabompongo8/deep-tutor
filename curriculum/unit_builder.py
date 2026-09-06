import json
import os
from curriculum_parser import CurriculumParser
from spacy_clusterer import SpacyClusterer
from vocab_engine import VocabEngine
from sentence_miner import SentenceMiner

class UnitBuilder:
    def __init__(self):
        self.output_file = os.path.join(os.path.dirname(__file__), 'data', 'units.json')
        self.authentic_pool_file = os.path.join(os.path.dirname(__file__), 'data', 'authentic_pool.json')

    def build_units(self):
        print("1. Parsing Curriculum...")
        parser = CurriculumParser()
        levels = parser.parse_all()

        print("2. Preparing Authentic Sentences...")
        if not os.path.exists(self.authentic_pool_file):
            print("   Mining authentic sentences (this may take a moment)...")
            miner = SentenceMiner()
            miner.mine_sentences()
            
        with open(self.authentic_pool_file, 'r', encoding='utf-8') as f:
            authentic_sentences = json.load(f)
            
        print(f"   Loaded {len(authentic_sentences)} authentic sentence pairs.")

        print("3. Initializing Engines (spaCy + FLELex)...")
        clusterer = SpacyClusterer(levels)
        vocab_engine = VocabEngine()

        print("4. Clustering & Profiling Exercises...")
        # Map topic_id -> list of processed sentences
        topic_exercises = {}

        for sentence in authentic_sentences:
            fr_text = sentence['fr']
            matched_topics = clusterer.cluster_sentence(fr_text)
            
            if not matched_topics:
                continue
                
            lvl, flelex_score = vocab_engine.score_sentence(fr_text)
            sentence['max_cefr_level'] = lvl
            sentence['flelex_score'] = flelex_score
            
            for topic_id in matched_topics:
                if topic_id not in topic_exercises:
                    topic_exercises[topic_id] = []
                topic_exercises[topic_id].append(sentence)

        print("5. Sequencing Exercises by Progressive Vocabulary...")
        for topic_id, exercises in topic_exercises.items():
            # Sort by FLELex score ascending (lower score = higher frequency/easier words)
            exercises.sort(key=lambda x: x['flelex_score'])
            
        print("6. Compiling Final Unit Structure...")
        # Inject the sorted exercises back into the curriculum tree
        output_dir = os.path.join(os.path.dirname(__file__), 'data', 'units')
        
        for level_id, level in levels.items():
            safe_level_id = level_id.replace('/', '_')
            level_dir = os.path.join(output_dir, safe_level_id)
            os.makedirs(level_dir, exist_ok=True)
            
            for mod in level['modules']:
                for topic in mod['topics']:
                    topic_id = topic['id']
                    # Attach exercises if any were clustered here
                    topic['exercises'] = topic_exercises.get(topic_id, [])
                    
                # Save this module to its own JSON file
                mod_filename = mod['id'].replace(' ', '_').replace('/', '_') + '.json'
                mod_path = os.path.join(level_dir, mod_filename)
                
                with open(mod_path, 'w', encoding='utf-8') as f:
                    json.dump(mod, f, ensure_ascii=False, indent=2)

        print(f"Successfully built and exported Curriculum Units per module into {output_dir}!")

if __name__ == "__main__":
    builder = UnitBuilder()
    builder.build_units()
