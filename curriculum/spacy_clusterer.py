import spacy
from spacy.matcher import Matcher, DependencyMatcher
import json
import sys

class SpacyClusterer:
    def __init__(self, curriculum_levels):
        self.curriculum_levels = curriculum_levels
        try:
            self.nlp = spacy.load('fr_core_news_sm')
        except OSError:
            import os
            os.system(f'{sys.executable} -m spacy download fr_core_news_sm')
            self.nlp = spacy.load('fr_core_news_sm')
            
        self.matcher = Matcher(self.nlp.vocab)
        self.dep_matcher = DependencyMatcher(self.nlp.vocab)
        self.compile_patterns()

    def _extract_patterns_from_scope(self, local_scope):
        """Extract all valid spaCy pattern lists from an executed code snippet's local scope."""
        patterns = []
        for var_name, val in local_scope.items():
            if var_name.startswith('_'):
                continue
            if isinstance(val, list) and len(val) > 0:
                # Check if it is a list of token dicts (single pattern)
                if isinstance(val[0], dict):
                    patterns.append(val)
                # Check if it is a list of lists of token dicts (multiple patterns)
                elif isinstance(val[0], list) and len(val[0]) > 0 and isinstance(val[0][0], dict):
                    for sub_pat in val:
                        patterns.append(sub_pat)
        return patterns

    def compile_patterns(self):
        """Compiles the string spaCy patterns from curriculum into Matcher / DependencyMatcher."""
        count = 0
        dep_count = 0
        for level_id, level in self.curriculum_levels.items():
            for mod in level.get('modules', []):
                for topic in mod.get('topics', []):
                    pattern_code = topic.get('spacy_pattern', '')
                    if not pattern_code:
                        continue
                    try:
                        local_scope = {}
                        exec(pattern_code, {}, local_scope)
                        extracted_patterns = self._extract_patterns_from_scope(local_scope)

                        for pat in extracted_patterns:
                            is_dep = any("RIGHT_ID" in token for token in pat)
                            if is_dep:
                                self.dep_matcher.add(topic['id'], [pat])
                                dep_count += 1
                            else:
                                self.matcher.add(topic['id'], [pat])
                                count += 1
                    except Exception as e:
                        print(f"Failed to compile pattern for {topic['id']}: {e}")
        print(f"Compiled {count} standard spaCy patterns and {dep_count} Dependency patterns across curriculum.")

    def cluster_sentence(self, fr_text):
        """Returns a list of topic IDs that match the syntactic structure of the sentence."""
        if not fr_text or not fr_text.strip():
            return []
        doc = self.nlp(fr_text)
        matched_topic_ids = set()

        try:
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                string_id = self.nlp.vocab.strings[match_id]
                matched_topic_ids.add(string_id)
        except Exception as e:
            pass

        try:
            dep_matches = self.dep_matcher(doc)
            for match_id, token_ids in dep_matches:
                string_id = self.nlp.vocab.strings[match_id]
                matched_topic_ids.add(string_id)
        except Exception as e:
            pass

        return list(matched_topic_ids)

if __name__ == "__main__":
    from curriculum_parser import CurriculumParser
    parser = CurriculumParser()
    levels = parser.parse_all()
    
    clusterer = SpacyClusterer(levels)
    test_sentence = "Je me lave les mains."
    topics = clusterer.cluster_sentence(test_sentence)
    print(f"Sentence: '{test_sentence}' matches topics: {topics}")
