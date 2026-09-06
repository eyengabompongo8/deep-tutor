import os
import re
import json

CURRICULUM_DIR = os.path.join(os.path.dirname(__file__), 'curriculum')

class CurriculumParser:
    def __init__(self):
        self.levels = {}

    def parse_all(self):
        for filename in sorted(os.listdir(CURRICULUM_DIR)):
            if filename.endswith(".md"):
                self.parse_file(os.path.join(CURRICULUM_DIR, filename))
        return self.levels

    def parse_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_level = None
        current_module = None
        current_topic = None
        
        in_code_block = False
        code_block_content = []

        level_pattern = re.compile(r'^# CEFR Level:\s*(.+)', re.IGNORECASE)
        # Handle "CEFR A2 French Curriculum Specification" which is in A2.md
        level_alt_pattern = re.compile(r'^# CEFR ([A-Z1-2_]+)', re.IGNORECASE)
        module_pattern = re.compile(r'^#+\s+Module\s+([A-Z0-9_\.]+):\s*(.+)', re.IGNORECASE)
        topic_pattern = re.compile(r'^### Topic ID:\s*(.+)', re.IGNORECASE)
        micro_topic_pattern = re.compile(r'^#### Micro-topic\s+([A-Z0-9_\.]+):\s*(.+)', re.IGNORECASE)
        title_pattern = re.compile(r'^- \*\*Title:\*\*\s*(.+)', re.IGNORECASE)
        obj_pattern = re.compile(r'^- \*\*Pedagogical Objective:\*\*\s*(.+)', re.IGNORECASE)

        for line in lines:
            line_str = line.strip()
            
            # CEFR Level
            m_level = level_pattern.match(line_str)
            m_alt = level_alt_pattern.match(line_str)
            if m_level or m_alt:
                if m_level:
                    level_id = m_level.group(1).split()[0]  # e.g. "A1", "C1_C2"
                else:
                    level_id = m_alt.group(1)
                
                # if level_id is parsed like "Level:" or similar, clean it up
                level_id = level_id.replace("Level:", "").strip()
                
                if level_id not in self.levels:
                    current_level = {"id": level_id, "modules": []}
                    self.levels[level_id] = current_level
                else:
                    current_level = self.levels[level_id]
                current_module = None
                current_topic = None
                continue

            # Module
            m_mod = module_pattern.match(line_str)
            if m_mod and current_level:
                mod_id = m_mod.group(1).strip()
                mod_name = m_mod.group(2).strip()
                current_module = {"id": mod_id, "name": mod_name, "topics": []}
                current_level["modules"].append(current_module)
                current_topic = None
                continue

            # Topic ID
            m_topic = topic_pattern.match(line_str)
            if m_topic and current_module:
                topic_id = m_topic.group(1).strip()
                current_topic = {"id": topic_id, "title": "", "objective": "", "spacy_pattern": ""}
                current_module["topics"].append(current_topic)
                continue

            # Micro-topic (A2 style)
            m_micro = micro_topic_pattern.match(line_str)
            if m_micro and current_module:
                topic_id = m_micro.group(1).strip()
                title = m_micro.group(2).strip()
                current_topic = {"id": topic_id, "title": title, "objective": "", "spacy_pattern": ""}
                current_module["topics"].append(current_topic)
                continue

            # Topic Title (A1 style)
            if current_topic and not current_topic["title"]:
                m_title = title_pattern.match(line_str)
                if m_title:
                    current_topic["title"] = m_title.group(1).strip()

            # Pedagogical Objective
            if current_topic:
                m_obj = obj_pattern.match(line_str)
                if m_obj:
                    current_topic["objective"] = m_obj.group(1).strip()

            # spaCy Pattern extraction
            if current_topic:
                if line_str.startswith("```python"):
                    in_code_block = True
                    code_block_content = []
                elif line_str.startswith("```") and in_code_block:
                    in_code_block = False
                    code_str = "\n".join(code_block_content)
                    if "pattern" in code_str or "{" in code_str:
                        current_topic["spacy_pattern"] = code_str
                elif in_code_block:
                    code_block_content.append(line.rstrip())

if __name__ == "__main__":
    parser = CurriculumParser()
    levels = parser.parse_all()
    print(f"Parsed {len(levels)} levels.")
    for level_id, level_data in levels.items():
        print(f"Level {level_id}: {len(level_data['modules'])} modules")
        for mod in level_data['modules']:
            print(f"  - Module {mod['id']}: {len(mod['topics'])} topics")
