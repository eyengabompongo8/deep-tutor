import json
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'sentences.json')

class DataManager:
    def __init__(self):
        self.sentences = self.load_data()

    def load_data(self):
        if not os.path.exists(DATA_FILE):
            return []
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_data(self):
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.sentences, f, ensure_ascii=False, indent=2)

    def get_all_sentences(self):
        return self.sentences

    def add_sentence(self, es, fr):
        new_id = max([s['id'] for s in self.sentences], default=0) + 1
        new_sentence = {'id': new_id, 'es': es, 'fr': fr}
        self.sentences.append(new_sentence)
        self.save_data()
        return new_sentence

    def update_sentence(self, item_id, es, fr):
        for idx, s in enumerate(self.sentences):
            if s['id'] == item_id:
                self.sentences[idx]['es'] = es
                self.sentences[idx]['fr'] = fr
                self.save_data()
                return True
        return False

    def delete_sentence(self, item_id):
        initial_length = len(self.sentences)
        self.sentences = [s for s in self.sentences if s['id'] != item_id]
        if len(self.sentences) < initial_length:
            self.save_data()
            return True
        return False
