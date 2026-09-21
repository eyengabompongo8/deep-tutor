import os
import sys
import csv
import re
import threading
import spacy
import urllib.request
import zipfile
import io
from typing import List, Dict, Any, Optional, Set, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
FLELEX_PATH = os.path.join(DATA_DIR, 'flelex_beacco.tsv')
LEXIQUE_DIR = os.path.join(DATA_DIR, 'lexique')
LEXIQUE_FILE = os.path.join(LEXIQUE_DIR, 'Lexique383.tsv')

# CEFR level mappings
LEVELS_VAL = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
VAL_TO_LEVEL = {1: 'A1', 2: 'A2', 3: 'B1', 4: 'B2', 5: 'C1', 6: 'C2'}

from backend.config import DEFAULT_UNLOCKED_WORDS as A1_WORDS

# POS Tag mapping from spaCy POS to Lexique383 cgram
SPACY_TO_LEXIQUE_POS = {
    'VERB': ['VER', 'AUX'],
    'AUX': ['AUX', 'VER'],
    'NOUN': ['NOM'],
    'ADJ': ['ADJ', 'ADJ:num', 'ADJ:pos', 'ADJ:dem', 'ADJ:ind'],
    'ADV': ['ADV'],
    'PRON': ['PRO:per', 'PRO:pos', 'PRO:dem', 'PRO:ind', 'PRO:rel', 'PRO'],
    'DET': ['ART:def', 'ART:ind', 'ART', 'ADJ:pos', 'ADJ:dem'],
    'ADP': ['PRE'],
    'CCONJ': ['CON'],
    'SCONJ': ['CON'],
    'NUM': ['ADJ:num', 'NOM']
}


class VocabEngine:
    def __init__(self):
        self.flelex_data = {}        # lemma -> {'level': str, 'score': int, 'freq': float}
        self.lexique_pos_lemma = {}  # (ortho, cgram) -> (lemme, freq)
        self.lexique_best_lemma = {} # ortho -> (lemme, freq)

        self._ensure_resources()
        self._load_lexique()
        self._load_flelex()

        try:
            self.nlp = spacy.load('fr_core_news_sm')
        except OSError:
            print("Downloading spaCy fr_core_news_sm model...")
            os.system(f'{sys.executable} -m spacy download fr_core_news_sm')
            self.nlp = spacy.load('fr_core_news_sm')

    def _ensure_resources(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(LEXIQUE_DIR, exist_ok=True)

        if not os.path.exists(FLELEX_PATH):
            print("Downloading FLELex / CEFRLex (Beacco)...")
            url = 'https://cental.uclouvain.be/cefrlex/static/resources/fr/FleLex_TT_Beacco.tsv'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(FLELEX_PATH, 'wb') as f:
                f.write(resp.read())
            print("FLELex dataset downloaded.")

        if not os.path.exists(LEXIQUE_FILE):
            print("Downloading Lexique383 dataset...")
            url = 'http://www.lexique.org/databases/Lexique383/Lexique383.zip'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp:
                with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
                    z.extractall(LEXIQUE_DIR)
            print("Lexique383 dataset downloaded and extracted.")

    def _load_lexique(self):
        if not os.path.exists(LEXIQUE_FILE):
            return

        with open(LEXIQUE_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                ortho = row['ortho'].lower()
                lemme = row['lemme'].lower()
                cgram = row['cgram']
                try:
                    freq = float(row.get('freqlemlivres', 0) or 0)
                except (ValueError, TypeError):
                    freq = 0.0

                key = (ortho, cgram)
                if key not in self.lexique_pos_lemma or freq > self.lexique_pos_lemma[key][1]:
                    self.lexique_pos_lemma[key] = (lemme, freq)

                if ortho not in self.lexique_best_lemma or freq > self.lexique_best_lemma[ortho][1]:
                    self.lexique_best_lemma[ortho] = (lemme, freq)

    def _load_flelex(self):
        if not os.path.exists(FLELEX_PATH):
            return

        with open(FLELEX_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                word = row['word'].lower()
                try:
                    freq = float(row.get('freq_total', 0) or 0)
                except (ValueError, TypeError):
                    freq = 0.0

                cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
                level_freqs = {}
                total_cefr_freq = 0.0
                for lvl in cefr_levels:
                    try:
                        lvl_freq = float(row.get(f'freq_{lvl}', 0) or 0)
                    except (ValueError, TypeError):
                        lvl_freq = 0.0
                    level_freqs[lvl] = lvl_freq
                    total_cefr_freq += lvl_freq
                
                assigned_lvl = 'C2'
                if total_cefr_freq > 0:
                    cum = 0.0
                    for lvl in cefr_levels:
                        cum += (level_freqs[lvl] / total_cefr_freq)
                        if cum >= 0.20:
                            assigned_lvl = lvl
                            break
                else:
                    assigned_lvl = row.get('level', 'C2').upper()
                    if assigned_lvl not in LEVELS_VAL:
                        assigned_lvl = 'C2'

                self.flelex_data[word] = {
                    'level': assigned_lvl,
                    'score': LEVELS_VAL[assigned_lvl],
                    'freq': freq
                }

    def get_word_level(self, tok):
        """
        Computes the CEFR level for an individual spaCy token.
        Returns: (cefr_level: str, score_val: int, lemma: str)
        """
        if tok.is_punct or tok.is_space or tok.is_digit:
            return None, None, None

        text_lower = tok.text.lower()
        if text_lower in A1_WORDS or tok.pos_ == 'PROPN':
            return 'A1', 1, text_lower

        # Find lemma using POS matching
        target_cgrams = SPACY_TO_LEXIQUE_POS.get(tok.pos_, [])
        lemma = None
        for cg in target_cgrams:
            if (text_lower, cg) in self.lexique_pos_lemma:
                lemma = self.lexique_pos_lemma[(text_lower, cg)][0]
                break

        if not lemma:
            if text_lower in self.lexique_best_lemma:
                lemma = self.lexique_best_lemma[text_lower][0]
            else:
                lemma = tok.lemma_.lower()

        if lemma in A1_WORDS:
            return 'A1', 1, lemma

        # Check FLELex by lemma
        if lemma in self.flelex_data:
            info = self.flelex_data[lemma]
            return info['level'], info['score'], lemma

        # Check FLELex by surface text
        if text_lower in self.flelex_data:
            info = self.flelex_data[text_lower]
            return info['level'], info['score'], text_lower

        # Check spaCy's native lemma
        spacy_lemma = tok.lemma_.lower()
        if spacy_lemma in self.flelex_data:
            info = self.flelex_data[spacy_lemma]
            return info['level'], info['score'], spacy_lemma

        # Word not found in FLELex dictionary or A1 list: ignore the word
        return None, None, None

    def score_sentence(self, fr_text):
        """
        Scores a French sentence according to FLELex / CEFRLex.
        The sentence score is the MAXIMUM score that a word from the sentence has.
        Returns:
            final_level (str): 'A1', 'A2', 'B1', 'B2', 'C1', or 'C2'
            max_score (int): 1 to 6
            word_breakdown (list[dict]): Per-word diagnostics
        """
        if not fr_text or not fr_text.strip():
            return 'A1', 1, []

        doc = self.nlp(fr_text.strip())
        max_score = 1
        word_breakdown = []

        for tok in doc:
            lvl, score, lemma = self.get_word_level(tok)
            if lvl is None:
                continue

            if score > max_score:
                max_score = score

            word_breakdown.append({
                'word': tok.text,
                'lemma': lemma,
                'pos': tok.pos_,
                'level': lvl,
                'score': score
            })

        final_level = VAL_TO_LEVEL.get(max_score, 'C2')
        return final_level, max_score, word_breakdown

    def get_cefr_classifier(self):
        """Lazy loader for the UniversalCEFR classifier."""
        if not hasattr(self, '_cefr_classifier') or self._cefr_classifier is None:
            from backend.models import resolve_device_and_dtype
            from backend.model_registry import registry
            device, dtype = resolve_device_and_dtype()
            self._cefr_classifier = registry.get_cefr("universal_cefr", device, dtype)
        return self._cefr_classifier

    def score_sentence_ml(self, fr_text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Scores a French sentence using the sequence classification ML model.
        Returns:
            level (str): 'A1' | 'A2' | 'B1' | 'B2' | 'C1' | 'C2'
            score_ml (float): 1.0 to 6.0 continuous expectation score
            probabilities (dict[str, float]): Class probabilities per CEFR level
        """
        if not fr_text or not fr_text.strip():
            return 'A1', 1.0, {'A1': 1.0}

        classifier = self.get_cefr_classifier()
        res = classifier.predict([fr_text.strip()])[0]
        return res["level"], res["score"], res["probabilities"]

    def score_sentences_ml_batch(self, fr_texts: List[str], batch_size: int = 64, show_progress: bool = True, desc: str = "Scoring CEFR (ML)") -> List[Dict[str, Any]]:
        """
        Batch scores multiple French sentences using the CEFR ML model for high ingestion throughput.
        """
        if not fr_texts:
            return []

        classifier = self.get_cefr_classifier()
        all_results = []
        
        try:
            from tqdm import tqdm
            iterator = range(0, len(fr_texts), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc=desc, total=(len(fr_texts) + batch_size - 1) // batch_size)
        except ImportError:
            iterator = range(0, len(fr_texts), batch_size)

        for i in iterator:
            chunk = fr_texts[i:i + batch_size]
            cleaned_chunk = [t.strip() if t and t.strip() else "Bonjour" for t in chunk]
            results = classifier.predict(cleaned_chunk)
            all_results.extend(results)
        return all_results

    def score_sentence_hybrid(self, fr_text: str) -> Tuple[str, float, float, List[Dict[str, Any]]]:
        """
        Computes both lexical and ML scores for a sentence.
        Returns:
            assigned_level (str): ML predicted CEFR level
            score_ml (float): ML expectation score
            score_lexical (float): Maximum lexical word score
            word_breakdown (list[dict]): Per-word lexical diagnostics
        """
        lex_lvl, lex_score, breakdown = self.score_sentence(fr_text)
        ml_lvl, ml_score, _ = self.score_sentence_ml(fr_text)
        return ml_lvl, ml_score, float(lex_score), breakdown

    def extract_sentence_lexemas(self, fr_text: str) -> List[Dict[str, Any]]:
        """
        Extracts canonical word roots (lexemas) from a French sentence text.
        Returns a list of unique lexema dicts:
        [{'lemma': str, 'pos': str, 'level': str, 'freq': float, 'is_base': bool}]
        """
        if not fr_text or not fr_text.strip():
            return []

        doc = self.nlp(fr_text.strip())
        seen = set()
        lexemas = []

        for tok in doc:
            if tok.is_punct or tok.is_space or tok.is_digit or tok.pos_ in ('PUNCT', 'SYM', 'SPACE', 'NUM', 'X'):
                continue

            lvl, score, lemma = self.get_word_level(tok)
            if not lemma:
                lemma = tok.lemma_.lower().strip()
            lemma = lemma.lower().strip()

            # Clean contractions like l', d', j' from lemma
            if lemma.endswith("'") or lemma.endswith("’"):
                continue

            pos = tok.pos_
            key = (lemma, pos)
            if key in seen:
                continue
            seen.add(key)

            # Determine level
            if not lvl or lvl not in LEVELS_VAL:
                if lemma in self.flelex_data:
                    lvl = self.flelex_data[lemma]['level']
                else:
                    lvl = 'A1' if lemma in A1_WORDS else 'A2'

            # Determine frequency from Lexique / FLELex
            freq = 0.0
            if lemma in self.lexique_best_lemma:
                freq = self.lexique_best_lemma[lemma][1]
            elif lemma in self.flelex_data:
                freq = self.flelex_data[lemma]['freq']

            # Determine if foundational base word (unlocked by default)
            is_base = (
                lemma in A1_WORDS or
                pos in ('DET', 'PRON', 'ADP', 'CCONJ', 'SCONJ', 'AUX') or
                (lvl == 'A1' and lemma in ('être', 'avoir', 'aller', 'faire', 'dire', 'pouvoir', 'vouloir', 'voir', 'savoir'))
            )

            lexemas.append({
                'lemma': lemma,
                'pos': pos,
                'level': lvl,
                'freq': freq,
                'is_base': bool(is_base)
            })

        return lexemas

    def get_token_lemmas(self, fr_text: str) -> List[Tuple[str, str, str, str]]:
        """
        Tokenizes French text and returns a list of (raw_token_text, clean_word, lemma, spacy_lemma) tuples
        for all content and functional words.
        """
        if not fr_text or not fr_text.strip():
            return []
        doc = self.nlp(fr_text.strip())
        results = []
        for tok in doc:
            if tok.is_punct or tok.is_space or tok.is_digit or tok.pos_ in ('PUNCT', 'SYM', 'SPACE', 'NUM', 'X'):
                continue
            lvl, score, lemma = self.get_word_level(tok)
            if not lemma:
                lemma = tok.lemma_.lower().strip()
            lemma = lemma.lower().strip().rstrip("'’")
            spacy_lemma = tok.lemma_.lower().strip().rstrip("'’")
            if not lemma and not spacy_lemma:
                continue
            raw_text = tok.text
            clean_word = re.sub(r"^[ldjcmnstLDJCMNST]['’]", "", raw_text)
            results.append((raw_text, clean_word, lemma or spacy_lemma, spacy_lemma))
        return results

    def extract_sentence_token_pairs(self, fr_text: str, target_lemmas: List[str]) -> List[Dict[str, str]]:
        """
        For a given French sentence and a list of target lexemas (lemmas),
        finds the actual surface derivation in the sentence matching each target lexema.
        Returns: [{'word': surface_word, 'lemma': target_lemma}, ...]
        """
        if not target_lemmas:
            return []
        token_info = self.get_token_lemmas(fr_text)
        pairs = []
        matched_tokens = set()

        def _norm(s: str) -> str:
            if not s:
                return ""
            s = s.lower().replace('œ', 'oe').replace('æ', 'ae').strip()
            s = re.sub(r"^(se\s+|s['’]|l['’]|d['’]|j['’]|c['’]|m['’]|n['’]|t['’])", "", s)
            return s.strip().rstrip("'’")

        def _stem_match(a: str, b: str) -> bool:
            a_clean = _norm(a)
            b_clean = _norm(b)
            if not a_clean or not b_clean:
                return False
            if a_clean == b_clean:
                return True
            suf_pattern = r"(issant|issante|issants|issantes|issiez|issons|issent|isse|isses|ant|ante|ants|antes|ée|ées|és|é|er|ir|re|oir|ons|ez|ait|ais|aient|as|âmes|âtes|èrent|ât|ît|us|ue|ues|ut|ûmes|ûtes|urent|ie|ies|it|ît|euse|euses|eur|eurs|eux|trice|trices|teur|teurs|es|e|ent)$"
            stem_a = re.sub(suf_pattern, "", a_clean)
            stem_b = re.sub(suf_pattern, "", b_clean)
            if len(stem_a) >= 3 and len(stem_b) >= 3 and stem_a == stem_b:
                return True
            min_len = min(len(a_clean), len(b_clean))
            if min_len >= 4 and a_clean[:4] == b_clean[:4] and abs(len(a_clean) - len(b_clean)) <= 4:
                return True
            return False

        for target in target_lemmas:
            tgt_clean = _norm(target)
            matched_word = target
            found = False

            # 1. Search for a token whose resolved lemma matches target lexema
            for idx, (raw_word, clean_word, tok_lemma, spacy_lemma) in enumerate(token_info):
                if idx not in matched_tokens:
                    if _norm(tok_lemma) == tgt_clean or _norm(spacy_lemma) == tgt_clean:
                        matched_word = clean_word if clean_word else raw_word
                        matched_tokens.add(idx)
                        found = True
                        break

            # 2. Match clean word or raw word directly to target
            if not found:
                for idx, (raw_word, clean_word, tok_lemma, spacy_lemma) in enumerate(token_info):
                    if idx not in matched_tokens:
                        if _norm(clean_word) == tgt_clean or _norm(raw_word) == tgt_clean:
                            matched_word = clean_word if clean_word else raw_word
                            matched_tokens.add(idx)
                            found = True
                            break

            # 3. Stem and derivation similarity matching
            if not found:
                for idx, (raw_word, clean_word, tok_lemma, spacy_lemma) in enumerate(token_info):
                    if idx not in matched_tokens:
                        if (
                            _stem_match(tok_lemma, tgt_clean) or
                            _stem_match(spacy_lemma, tgt_clean) or
                            _stem_match(clean_word, tgt_clean) or
                            _stem_match(raw_word, tgt_clean)
                        ):
                            matched_word = clean_word if clean_word else raw_word
                            matched_tokens.add(idx)
                            found = True
                            break

            pairs.append({'word': matched_word, 'lemma': target})
        return pairs


    def link_sentence_lexemas_to_database(self, data_manager, batch_size: int = 500, verbose: bool = True):
        """
        Links all sentences in the database to the pre-seeded lexemas table:
        - Uses pre-seeded lexemas from FLELex (does not insert any new lexemas)
        - Links sentences to matching lexemas in 'sentence_lexemas'
        """
        # 1. Fetch pre-seeded lexemas lookup map from DB
        with data_manager._get_connection() as conn:
            cursor = conn.execute("SELECT id, lemma, pos FROM lexemas;")
            rows = cursor.fetchall()
            lemma_pos_to_id = {}
            lemma_fallback_to_id = {}
            for r in rows:
                lem = r['lemma']
                pos = r['pos']
                lid = r['id']
                if pos:
                    lemma_pos_to_id[(lem, pos)] = lid
                if lem not in lemma_fallback_to_id:
                    lemma_fallback_to_id[lem] = lid

        # 2. Find sentences that need linking
        with data_manager._get_connection() as conn:
            cursor = conn.execute("SELECT DISTINCT sentence_id FROM sentence_lexemas;")
            indexed_sentence_ids = {r[0] for r in cursor.fetchall()}

        all_sentences = data_manager.get_all_sentences()
        unindexed = [s for s in all_sentences if s['id'] not in indexed_sentence_ids]

        if verbose:
            print(f"Total sentences: {len(all_sentences)}, already linked: {len(indexed_sentence_ids)}, to link: {len(unindexed)}")

        if not unindexed:
            return

        # 3. Process unindexed sentences in batches using nlp.pipe
        for i in range(0, len(unindexed), batch_size):
            chunk = unindexed[i:i + batch_size]
            texts = [s['fr'] for s in chunk]
            docs = list(self.nlp.pipe(texts, batch_size=batch_size))

            chunk_links = []

            for s_dict, doc in zip(chunk, docs):
                s_id = s_dict['id']
                seen_lids = set()
                for tok in doc:
                    if tok.is_punct or tok.is_space or tok.is_digit or tok.pos_ in ('PUNCT', 'SYM', 'SPACE', 'NUM', 'X'):
                        continue
                    _, _, lemma = self.get_word_level(tok)
                    if not lemma:
                        lemma = tok.lemma_.lower().strip()
                    lemma = lemma.lower().strip()
                    if lemma.endswith("'") or lemma.endswith("’"):
                        continue
                    pos = tok.pos_
                    lid = lemma_pos_to_id.get((lemma, pos)) or lemma_fallback_to_id.get(lemma)
                    if lid and lid not in seen_lids:
                        seen_lids.add(lid)
                        chunk_links.append((s_id, lid))

            if chunk_links:
                data_manager.bulk_link_sentence_lexemas(chunk_links)

    # Alias for backward compatibility
    index_database_lexemas = link_sentence_lexemas_to_database

_vocab_engine_instance = None
_vocab_engine_lock = threading.Lock()


def get_vocab_engine() -> VocabEngine:
    global _vocab_engine_instance
    if _vocab_engine_instance is None:
        with _vocab_engine_lock:
            if _vocab_engine_instance is None:
                _vocab_engine_instance = VocabEngine()
    return _vocab_engine_instance


if __name__ == "__main__":
    engine = get_vocab_engine()
    print("VocabEngine loaded successfully!")
    test_sentences = [
        "Bonjour, je m'appelle Marie et j'habite à Paris.",
        "Le chat noir dort sur la table de la cuisine.",
        "Quand j'étais enfant, nous partions souvent en vacances à la mer.",
        "Bien que cette décision paraisse opportune, elle suscite de vives controverses parmi les spécialistes.",
        "L'épistémologie cartésienne postule l'indubitabilité du cogito comme fondement ontologique."
    ]
    for s in test_sentences:
        lvl, score, breakdown = engine.score_sentence(s)
        print(f"\n[{lvl} - Score {score}] {s}")
        print("  Words:", " ".join(f"{b['word']}({b['level']})" for b in breakdown))
