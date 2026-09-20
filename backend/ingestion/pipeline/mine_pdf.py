import os
import sys
import pymupdf as fitz
import re
import spacy
from datasketch import MinHash, MinHashLSH
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

from backend.data_manager import DataManager
from backend.vocab_engine import VocabEngine

_nlp = None


def get_spacy_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("fr_core_news_sm")
            _nlp.max_length = 5000000
        except OSError:
            print("Downloading spaCy model 'fr_core_news_sm'...")
            os.system(f'{sys.executable} -m spacy download fr_core_news_sm')
            _nlp = spacy.load("fr_core_news_sm")
            _nlp.max_length = 5000000
    return _nlp


def format_sentence(text: str) -> str:
    """Cleans up internal spacing and formatting."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*([?!:;])', r' \1', text)
    return text.strip()


def is_valid_sentence(text: str) -> bool:
    """Heuristics to drop unwanted sentences."""
    words = text.split()
    
    # 1. Check length
    if len(words) < 3 or len(words) > 40:
        return False
        
    # 2. Check for numbers
    if re.search(r'\d', text):
        return False
        
    # 3. Drop sentences that have a spaced hyphen or em-dash
    if re.search(r'\s[-—–]\s', text):
        return False
        
    # 4. Repetitive punctuation
    if re.search(r'\.{4,}|_{4,}|-{4,}', text):
        return False
        
    # 5. Check if first alphabetical character is uppercase
    first_alpha = next((c for c in text if c.isalpha()), None)
    if not first_alpha or not first_alpha.isupper():
        return False
        
    # 6. Uppercase ratio (reject if > 50% of letters are uppercase)
    letters = [c for c in text if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.5:
            return False
            
    # 7. Terminal Punctuation Check
    if not re.search(r'[.!?…](?:["»”’‘\s]*)$', text):
        return False
        
    # 8. Check for rare symbols / math
    if re.search(r'[^\w\s.,!?\'"():;áàâäéèêëíìîïóòôöúùûüçœæ\-’‘“”\xad]', text, re.IGNORECASE):
        return False
        
    return True


def extract_sentences_from_pdf(pdf_path: str) -> list[str]:
    nlp = get_spacy_nlp()
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    # FIX PDF LINE BREAKS: rejoin hyphenated words at the end of a line
    full_text = re.sub(r'-\n\s*', '', full_text)
    # Replace newlines with spaces before passing to spaCy to avoid bad sentence breaks
    full_text = full_text.replace('\n', ' ')
    
    doc_spacy = nlp(full_text)
    
    valid_sentences = []
    for sent in doc_spacy.sents:
        clean_text = sent.text.strip().replace('\n', ' ')
        clean_text = re.sub(r'^[\s\-—–«"<>]+', '', clean_text)
        clean_text = re.sub(r'[\s»"<>]+$', '', clean_text)
        clean_text = clean_text.replace('«', '').replace('»', '').replace('<', '').replace('>', '')
        clean_text = clean_text.strip()
        
        if not clean_text:
            continue
            
        clean_text = format_sentence(clean_text)
            
        if is_valid_sentence(clean_text):
            valid_sentences.append(clean_text)
            
    return valid_sentences


def get_lemma_minhash(text: str, num_perm=128):
    """Generates a MinHash signature based on the lemmas of a sentence."""
    nlp = get_spacy_nlp()
    doc = nlp(text)
    m = MinHash(num_perm=num_perm)
    for token in doc:
        if not token.is_punct and not token.is_space:
            m.update(token.lemma_.encode('utf8'))
    return m


def translate_batch(sentences: list[str]) -> list[str]:
    """Translates a list of French sentences to Spanish."""
    if not sentences:
        return []
        
    model_name = 'Helsinki-NLP/opus-mt-fr-es'
    print(f"\nLoading translation model {model_name}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 7:
            device = "cuda"
        else:
            print(f"Warning: GPU compute capability {major}.{minor} is too old for this PyTorch version. Falling back to CPU.")
            device = "cpu"
    else:
        device = "cpu"
    model.to(device)
    print(f"Model loaded on {device}. Translating...")
    
    batch_size = 16
    results = []
    
    for i in tqdm(range(0, len(sentences), batch_size), desc="Translating"):
        batch = sentences[i:i+batch_size]
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            translated = model.generate(**inputs)
            
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        results.extend(translated_texts)
        
    return results


def run_mine_pdf(pdf_path: str, interactive: bool = True) -> list[dict]:
    """Mines and scores sentences from a PDF file, returning standard sentence records."""
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return []
        
    print(f"Extracting sentences from {pdf_path}...")
    new_sentences = extract_sentences_from_pdf(pdf_path)
    
    print("Loading database sentences for deduplication...")
    dm = DataManager()
    db_sentences = [s['fr'] for s in dm.get_all_sentences()]
    
    print("Building LSH Index...")
    lsh = MinHashLSH(threshold=0.85, num_perm=128)
    
    for i, db_sent in enumerate(tqdm(db_sentences, desc="Indexing DB sentences")):
        m = get_lemma_minhash(db_sent)
        lsh.insert(f"db_{i}", m)
        
    unique_new_sentences = []
    for i, new_sent in enumerate(tqdm(new_sentences, desc="Deduplicating new sentences")):
        m = get_lemma_minhash(new_sent)
        result = lsh.query(m)
        
        if not result:
            unique_new_sentences.append(new_sent)
            lsh.insert(f"new_{i}", m)
            
    print(f"\nExtracted {len(new_sentences)} sentences from PDF.")
    print(f"After deduplication, {len(unique_new_sentences)} unique sentences remain.")
    if not unique_new_sentences:
        print("No new unique sentences found.")
        return []
        
    print("\nCalculating CEFR Levels using ML Classifier...")
    engine = VocabEngine()
    level_groups = {'A1': [], 'A2': [], 'B1': [], 'B2': [], 'C1': [], 'C2': []}

    ml_results = engine.score_sentences_ml_batch(unique_new_sentences, batch_size=64)
    for sent, ml_res in zip(unique_new_sentences, ml_results):
        lvl = ml_res["level"]
        score_ml = ml_res["score"]
        if lvl in level_groups:
            level_groups[lvl].append((sent, score_ml))
            
    print("\nCEFR Level Distribution of Extracted Sentences:")
    for lvl, sents in level_groups.items():
        print(f"  {lvl}: {len(sents)}")
        
    selected_sentences = []
    for lvl, sents in level_groups.items():
        if not sents:
            continue
        if interactive:
            while True:
                ans = input(f"How many sentences of level {lvl} to mine? (number or 'all', max {len(sents)}): ").strip().lower()
                if ans == 'all':
                    selected_sentences.extend(sents)
                    break
                elif ans.isdigit():
                    count = min(int(ans), len(sents))
                    selected_sentences.extend(sents[:count])
                    break
                else:
                    print("Invalid input. Please enter a number or 'all'.")
        else:
            selected_sentences.extend(sents)
            
    if not selected_sentences:
        print("No sentences selected.")
        return []
        
    print(f"\nTranslating {len(selected_sentences)} selected sentences...")
    texts_to_translate = [s[0] for s in selected_sentences]
    translated_texts = translate_batch(texts_to_translate)
    
    filename = os.path.basename(pdf_path)
    records = []
    for (fr, score_ml), es in zip(selected_sentences, translated_texts):
        lvl, lex_sc, _ = engine.score_sentence(fr)
        records.append({
            "fr": fr,
            "es": es,
            "level": lvl,
            "score": float(score_ml),
            "score_lexical": float(lex_sc),
            "score_ml": float(score_ml),
            "source": "pdf",
            "metadata": {"source_file": filename}
        })
        
    print(f"Successfully mined {len(records)} sentences from PDF.")
    return records
