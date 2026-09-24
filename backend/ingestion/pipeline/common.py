import os
import sys
import re
from typing import List, Dict, Any, Optional
import pymupdf
import spacy
from datasketch import MinHash, MinHashLSH
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

from backend.db import DataManager
from backend.nlp.vocab_engine import VocabEngine

_nlp = None


def get_spacy_nlp():
    """Thread-safe / cached loader for spaCy French pipeline."""
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
    """Cleans up internal spacing and formats punctuation spacing for French."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*([?!:;])', r' \1', text)
    return text.strip()


def is_valid_sentence(text: str, min_words: int = 3, max_words: int = 40) -> bool:
    """Heuristics to drop noisy, incomplete, or ill-formed sentences."""
    if not text:
        return False

    words = text.split()
    # 1. Check length
    if len(words) < min_words or len(words) > max_words:
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


def get_lemma_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Generates a MinHash signature based on the lemmas of a sentence."""
    nlp = get_spacy_nlp()
    doc = nlp(text)
    m = MinHash(num_perm=num_perm)
    for token in doc:
        if not token.is_punct and not token.is_space:
            m.update(token.lemma_.encode('utf8'))
    return m


def deduplicate_against_db(
    sentences: List[str],
    threshold: float = 0.85,
    num_perm: int = 128,
    data_manager: Optional[DataManager] = None
) -> List[str]:
    """
    Deduplicates incoming candidate sentences against the database using LSH MinHash.
    Also deduplicates within the candidate batch itself.
    """
    if not sentences:
        return []

    dm = data_manager or DataManager()
    db_sentences = [s['fr'] for s in dm.get_all_sentences()]

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, db_sent in enumerate(tqdm(db_sentences, desc="Indexing DB sentences")):
        m = get_lemma_minhash(db_sent, num_perm=num_perm)
        lsh.insert(f"db_{i}", m)

    unique_sentences = []
    for i, new_sent in enumerate(tqdm(sentences, desc="Deduplicating sentences")):
        m = get_lemma_minhash(new_sent, num_perm=num_perm)
        result = lsh.query(m)
        if not result:
            unique_sentences.append(new_sent)
            lsh.insert(f"batch_{i}", m)

    return unique_sentences


def extract_sentences_from_pdf(
    pdf_path_or_bytes: str | bytes,
    stop_phrase: Optional[str] = None
) -> List[str]:
    """
    Extracts, cleans, and segments sentences from a PDF document.
    Optionally stops extracting content once `stop_phrase` is encountered.
    """
    nlp = get_spacy_nlp()
    if isinstance(pdf_path_or_bytes, bytes):
        doc = pymupdf.open(stream=pdf_path_or_bytes, filetype="pdf")
    else:
        doc = pymupdf.open(pdf_path_or_bytes)

    full_text = ""
    for page in doc:
        page_text = page.get_text()
        if stop_phrase and stop_phrase in page_text:
            full_text += page_text.split(stop_phrase)[0]
            break
        full_text += page_text + "\n"

    # Fix PDF line breaks: rejoin hyphenated words at the end of a line
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


def translate_batch(sentences: List[str], batch_size: int = 16) -> List[str]:
    """Translates a list of French sentences to Spanish using MarianMT."""
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
            print(f"Warning: GPU compute capability {major}.{minor} is too old. Falling back to CPU.")
            device = "cpu"
    else:
        device = "cpu"

    model.to(device)
    print(f"Model loaded on {device}. Translating...")

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


def process_and_score_sentences(
    sentences: List[str],
    source_name: str,
    source_metadata: Optional[Dict[str, Any]] = None,
    engine: Optional[VocabEngine] = None,
    batch_size: int = 64
) -> List[Dict[str, Any]]:
    """
    Processes a list of unique French sentences:
    1. Scores them with the CEFR ML classifier.
    2. Translates them to Spanish.
    3. Scores them with the lexical engine for progressive vocabulary.
    4. Packages them into standard database ingestion records.
    """
    if not sentences:
        return []

    engine = engine or VocabEngine()
    metadata = source_metadata or {}

    print(f"\nCalculating CEFR Levels using ML Classifier for {len(sentences)} sentences...")
    ml_results = engine.score_sentences_ml_batch(sentences, batch_size=batch_size)

    print(f"\nTranslating {len(sentences)} sentences to Spanish...")
    translated_texts = translate_batch(sentences)

    records = []
    for fr, ml_res, es in zip(sentences, ml_results, translated_texts):
        lvl = ml_res["level"]
        score_ml = float(ml_res["score"])

        # Lexical score for progressive vocabulary
        _, lex_sc, _ = engine.score_sentence(fr)

        records.append({
            "fr": fr,
            "es": es,
            "level": lvl,
            "score": score_ml,
            "score_lexical": float(lex_sc),
            "score_ml": score_ml,
            "source": source_name,
            "metadata": metadata
        })

    return records
