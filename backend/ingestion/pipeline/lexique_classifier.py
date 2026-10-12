import math
import csv
from typing import List, Dict, Any, Tuple

# Core A1 foundational vocabulary (pronouns, auxiliaries, common prepositions, conjunctions, adverbs)
CORE_A1_LEMMAS = {
    'le', 'la', 'un', 'ce', 'cet', 'cette', 'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
    'me', 'te', 'se', 'moi', 'toi', 'lui', 'y', 'en', 'mon', 'ton', 'son', 'notre', 'votre', 'leur',
    'à', 'de', 'pour', 'dans', 'sur', 'avec', 'sans', 'sous', 'chez', 'vers', 'par', 'entre',
    'et', 'ou', 'mais', 'car', 'si', 'que', 'qui', 'quand', 'où', 'comme', 'donc',
    'être', 'avoir', 'aller', 'faire', 'dire', 'pouvoir', 'vouloir', 'savoir', 'voir', 'devoir', 'venir', 'falloir',
    'ne', 'pas', 'plus', 'très', 'bien', 'oui', 'non', 'merci', 'bonjour', 'bonsoir', 'salut',
    "aujourd'hui", 'hier', 'demain'
}


def compute_zipf(freq: float) -> float:
    """
    Computes standard Zipf frequency score: log10(frequency per million) + 3.0.
    Returns 0.0 for non-positive frequencies.
    """
    if freq > 0:
        return round(math.log10(freq) + 3.0, 2)
    return 0.0


def rank_to_cefr(rank: int) -> str:
    """
    Maps 1-based integer rank R to CEFR level:
    - R <= 800: 'A1'
    - R <= 1800: 'A2'
    - R <= 3500: 'B1'
    - R <= 6500: 'B2'
    - R <= 12000: 'C1'
    - R > 12000: 'C2'
    """
    if rank <= 800:
        return 'A1'
    elif rank <= 1800:
        return 'A2'
    elif rank <= 3500:
        return 'B1'
    elif rank <= 6500:
        return 'B2'
    elif rank <= 12000:
        return 'C1'
    else:
        return 'C2'


def parse_and_classify_lexique(tsv_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Reads Lexique 4 TSV dataset and extracts:
    1. Lemmas: Unique (lemma, pos) canonical entries ranked descending by FreqLemme.
       Includes all rows where 14_IsLem == '1' plus any inflected-only lemmas not marked with IsLem.
    2. Words: All surface word forms mapped to parent (lemma, pos) with max observed FreqMot.

    Returns:
        (lemmas, words): Tuple of lists of dict records ready for database insertion.
    """
    is_lem_lemmas: Dict[Tuple[str, str], float] = {}
    non_islem_lemmas: Dict[Tuple[str, str], float] = {}
    words_dict: Dict[Tuple[str, str, str], float] = {}

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            mot = (row.get('1_Mot') or '').strip().lower()
            lem = (row.get('4_Lemme') or '').strip().lower() or mot
            pos = (row.get('5_Cgram') or '').strip().upper()

            try:
                freq_lem = float(row.get('12_FreqLemme') or 0.0)
            except (ValueError, TypeError):
                freq_lem = 0.0

            try:
                freq_mot = float(row.get('10_FreqMot') or 0.0)
            except (ValueError, TypeError):
                freq_mot = 0.0

            is_lem = (row.get('14_IsLem') or '').strip() == '1'
            lemma_key = (lem, pos)

            if is_lem:
                # Canonical lemma entry
                is_lem_lemmas[lemma_key] = freq_lem
            else:
                # Track max frequency for non-IsLem lemma candidates
                if lemma_key not in non_islem_lemmas or freq_lem > non_islem_lemmas[lemma_key]:
                    non_islem_lemmas[lemma_key] = freq_lem

            # Surface word record: deduplicate by (word, lemma, pos) keeping max FreqMot
            word_key = (mot, lem, pos)
            if word_key not in words_dict or freq_mot > words_dict[word_key]:
                words_dict[word_key] = freq_mot

    # Combine canonical lemmas with non-IsLem lemmas that never appeared as IsLem == 1
    combined_lemmas = dict(is_lem_lemmas)
    for lemma_key, max_freq in non_islem_lemmas.items():
        if lemma_key not in combined_lemmas:
            combined_lemmas[lemma_key] = max_freq

    # Sort descending by FreqLemme, with stable alphabetical tie-break by lemma and pos
    sorted_lemmas = sorted(
        combined_lemmas.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1])
    )

    # Build lemmas list with 1-based ranks and CEFR classification
    lemmas_result: List[Dict[str, Any]] = []
    for rank, ((lemma, pos), freq) in enumerate(sorted_lemmas, 1):
        level = rank_to_cefr(rank)
        zipf = compute_zipf(freq)
        is_base = lemma in CORE_A1_LEMMAS
        lemmas_result.append({
            'lemma': lemma,
            'pos': pos,
            'level': level,
            'frequency': freq,
            'is_base': is_base,
            'metadata': {
                'rank': rank,
                'zipf': zipf,
            }
        })

    # Build surface words list
    words_result: List[Dict[str, Any]] = []
    for (word, lemma, pos), freq in words_dict.items():
        words_result.append({
            'word': word,
            'lemma': lemma,
            'pos': pos,
            'frequency': freq,
            'metadata': {},
        })

    return lemmas_result, words_result
