import re
import unicodedata
from spellchecker import SpellChecker

spell = SpellChecker(language='fr')

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def autocorrect_text(student_text, reference_text):
    if not student_text:
        return student_text, []
        
    ref_words_unaccented = {remove_accents(w.lower()): w for w in re.findall(r"\b[^\W\d_]+\b", reference_text, re.UNICODE)}
    corrections_made = []
    
    def replace_word(match):
        word = match.group(0)
        word_lower = word.lower()
        word_unaccented = remove_accents(word_lower)
        
        # 1. Check against reference text for missing accents on valid words
        if word_unaccented in ref_words_unaccented and ref_words_unaccented[word_unaccented].lower() != word_lower:
            correct_word = ref_words_unaccented[word_unaccented]
            if word.istitle():
                correct_word = correct_word.capitalize()
            elif word.isupper():
                correct_word = correct_word.upper()
            corrections_made.append(f"{word} -> {correct_word}")
            return correct_word
            
        # 2. Check pyspellchecker for unknown words
        if not spell.known([word_lower]):
            candidate = spell.correction(word_lower)
            if candidate and candidate != word_lower:
                if levenshtein(word_lower, candidate) <= 2:
                    if word.istitle():
                        candidate = candidate.capitalize()
                    elif word.isupper():
                        candidate = candidate.upper()
                    corrections_made.append(f"{word} -> {candidate}")
                    return candidate
                    
        return word

    corrected_text = re.sub(r"\b[^\W\d_]+\b", replace_word, student_text, flags=re.UNICODE)
    
    # Deduplicate corrections
    seen = set()
    unique_corrections = []
    for c in corrections_made:
        if c not in seen:
            seen.add(c)
            unique_corrections.append(c)
            
    return corrected_text, unique_corrections

if __name__ == "__main__":
    t, c = autocorrect_text("J'aime l'eleve ou la garcon.", "J'aime l'élève où le garçon.")
    print("Corrected:", t)
    print("Corrections:", c)

    t2, c2 = autocorrect_text("Le garcon ou la fille", "Le garçon ou la fille")
    print("Corrected 2:", t2)
    print("Corrections 2:", c2)
