import difflib
import re
from typing import Any, Dict

from backend.nlp.text_utils import levenshtein, remove_accents


def evaluate_transcription(
    student_text: str,
    reference_text: str,
    is_skipped: bool = False
) -> Dict[str, Any]:
    """
    Evaluates student French transcription against reference text.
    Computes token-level and character-level alignment, identifying exact matches,
    accent errors, typos, missing words, and extra words.
    """
    if is_skipped or not student_text.strip():
        ref_words = re.findall(r"\b[^\W\d_]+(?:['’][^\W\d_]+)?\b", reference_text, flags=re.UNICODE)
        diff = [{"word": w, "ref_word": w, "type": "missing"} for w in ref_words]
        return {
            "quality": 0.0,
            "composite_score": 0.0,
            "is_exact": False,
            "status_message": "Sentence skipped" if is_skipped else "No transcription submitted",
            "diff": diff,
            "notes": ["Sentence skipped" if is_skipped else "No transcription submitted"],
            "reference_text": reference_text,
            "student_text": student_text
        }

    ref_words = re.findall(r"\b[^\W\d_]+(?:['’][^\W\d_]+)?\b", reference_text, flags=re.UNICODE)
    stu_words = re.findall(r"\b[^\W\d_]+(?:['’][^\W\d_]+)?\b", student_text, flags=re.UNICODE)

    ref_norm = [remove_accents(w.lower()) for w in ref_words]
    stu_norm = [remove_accents(w.lower()) for w in stu_words]

    matcher = difflib.SequenceMatcher(None, ref_norm, stu_norm)
    diff_items = []
    token_scores = []
    notes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for r_idx, s_idx in zip(range(i1, i2), range(j1, j2)):
                r_orig = ref_words[r_idx]
                s_orig = stu_words[s_idx]
                if r_orig.lower() == s_orig.lower():
                    diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "exact"})
                    token_scores.append(1.0)
                else:
                    diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "accent"})
                    token_scores.append(0.85)
                    notes.append(f"Accent: '{s_orig}' → '{r_orig}'")
        elif tag == 'replace':
            r_slice = list(range(i1, i2))
            s_slice = list(range(j1, j2))
            min_len = min(len(r_slice), len(s_slice))
            for k in range(min_len):
                r_orig = ref_words[r_slice[k]]
                s_orig = stu_words[s_slice[k]]
                dist = levenshtein(s_orig.lower(), r_orig.lower())
                if remove_accents(s_orig.lower()) == remove_accents(r_orig.lower()):
                    diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "accent"})
                    token_scores.append(0.85)
                    notes.append(f"Accent: '{s_orig}' → '{r_orig}'")
                elif dist <= 2:
                    diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "typo"})
                    token_scores.append(0.50)
                    notes.append(f"Typo: '{s_orig}' → '{r_orig}'")
                else:
                    diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "mismatch"})
                    token_scores.append(0.0)

            for k in range(min_len, len(r_slice)):
                r_orig = ref_words[r_slice[k]]
                diff_items.append({"word": r_orig, "ref_word": r_orig, "type": "missing"})
                token_scores.append(0.0)
                notes.append(f"Missing: '{r_orig}'")

            for k in range(min_len, len(s_slice)):
                s_orig = stu_words[s_slice[k]]
                diff_items.append({"word": s_orig, "ref_word": "", "type": "extra"})
                token_scores.append(0.0)
        elif tag == 'delete':
            for r_idx in range(i1, i2):
                r_orig = ref_words[r_idx]
                diff_items.append({"word": r_orig, "ref_word": r_orig, "type": "missing"})
                token_scores.append(0.0)
                notes.append(f"Missing: '{r_orig}'")
        elif tag == 'insert':
            for s_idx in range(j1, j2):
                s_orig = stu_words[s_idx]
                diff_items.append({"word": s_orig, "ref_word": "", "type": "extra"})
                token_scores.append(0.0)

    # Character similarity fallback
    clean_ref = re.sub(r'[^\w\s]', '', reference_text.lower()).strip()
    clean_stu = re.sub(r'[^\w\s]', '', student_text.lower()).strip()
    max_c = max(1, max(len(clean_ref), len(clean_stu)))
    c_dist = levenshtein(clean_ref, clean_stu)
    char_sim = max(0.0, 1.0 - (c_dist / max_c))

    token_quality = sum(token_scores) / max(1, len(ref_words))
    raw_quality = 0.70 * token_quality + 0.30 * char_sim
    quality = round(max(0.0, min(1.0, raw_quality)), 4)
    composite_score = round(quality * 100.0, 1)

    is_exact = quality >= 0.98
    status_msg = "Exact match!" if is_exact else ("Minor corrections" if quality >= 0.70 else "Transcription reviewed")

    return {
        "quality": quality,
        "composite_score": composite_score,
        "is_exact": is_exact,
        "status_message": status_msg,
        "diff": diff_items,
        "notes": notes,
        "reference_text": reference_text,
        "student_text": student_text
    }
