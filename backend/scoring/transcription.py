import difflib
import re
from typing import Any, Dict

from backend.nlp.text_utils import levenshtein, remove_accents, normalize_french_typography

WORD_RE = re.compile(r"\b[^\W\d_]+(?:[-'][^\W\d_]+)*\b", flags=re.UNICODE)


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
    reference_text = normalize_french_typography(reference_text)
    student_text = normalize_french_typography(student_text)

    if is_skipped or not student_text.strip():
        ref_words = WORD_RE.findall(reference_text)
        diff = [{"word": w, "ref_word": w, "type": "missing"} for w in ref_words]
        return {
            "quality": 0.0,
            "composite_score": 0.0,
            "is_exact": False,
            "is_correct": False,
            "status_message": "Sentence skipped" if is_skipped else "No transcription submitted",
            "diff": diff,
            "notes": ["Sentence skipped" if is_skipped else "No transcription submitted"],
            "reference_text": reference_text,
            "student_text": student_text
        }

    ref_words = WORD_RE.findall(reference_text)
    stu_words = WORD_RE.findall(student_text)

    ref_norm = [remove_accents(w.lower()) for w in ref_words]
    stu_norm = [remove_accents(w.lower()) for w in stu_words]

    matcher = difflib.SequenceMatcher(None, ref_norm, stu_norm)
    diff_items = []
    notes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for r_idx, s_idx in zip(range(i1, i2), range(j1, j2)):
                r_orig = ref_words[r_idx]
                s_orig = stu_words[s_idx]
                if r_orig.lower() == s_orig.lower():
                    diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "exact"})
                else:
                    diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "accent"})
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
                    notes.append(f"Accent: '{s_orig}' → '{r_orig}'")
                elif dist <= 2:
                    diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "typo"})
                    notes.append(f"Typo: '{s_orig}' → '{r_orig}'")
                else:
                    diff_items.append({"word": s_orig, "ref_word": r_orig, "type": "mismatch"})

            for k in range(min_len, len(r_slice)):
                r_orig = ref_words[r_slice[k]]
                diff_items.append({"word": r_orig, "ref_word": r_orig, "type": "missing"})
                notes.append(f"Missing: '{r_orig}'")

            for k in range(min_len, len(s_slice)):
                s_orig = stu_words[s_slice[k]]
                diff_items.append({"word": s_orig, "ref_word": "", "type": "extra"})
        elif tag == 'delete':
            for r_idx in range(i1, i2):
                r_orig = ref_words[r_idx]
                diff_items.append({"word": r_orig, "ref_word": r_orig, "type": "missing"})
                notes.append(f"Missing: '{r_orig}'")
        elif tag == 'insert':
            for s_idx in range(j1, j2):
                s_orig = stu_words[s_idx]
                diff_items.append({"word": s_orig, "ref_word": "", "type": "extra"})

    # Determine binary correctness: all tokens must be exact matches
    has_errors = any(item.get("type") != "exact" for item in diff_items)
    is_correct = bool(len(diff_items) > 0 and not has_errors and len(diff_items) == len(ref_words))
    is_exact = is_correct

    quality = 1.0 if is_correct else 0.0
    composite_score = 100.0 if is_correct else 0.0
    status_msg = "Transcription correcte" if is_correct else "À corriger"

    return {
        "quality": quality,
        "composite_score": composite_score,
        "is_exact": is_exact,
        "is_correct": is_correct,
        "status_message": status_msg,
        "diff": diff_items,
        "notes": notes,
        "reference_text": reference_text,
        "student_text": student_text
    }
