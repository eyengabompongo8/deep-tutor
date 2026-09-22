import re
import unicodedata
import numpy as np
from typing import List

def levenshtein(s1: str, s2: str) -> int:
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

def remove_accents(input_str: str) -> str:
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def normalize_french_typography(text: str) -> str:
    """
    Normalizes typography for French text symmetrically:
    - Normalizes Unicode single quotes/apostrophes (’ ‘ ` ´ ′) to standard ASCII '.
    - Normalizes Unicode dashes/hyphens (‑ – — −) to standard ASCII -.
    - Normalizes non-breaking and special spaces (\u00a0 \u202f \u2009) to standard spaces.
    - Ensures standard French spacing before high punctuation marks (?, !, :, ;).
    - Collapses redundant spaces and strips leading/trailing whitespace.
    """
    if not text:
        return ""

    # 1. Normalize Unicode apostrophes
    text = re.sub(r"[’‘`´′]", "'", text)

    # 2. Normalize Unicode dashes
    text = re.sub(r"[‑–—−]", "-", text)

    # 3. Normalize non-breaking and special spaces
    text = re.sub(r"[\u00a0\u202f\u2009]", " ", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)

    # 4. French high punctuation spacing (?, !, :, ;)
    for mark in [r"\?", r"!", r":", r";"]:
        pattern = r"(\S)" + mark
        def add_space(m):
            return f"{m.group(1)} {m.group(0)[-1]}"
        text = re.sub(pattern, add_space, text)

    # 5. Clean up redundant spaces around punctuation and words
    text = re.sub(r"\s+([?!:;])", r" \1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def autocorrect_text(student_text: str, reference_text: str, lang_tool=None):
    if not student_text:
        return student_text, []

    corrections_made = []
    # Normalize typography symmetrically
    text = normalize_french_typography(student_text)
    ref = normalize_french_typography(reference_text) if reference_text else ""

    # 1. Punctuation Reconciliation: terminal marks (. ? !) from reference
    if ref and ref[-1] in {".", "?", "!"}:
        target_mark = ref[-1]
        if not text.endswith((".", "?", "!")):
            text = f"{text}{target_mark}"
            corrections_made.append(f"added '{target_mark}'")
        elif text[-1] != target_mark and text[-1] in {".", "?", "!"}:
            old_mark = text[-1]
            text = f"{text[:-1]}{target_mark}"
            corrections_made.append(f"'{old_mark}' -> '{target_mark}'")

    # 2. Re-apply typography spacing after terminal punctuation reconciliation
    text = normalize_french_typography(text)

    # 3. LanguageTool Minor Typo Correction (Tier 1: strictly typos, edit distance <= 2)
    ignored_words = {'j', 'l', 'c', 'd', 'm', 't', 's', 'n', 'qu'}
    if lang_tool is not None:
        try:
            matches = lang_tool.check(text)
            # Sort matches in reverse order so character offsets remain valid during replacements
            matches = sorted(matches, key=lambda m: m.offset, reverse=True)
            for m in matches:
                # Strictly allow only spelling typos; never solve grammar, confused words, or syntax
                is_spelling_issue = (
                    getattr(m, "category", "") == "TYPOS"
                    or getattr(m, "rule_issue_type", "") == "misspelling"
                    or "SPELL" in getattr(m, "rule_id", "")
                    or "MORFOLOGIK" in getattr(m, "rule_id", "")
                )
                is_grammar_or_confusion = (
                    getattr(m, "category", "") in {"GRAMMAR", "CONFUSED_WORDS"}
                    or getattr(m, "rule_issue_type", "") in {"grammar", "style", "register"}
                )
                if not is_spelling_issue or is_grammar_or_confusion:
                    continue

                if not m.replacements:
                    continue

                original_word = text[m.offset:m.offset + m.error_length]
                if original_word.lower() in ignored_words:
                    continue

                candidate = m.replacements[0]
                if candidate and candidate.lower() != original_word.lower():
                    if levenshtein(original_word.lower(), candidate.lower()) <= 2:
                        if original_word.istitle():
                            candidate = candidate.capitalize()
                        elif original_word.isupper():
                            candidate = candidate.upper()
                        text = text[:m.offset] + candidate + text[m.offset + m.error_length:]
                        corrections_made.append(f"{original_word} -> {candidate}")
        except Exception as e:
            print(f"LanguageTool tier 1 warning: {e}")

    # Deduplicate corrections while preserving order
    seen = set()
    unique_corrections = []
    for c in corrections_made:
        if c not in seen:
            seen.add(c)
            unique_corrections.append(c)

    return text, unique_corrections

def get_aligned_targets(selected_source_indices: List[int], matrix: np.ndarray, num_tgt_words: int) -> List[int]:
    if not selected_source_indices or matrix is None or matrix.size == 0:
        return []

    num_tgt, num_src = matrix.shape
    aligned = set()

    for s_idx in selected_source_indices:
        if 0 <= s_idx < num_src:
            col = matrix[:, s_idx]
            best_t = int(np.argmax(col))
            aligned.add(best_t)

            max_col_val = col[best_t]
            for t_idx in range(num_tgt):
                row_best_s = int(np.argmax(matrix[t_idx, :]))
                if row_best_s == s_idx and col[t_idx] >= 0.65 * max_col_val:
                    aligned.add(t_idx)

    return sorted(list(aligned))
