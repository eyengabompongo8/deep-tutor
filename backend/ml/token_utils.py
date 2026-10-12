import re
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from backend.ml.registry import NMTRawOutput, FluencyRawOutput
from backend.config import NMT_DELTA_L_THRESHOLD

# SentencePiece word-boundary markers used by M2M100 and related models
_SP_BOUNDARIES = (
    "\u2581",   # ▁  LOWER ONE EIGHTH BLOCK (primary marker in M2M100)
    " ",         # regular space
    "\u00a0",    # narrow no-break space
)


def _is_special_token(tok: str) -> bool:
    """Return True for any SentencePiece/HuggingFace special token."""
    return tok.startswith("<") or tok.startswith("__") or tok in {"</s>", "<pad>", "<s>", "<unk>"}


def _strip_boundary(tok: str) -> str:
    """Strip leading boundary markers and surrounding whitespace from a token."""
    for marker in _SP_BOUNDARIES:
        tok = tok.lstrip(marker)
    return tok.strip()


def _starts_new_word(tok: str) -> bool:
    """Return True if the token begins a new word (starts with a boundary marker)."""
    return any(tok.startswith(m) for m in _SP_BOUNDARIES)


def aggregate_sentencepiece_tokens_to_words(tokens: List[str]) -> Tuple[List[str], List[List[int]]]:
    """
    Groups SentencePiece sub-tokens into full, clean words.
    Handles the ▁ (U+2581) boundary marker used by M2M100 and mBART tokenizers.
    Strips ALL special tokens (<s>, </s>, <pad>, <unk>, __lang__, etc.).
    Dynamically splits French/apostrophe contractions (e.g. "J'aime" -> ["J'", "aime"],
    "c'est" -> ["c'", "est"], "d'un" -> ["d'", "un"], "qu'il" -> ["qu'", "il"])
    so that contracted words are separated into distinct words for fine-grained mapping.

    Returns:
        words: list[str] — clean word strings visible to the user.
        word_subtoken_indices: list[list[int]] — original token indices per word.
    """
    word_subtoken_indices = []
    current_subtokens = []

    for idx, token in enumerate(tokens):
        if _is_special_token(token):
            continue

        if _starts_new_word(token) or not current_subtokens:
            if current_subtokens:
                word_subtoken_indices.append(current_subtokens)
                current_subtokens = []
            current_subtokens.append(idx)
        else:
            current_subtokens.append(idx)

    if current_subtokens:
        word_subtoken_indices.append(current_subtokens)

    words = []
    final_subtoken_indices = []

    for sub_indices in word_subtoken_indices:
        parts = [_strip_boundary(tokens[i]) for i in sub_indices]
        word_text = "".join(parts).strip()
        if not word_text:
            continue
        words.append(word_text)
        final_subtoken_indices.append(sub_indices)

    return words, final_subtoken_indices


def _build_nmt_breakdowns(raw: NMTRawOutput) -> Tuple[List[str], List[str], List[List[int]], List[List[int]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Computes token-level and word-level metric breakdowns from raw NMT inference outputs.
    Calculates surprise Delta_L = max(0, loss - top_1_loss) and flags errors when Delta_L > 8.0.
    """
    src_words, src_sub_groups = aggregate_sentencepiece_tokens_to_words(raw.src_raw_tokens)
    tgt_words, tgt_sub_groups = aggregate_sentencepiece_tokens_to_words(raw.tgt_raw_tokens)

    nmt_token_breakdown = []
    token_loss_map = {}
    top_1_loss_map = {}

    for idx, t_id in enumerate(raw.target_ids):
        tok_str = raw.tgt_raw_tokens[idx] if idx < len(raw.tgt_raw_tokens) else str(t_id)
        if _is_special_token(tok_str) or t_id == -100:
            continue

        loss_i = raw.token_losses[idx] if idx < len(raw.token_losses) else 0.0
        top_1_tok_str = raw.top_1_tokens[idx] if idx < len(raw.top_1_tokens) else ""
        top_1_loss_i = raw.top_1_losses[idx] if idx < len(raw.top_1_losses) else 0.0

        tok_clean = _strip_boundary(tok_str)
        top_1_clean = _strip_boundary(top_1_tok_str)

        ppl_i = float(np.exp(loss_i))
        prob_i = float(np.exp(-loss_i)) * 100.0
        delta_l_i = max(0.0, loss_i - top_1_loss_i)

        token_loss_map[idx] = loss_i
        top_1_loss_map[idx] = top_1_loss_i

        nmt_token_breakdown.append({
            "index": idx,
            "token": tok_clean if tok_clean else tok_str,
            "loss": loss_i,
            "ppl": ppl_i,
            "prob": prob_i,
            "top_prediction": top_1_clean if top_1_clean else top_1_tok_str,
            "delta_l": delta_l_i,
            "is_error": delta_l_i > NMT_DELTA_L_THRESHOLD
        })

    nmt_word_breakdown = []
    for w_idx, (word_text, sub_indices) in enumerate(zip(tgt_words, tgt_sub_groups)):
        word_losses = [token_loss_map[idx] for idx in sub_indices if idx in token_loss_map]
        word_top_1_losses = [top_1_loss_map[idx] for idx in sub_indices if idx in top_1_loss_map]

        if word_losses:
            w_loss = float(np.mean(word_losses))
            w_top_1_loss = float(np.mean(word_top_1_losses))
        else:
            w_loss = 0.0
            w_top_1_loss = 0.0

        w_ppl = float(np.exp(w_loss))
        w_prob = float(np.exp(-w_loss)) * 100.0
        w_delta_l = max(0.0, w_loss - w_top_1_loss)

        top_pred_parts = []
        for s_idx in sub_indices:
            if s_idx < len(raw.top_1_tokens):
                tok_str = raw.top_1_tokens[s_idx]
                if not _is_special_token(tok_str):
                    top_pred_parts.append(_strip_boundary(tok_str))
        top_pred_word = "".join(top_pred_parts).strip() if top_pred_parts else "—"

        nmt_word_breakdown.append({
            "index": w_idx,
            "word": word_text,
            "loss": w_loss,
            "ppl": w_ppl,
            "prob": w_prob,
            "top_prediction": top_pred_word,
            "delta_l": w_delta_l,
            "is_error": w_delta_l > NMT_DELTA_L_THRESHOLD
        })

    return src_words, tgt_words, src_sub_groups, tgt_sub_groups, nmt_token_breakdown, nmt_word_breakdown


def _build_fluency_breakdowns(raw: FluencyRawOutput, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Computes token-level and word-level metric breakdowns from raw Fluency inference outputs.
    """
    fluency_token_breakdown = []
    for idx in range(len(raw.tokens)):
        tok_str = raw.tokens[idx]
        loss_i = raw.token_losses[idx]
        top_1_str = raw.top_1_tokens[idx]
        top_1_loss_i = raw.top_1_losses[idx]

        ppl_i = float(np.exp(loss_i))
        prob_i = float(np.exp(-loss_i)) * 100.0
        delta_l_i = max(0.0, loss_i - top_1_loss_i)

        fluency_token_breakdown.append({
            "index": idx + 1,
            "token": tok_str if tok_str else f"token_{idx}",
            "loss": loss_i,
            "ppl": ppl_i,
            "prob": prob_i,
            "top_prediction": top_1_str,
            "delta_l": delta_l_i,
            "is_error": delta_l_i > NMT_DELTA_L_THRESHOLD
        })

    words = text.strip().split()
    fluency_word_breakdown = []
    if words and fluency_token_breakdown:
        num_words = len(words)
        tokens_per_word = max(1, len(fluency_token_breakdown) // num_words)

        for w_idx, w_text in enumerate(words):
            start_t = w_idx * tokens_per_word
            end_t = (w_idx + 1) * tokens_per_word if w_idx < num_words - 1 else len(fluency_token_breakdown)
            word_toks = fluency_token_breakdown[start_t:end_t]

            if word_toks:
                w_loss = float(np.mean([t["loss"] for t in word_toks]))
                w_top_1 = word_toks[0]["top_prediction"]
            else:
                w_loss = 0.0
                w_top_1 = "—"

            w_ppl = float(np.exp(w_loss))
            w_prob = float(np.exp(-w_loss)) * 100.0
            w_top_1_loss = float(np.mean([t["loss"] - t["delta_l"] for t in word_toks])) if word_toks else 0.0
            w_delta_l = max(0.0, w_loss - w_top_1_loss)

            fluency_word_breakdown.append({
                "index": w_idx,
                "word": w_text,
                "loss": w_loss,
                "ppl": w_ppl,
                "prob": w_prob,
                "top_prediction": w_top_1,
                "delta_l": w_delta_l,
                "is_error": w_delta_l > NMT_DELTA_L_THRESHOLD
            })

    return fluency_token_breakdown, fluency_word_breakdown


def _compute_attention_alignment_matrix(
    subtoken_attn: np.ndarray,
    src_words: List[str],
    tgt_words: List[str],
    src_sub_groups: List[List[int]],
    tgt_sub_groups: List[List[int]],
) -> np.ndarray:
    """
    Constructs a word alignment matrix purely from NMT cross-attention weights.
    Aggregates row-softmax normalized subtoken attention over whole-word boundaries:
      M(t, s) = sum_{i in sub(t)} sum_{j in sub(s)} A_{i, j}
    Normalized such that max(M) == 1.0 (or 0 if all zeros) for downstream thresholding.
    """
    row_sums = subtoken_attn.sum(axis=1, keepdims=True) + 1e-8
    norm_subtoken_attn = subtoken_attn / row_sums

    num_tgt_words = max(1, len(tgt_words))
    num_src_words = max(1, len(src_words))
    word_attn_matrix = np.zeros((num_tgt_words, num_src_words), dtype=np.float32)

    for tgt_w_idx, tgt_subs in enumerate(tgt_sub_groups):
        for src_w_idx, src_subs in enumerate(src_sub_groups):
            attn_sum = 0.0
            for t_sub in tgt_subs:
                for s_sub in src_subs:
                    if t_sub < norm_subtoken_attn.shape[0] and s_sub < norm_subtoken_attn.shape[1]:
                        attn_sum += float(norm_subtoken_attn[t_sub, s_sub])
            word_attn_matrix[tgt_w_idx, src_w_idx] = attn_sum

    max_attn = float(np.max(word_attn_matrix))
    if max_attn > 1e-8:
        word_attn_matrix = word_attn_matrix / max_attn

    return word_attn_matrix
