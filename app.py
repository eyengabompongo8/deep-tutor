import gradio as gr
import pandas as pd
from data_manager import DataManager
import numpy as np
import re
import unicodedata
from spellchecker import SpellChecker

# Initialize spellchecker once
spell = SpellChecker(language='fr')

print("Initializing Data Manager...")
data_manager = DataManager()

print("Initializing ML Engine...")
from models import MLEngine
engine = MLEngine()


def get_sentences_choices():
    return [f"[{s['id']}] {s['es']}" for s in data_manager.get_all_sentences()]


def get_sentences_ids():
    return [str(s['id']) for s in data_manager.get_all_sentences()]


def get_dataframe_data():
    return [[s['id'], s['es'], s['fr']] for s in data_manager.get_all_sentences()]


def on_sentence_select(choice):
    if not choice:
        return gr.update(value=""), gr.update(value=""), gr.update(value="")
    try:
        s_id = int(str(choice).split(']')[0].replace('[', ''))
        target_s = next((s for s in data_manager.get_all_sentences() if s['id'] == s_id), None)
        if target_s:
            return gr.update(value=target_s['es']), gr.update(value=target_s['fr']), gr.update(value="")
    except Exception as e:
        print(f"Error in on_sentence_select: {e}")
    return gr.update(value=""), gr.update(value=""), gr.update(value="")


def get_aligned_targets(selected_source_indices, matrix, num_tgt_words):
    """
    High-confidence bi-directional alignment algorithm:
    For selected source words, target word t is aligned if:
      1. t is the column maximum target for source s (col max), OR
      2. s is the row maximum source for target t (row max) AND matrix[t, s] >= 0.65 * max_col_val.
    100% dynamic, generic, and un-hardcoded.
    """
    if not selected_source_indices or matrix is None:
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


def build_word_mapping_table(src_words, tgt_words, attn_matrix):
    """
    Builds a human-readable word alignment table from the word-level attention matrix.
    Maps each Spanish source word to its dynamic French target word(s), supporting
    1-to-1, 1-to-many, many-to-1, and many-to-many mappings.
    """
    if attn_matrix is None or len(src_words) == 0 or len(tgt_words) == 0:
        return []

    num_src = len(src_words)
    num_tgt = len(tgt_words)

    rows = []
    for j, src_word in enumerate(src_words):
        aligned_t_indices = get_aligned_targets([j], attn_matrix, num_tgt)
        mapped_tgt_words = [tgt_words[i] for i in aligned_t_indices if i < num_tgt]
        tgt_str = " / ".join(mapped_tgt_words) if mapped_tgt_words else "—"
        rows.append([src_word, tgt_str])

    return rows


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
    
    # Common French contractions to ignore
    ignored_words = {'j', 'l', 'c', 'd', 'm', 't', 's', 'n', 'qu'}
    
    def replace_word(match):
        word = match.group(0)
        word_lower = word.lower()
        word_unaccented = remove_accents(word_lower)
        
        # Skip autocorrecting common contractions
        if word_lower in ignored_words:
            return word
            
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


def render_breakdown_badges(breakdown_list, is_word_view=True):
    """
    Renders color-coded HTML badges for per-token or per-word breakdown items.
    High relative surprise items (Delta L > 2.0) are rendered in a distinct red warning badge.
    """
    if not breakdown_list:
        return "*No breakdown data available. Submit a translation above.*"

    html = "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; margin-bottom: 15px;'>"
    for item in breakdown_list:
        text_str = item.get("word") if is_word_view else item.get("token")
        ppl = item.get("ppl", 0.0)
        prob = item.get("prob", 0.0)
        top_pred = item.get("top_prediction", "—")
        delta_l = item.get("delta_l", 0.0)
        is_err = item.get("is_error", False)

        bg_color = "rgba(239, 68, 68, 0.15)" if is_err else "rgba(255, 255, 255, 0.05)"
        border_color = "#ef4444" if is_err else "#374151"
        text_color = "#f87171" if is_err else "#f3f4f6"
        badge_title = "⚠️ High Relative Surprise (Error)" if is_err else "Normal Token/Word"

        html += f"""
        <div style='background: {bg_color}; border: 1.5px solid {border_color}; border-radius: 8px; padding: 10px 14px; min-width: 140px;' title='{badge_title}'>
            <div style='font-size: 16px; font-weight: bold; color: {text_color}; margin-bottom: 4px;'>
                {text_str} {'⚠️' if is_err else ''}
            </div>
            <div style='font-size: 12px; color: #9ca3af;'>
                PPL: <span style='font-weight: 600; color: #e5e7eb;'>{ppl:.2f}</span> | Prob: <span style='font-weight: 600; color: #e5e7eb;'>{prob:.1f}%</span>
            </div>
            <div style='font-size: 11px; color: #9ca3af; margin-top: 2px;'>
                Top Expected: <span style='font-weight: 600; color: #60a5fa;'>{top_pred}</span> (&Delta;L: {delta_l:.2f})
            </div>
        </div>
        """
    html += "</div>"
    return html


def build_breakdown_dataframe(breakdown_list, is_word_view=True):
    """
    Builds a clean table for Dataframe display showing per-token / per-word breakdown.
    """
    if not breakdown_list:
        return []

    rows = []
    for item in breakdown_list:
        unit_text = item.get("word") if is_word_view else item.get("token")
        ppl = f"{item.get('ppl', 0.0):.2f}"
        prob = f"{item.get('prob', 0.0):.2f}%"
        loss = f"{item.get('loss', 0.0):.3f}"
        top_pred = item.get("top_prediction", "—")
        delta_l = f"{item.get('delta_l', 0.0):.3f}"
        status = "⚠️ Error (High Surprise)" if item.get("is_error", False) else "OK"
        rows.append([unit_text, ppl, prob, loss, top_pred, delta_l, status])

    return rows


def evaluate_translation(source_text, reference_text, student_text):
    if not source_text or not str(source_text).strip():
        return "Please select a Spanish sentence from the dropdown.", "", "", "", "", None, [], [], [], [], []

    if not student_text or not str(student_text).strip():
        return "Please type your French translation.", "", "", "", "", None, [], [], [], [], []

    if not reference_text or not str(reference_text).strip():
        reference_text = student_text

    student_text, autocorrect_msgs = autocorrect_text(student_text, reference_text)

    result = engine.compute_final_score(source_text, student_text, reference_text)

    status_msg = "✅ Evaluation Complete"
    if autocorrect_msgs:
        status_msg += "\n\n⚠️ **Autocorrected Words:**\n- " + "\n- ".join(autocorrect_msgs)
    if not result.get("is_french", True):
        status_msg = f"⚠️ Language Gatekeeper Triggered: Input classified as non-French or confidence below 75% threshold (Prob: {result['lid_prob']:.2f})."

    mapping_data = result["mapping_data"]

    sem_text = f"{result['s_sem']:.1f} / 100"
    nmt_text = f"{result['s_nmt']:.1f} / 100 (PPL: {result['nmt_ppl']:.2f})"
    fluency_text = f"{result['s_fluency']:.1f} / 100 (PPL: {result['fluency_ppl']:.2f})"

    # Build word mapping table
    word_table = []
    if mapping_data is not None:
        src_words, tgt_words, attn_matrix = mapping_data
        word_table = build_word_mapping_table(src_words, tgt_words, attn_matrix)

    syntactical_errors = result.get("syntactical_errors", [])
    if syntactical_errors:
        syntax_msg = "### ❌ Syntactical Errors\n- " + "\n- ".join(syntactical_errors)
    else:
        syntax_msg = "### ✅ Syntax Check\nNo basic syntactical errors found."

    nmt_token_bd = result.get("nmt_token_breakdown", [])
    nmt_word_bd = result.get("nmt_word_breakdown", [])
    fluency_token_bd = result.get("fluency_token_breakdown", [])
    fluency_word_bd = result.get("fluency_word_breakdown", [])

    return (
        status_msg, sem_text, nmt_text, fluency_text, syntax_msg,
        mapping_data, word_table,
        nmt_token_bd, nmt_word_bd, fluency_token_bd, fluency_word_bd
    )


with gr.Blocks(title="FrenchTutor Engine") as demo:
    gr.Markdown("# 🇫🇷 FrenchTutor: Translation Mapping & Evaluation System")

    mapping_data_state = gr.State(None)
    # State holds a list of selected source word indices for multi-word selection
    selected_source_indices = gr.State([])

    with gr.Tab("🎯 Learning & Mapping"):
        with gr.Row():
            with gr.Column(scale=1):
                sentence_dropdown = gr.Dropdown(
                    choices=get_sentences_choices(),
                    label="Select a Spanish Sentence",
                    interactive=True
                )
                source_text = gr.Textbox(
                    label="Spanish Source Sentence",
                    interactive=True,
                    placeholder="Select a sentence above or type your own..."
                )
                reference_text = gr.Textbox(
                    label="Reference Translation (French)",
                    visible=False
                )

                student_input = gr.Textbox(
                    label="Your French Translation",
                    placeholder="Type your translation here..."
                )
                eval_btn = gr.Button("⚡ Evaluate & Map Words", variant="primary", size="lg")

                status_output = gr.Markdown()

                with gr.Row():
                    score_sem = gr.Textbox(label="🔵 Semantic Adequacy", interactive=False)
                    score_nmt = gr.Textbox(label="🟡 NMT Perplexity Ratio", interactive=False)
                    score_fluency = gr.Textbox(label="🟢 Monolingual Fluency Ratio", interactive=False)
                    
                syntax_output = gr.Markdown()

            with gr.Column(scale=1):
                gr.Markdown("### 🔗 NMT Cross-Attention Word Mapping")
                gr.Markdown(
                    "Click one or **multiple source words** (Spanish) below to highlight their aligned "
                    "**target words** (French) according to the cross-attention matrix. "
                    "Contractions like *J'aime* are dynamically separated (*J'* and *aime*)."
                )

                @gr.render(inputs=[mapping_data_state, selected_source_indices])
                def render_mapping(mapping_data, selected_indices):
                    if mapping_data is None:
                        gr.Markdown("*Submit a translation above to view the cross-attention word alignment.*")
                        return

                    src_words, tgt_words, attn_matrix = mapping_data

                    if not src_words or not tgt_words:
                        gr.Markdown("*No words to display. Try a longer sentence.*")
                        return

                    gr.Markdown("#### 🇪🇸 Source Words (click one or multiple to select)")

                    def toggle_word(current_indices, idx_to_toggle):
                        if current_indices is None:
                            current_indices = []
                        if idx_to_toggle in current_indices:
                            return [i for i in current_indices if i != idx_to_toggle]
                        else:
                            return current_indices + [idx_to_toggle]

                    def select_all_words():
                        return list(range(len(src_words)))

                    def clear_all_words():
                        return []

                    with gr.Row(equal_height=True):
                        for i, word in enumerate(src_words):
                            is_selected = i in (selected_indices or [])
                            variant = "primary" if is_selected else "secondary"
                            btn = gr.Button(word, variant=variant, size="sm")
                            btn.click(
                                toggle_word,
                                inputs=[selected_source_indices, gr.State(i)],
                                outputs=[selected_source_indices]
                            )

                    with gr.Row():
                        select_all_btn = gr.Button("Select All Words", size="sm")
                        clear_btn = gr.Button("Clear Selection", size="sm")

                        select_all_btn.click(select_all_words, outputs=[selected_source_indices])
                        clear_btn.click(clear_all_words, outputs=[selected_source_indices])

                    gr.Markdown("#### 🇫🇷 Target Words (highlighted = aligned)")

                    # Dynamically calculate aligned target word indices using high-confidence algorithm
                    mapped_tgt_indices = get_aligned_targets(
                        selected_indices or [], attn_matrix, len(tgt_words)
                    )

                    with gr.Row(equal_height=True):
                        for j, word in enumerate(tgt_words):
                            if j in mapped_tgt_indices:
                                css = (
                                    "background:#ffd54f;font-weight:bold;padding:5px 12px;"
                                    "border-radius:6px;border:2px solid #ffb300;display:inline-block;"
                                    "margin:3px;font-size:16px;color:#111;"
                                )
                            else:
                                css = (
                                    "padding:5px 12px;border-radius:6px;"
                                    "border:1px solid #ccc;display:inline-block;"
                                    "margin:3px;background:#f5f5f5;font-size:16px;color:#555;"
                                )
                            gr.HTML(f"<span style='{css}'>{word}</span>")

        gr.Markdown("---")
        gr.Markdown("### 📋 Word-to-Word Alignment Table")
        gr.Markdown(
            "This table maps every **source word (Spanish)** to its best-aligned **target word(s) (French)** "
            "according to the cross-attention matrix. Supports 1-to-1, 1-to-many, many-to-1, and many-to-many alignments."
        )

        word_mapping_table = gr.Dataframe(
            headers=["Spanish Word (Source)", "French Word(s) (Target)"],
            datatype=["str", "str"],
            value=[],
            interactive=False,
            wrap=True,
            label="Word Alignment Map (Source → Target)"
        )

        gr.Markdown("---")
        gr.Markdown("## 📊 Token & Word Perplexity & Surprise Breakdowns")

        nmt_tok_state = gr.State([])
        nmt_word_state = gr.State([])
        fluency_tok_state = gr.State([])
        fluency_word_state = gr.State([])

        with gr.Accordion("🟡 NMT Conditional Perplexity & Surprise Breakdown", open=True):
            gr.Markdown(
                "Breaks down NMT cross-entropy loss, token/word perplexity (PPL), generation probability (%), "
                "and **Context-Adaptive Relative Surprise Margin ($\Delta L$)** relative to the model's top prediction. "
                "Units with $\Delta L > 8.0$ are highlighted as **High Surprise Errors** ⚠️."
            )
            nmt_view_mode = gr.Radio(
                ["Word View", "Token View"],
                label="Display View Unit",
                value="Word View",
                interactive=True
            )
            nmt_badges_output = gr.HTML()
            nmt_df_output = gr.Dataframe(
                headers=["Unit", "Perplexity (PPL)", "Probability (%)", "Loss (CE)", "Top Model Prediction", "Relative Surprise (ΔL)", "Status"],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                value=[],
                interactive=False,
                wrap=True,
                label="NMT Per-Unit Breakdown Table"
            )

        with gr.Accordion("🟢 Monolingual Target Fluency Breakdown", open=True):
            gr.Markdown(
                "Breaks down Causal LM (DistilGPT2) target language fluency, perplexity (PPL), generation probability (%), "
                "and **Context-Adaptive Relative Surprise Margin ($\Delta L$)**. Flagged units represent grammatical or syntactic anomalies in French."
            )
            fluency_view_mode = gr.Radio(
                ["Word View", "Token View"],
                label="Display View Unit",
                value="Word View",
                interactive=True
            )
            fluency_badges_output = gr.HTML()
            fluency_df_output = gr.Dataframe(
                headers=["Unit", "Perplexity (PPL)", "Probability (%)", "Loss (CE)", "Top Model Prediction", "Relative Surprise (ΔL)", "Status"],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                value=[],
                interactive=False,
                wrap=True,
                label="Fluency Per-Unit Breakdown Table"
            )

        def update_nmt_view(mode, tok_bd, word_bd):
            is_word = (mode == "Word View")
            bd_list = word_bd if is_word else tok_bd
            html = render_breakdown_badges(bd_list, is_word_view=is_word)
            df_rows = build_breakdown_dataframe(bd_list, is_word_view=is_word)
            return html, df_rows

        def update_fluency_view(mode, tok_bd, word_bd):
            is_word = (mode == "Word View")
            bd_list = word_bd if is_word else tok_bd
            html = render_breakdown_badges(bd_list, is_word_view=is_word)
            df_rows = build_breakdown_dataframe(bd_list, is_word_view=is_word)
            return html, df_rows

        nmt_view_mode.change(
            update_nmt_view,
            inputs=[nmt_view_mode, nmt_tok_state, nmt_word_state],
            outputs=[nmt_badges_output, nmt_df_output]
        )

        fluency_view_mode.change(
            update_fluency_view,
            inputs=[fluency_view_mode, fluency_tok_state, fluency_word_state],
            outputs=[fluency_badges_output, fluency_df_output]
        )

        sentence_dropdown.change(
            on_sentence_select,
            inputs=[sentence_dropdown],
            outputs=[source_text, reference_text, student_input]
        )

        def on_eval_click(source_text, reference_text, student_text, nmt_view_m, fluency_view_m):
            (
                status_msg, sem_text, nmt_text, fluency_text, syntax_msg,
                mapping_data, word_table,
                nmt_tok_bd, nmt_word_bd, fluency_tok_bd, fluency_word_bd
            ) = evaluate_translation(source_text, reference_text, student_text)

            nmt_html, nmt_df = update_nmt_view(nmt_view_m, nmt_tok_bd, nmt_word_bd)
            fluency_html, fluency_df = update_fluency_view(fluency_view_m, fluency_tok_bd, fluency_word_bd)

            return (
                status_msg, sem_text, nmt_text, fluency_text, syntax_msg,
                mapping_data, word_table,
                nmt_tok_bd, nmt_word_bd, fluency_tok_bd, fluency_word_bd,
                nmt_badges_output, nmt_df_output, fluency_badges_output, fluency_df_output
            )

        eval_btn.click(
            on_eval_click,
            inputs=[source_text, reference_text, student_input, nmt_view_mode, fluency_view_mode],
            outputs=[
                status_output, score_sem, score_nmt, score_fluency, syntax_output,
                mapping_data_state, word_mapping_table,
                nmt_tok_state, nmt_word_state, fluency_tok_state, fluency_word_state,
                nmt_badges_output, nmt_df_output, fluency_badges_output, fluency_df_output
            ]
        )

    with gr.Tab("📚 Sentence Management"):
        gr.Markdown("### Consult, Manage, Edit, and Create Sentence Pairs")

        df_component = gr.Dataframe(
            headers=["id", "es", "fr"],
            datatype=["number", "str", "str"],
            value=get_dataframe_data(),
            interactive=True,
            column_count=(3, "fixed")
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### ➕ Create New Sentence Pair")
                new_es = gr.Textbox(label="Spanish Sentence (es)", placeholder="e.g. Me gusta la comida.")
                new_fr = gr.Textbox(label="French Sentence (fr)", placeholder="e.g. J'aime le repas.")
                add_btn = gr.Button("Add Pair", variant="primary")

            with gr.Column():
                gr.Markdown("#### ❌ Remove Sentence Pair")
                delete_id_input = gr.Number(
                    label="Sentence ID to Delete",
                    precision=0,
                    minimum=1,
                    step=1,
                    placeholder="Enter the ID number..."
                )
                delete_btn = gr.Button("🗑️ Delete Pair", variant="stop")

        mgmt_status = gr.Markdown()
        save_all_btn = gr.Button("💾 Save All Table Changes to File")

        def add_pair(es, fr):
            if not es or not fr:
                return (
                    "⚠️ Error: Both Spanish and French sentences are required.",
                    get_dataframe_data(),
                    gr.update(choices=get_sentences_choices()),
                    "",
                    ""
                )
            data_manager.add_sentence(es, fr)
            return (
                "✅ New pair added successfully!",
                get_dataframe_data(),
                gr.update(choices=get_sentences_choices()),
                "",
                ""
            )

        def delete_pair(pair_id):
            if pair_id is None or pair_id == 0:
                return (
                    "⚠️ Error: Enter a valid sentence ID to delete.",
                    get_dataframe_data(),
                    gr.update(choices=get_sentences_choices())
                )
            try:
                success = data_manager.delete_sentence(int(pair_id))
                if success:
                    msg = f"✅ Pair ID {int(pair_id)} deleted successfully!"
                else:
                    msg = f"⚠️ No sentence found with ID {int(pair_id)}."
            except Exception as e:
                msg = f"❌ Error: {e}"
            return (
                msg,
                get_dataframe_data(),
                gr.update(choices=get_sentences_choices())
            )

        def save_table(df_val):
            data_manager.sentences = []
            rows = []
            if isinstance(df_val, pd.DataFrame):
                rows = df_val.values.tolist()
            elif isinstance(dict):
                rows = df_val.get('data', [])
            elif isinstance(df_val, list):
                rows = df_val

            for row in rows:
                try:
                    s_id = int(row[0])
                    es = str(row[1])
                    fr = str(row[2])
                    data_manager.sentences.append({'id': s_id, 'es': es, 'fr': fr})
                except Exception:
                    continue
            data_manager.save_data()
            return (
                "✅ All changes saved to sentences.json!",
                get_dataframe_data(),
                gr.update(choices=get_sentences_choices())
            )

        add_btn.click(
            add_pair,
            inputs=[new_es, new_fr],
            outputs=[mgmt_status, df_component, sentence_dropdown, new_es, new_fr]
        )

        delete_btn.click(
            delete_pair,
            inputs=[delete_id_input],
            outputs=[mgmt_status, df_component, sentence_dropdown]
        )

        save_all_btn.click(
            save_table,
            inputs=[df_component],
            outputs=[mgmt_status, df_component, sentence_dropdown]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865)
