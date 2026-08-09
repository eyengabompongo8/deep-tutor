import sys
from data_manager import DataManager
from models import MLEngine
import app

def run_tests():
    print("--- 1. Testing DataManager ---")
    dm = DataManager()
    sentences = dm.get_all_sentences()
    print(f"Loaded {len(sentences)} sentence pairs.")
    assert len(sentences) > 0, "No sentences loaded!"
    
    print("\n--- 2. Testing Word Mapping & Strict Correction ---")
    engine = MLEngine()
    
    src = "Tengo que irme ahora."
    ref = "Je dois partir maintenant."
    wrong_student = "Je mange de poisson."
    correct_student = "Je dois partir maintenant."
    
    print(f"\nEvaluating WRONG translation: '{wrong_student}' for '{src}'...")
    res_wrong = engine.compute_final_score(src, wrong_student, ref)
    print(f"Semantic Score: {res_wrong['s_sem']:.2f}")
    assert res_wrong['s_sem'] < 50.0, f"Expected <50.0 for wrong translation, got {res_wrong['s_sem']}"
    print("✅ WRONG translation correctly scored low semantic score!")
    
    print(f"\nEvaluating CORRECT translation: '{correct_student}' for '{src}'...")
    res_correct = engine.compute_final_score(src, correct_student, ref)
    print(f"Semantic Score: {res_correct['s_sem']:.2f}")
    assert res_correct['s_sem'] > 85.0, f"Expected >85.0 for correct translation, got {res_correct['s_sem']}"
    print("✅ CORRECT translation scored high!")
    
    src_words, tgt_words, word_attn = res_correct["mapping_data"]
    print(f"\nSource Words: {src_words}")
    print(f"Target Words: {tgt_words}")
    print(f"Word Attention Matrix Shape: {word_attn.shape}")
    assert "Tengo" in src_words and "irme" in src_words, "Subtokens not cleaned into real words!"
    assert "__es__" not in src_words and "</s>" not in src_words, "Special subtokens leaked into word selection!"
    print("✅ WORD SELECTION CLEAN (no subtokens)!")

if __name__ == "__main__":
    run_tests()
