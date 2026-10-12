import os
import math
import tempfile
import sqlite3
import pytest
from unittest.mock import patch

from backend.ingestion.pipeline.lexique_classifier import (
    compute_zipf,
    rank_to_cefr,
    parse_and_classify_lexique,
    CORE_A1_LEMMAS,
)
from backend.ingestion.pipeline.seed_lemmas import run_seed_lemmas
from backend.ingestion.pipeline.seed_lexemas import run_seed_lexemas
from backend.db.connection import ROOT_DIR


def test_compute_zipf():
    # Non-positive frequencies
    assert compute_zipf(0.0) == 0.0
    assert compute_zipf(-5.0) == 0.0

    # Positive frequencies
    assert compute_zipf(1.0) == 3.0
    assert compute_zipf(10.0) == 4.0
    assert compute_zipf(100.0) == 5.0
    assert compute_zipf(0.5) == round(math.log10(0.5) + 3.0, 2)


def test_rank_to_cefr():
    # A1: 1 <= R <= 800
    assert rank_to_cefr(1) == 'A1'
    assert rank_to_cefr(800) == 'A1'

    # A2: 801 <= R <= 1800
    assert rank_to_cefr(801) == 'A2'
    assert rank_to_cefr(1800) == 'A2'

    # B1: 1801 <= R <= 3500
    assert rank_to_cefr(1801) == 'B1'
    assert rank_to_cefr(3500) == 'B1'

    # B2: 3501 <= R <= 6500
    assert rank_to_cefr(3501) == 'B2'
    assert rank_to_cefr(6500) == 'B2'

    # C1: 6501 <= R <= 12000
    assert rank_to_cefr(6501) == 'C1'
    assert rank_to_cefr(12000) == 'C1'

    # C2: R > 12000
    assert rank_to_cefr(12001) == 'C2'
    assert rank_to_cefr(70000) == 'C2'


def test_parse_and_classify_synthetic(tmp_path):
    tsv_content = (
        "1_Mot\t2_Phono\t3_Phono_IPA\t4_Lemme\t5_Cgram\t6_CgramOrtho\t7_Genre\t8_Nombre\t9_InfoVER\t10_FreqMot\t11_FreqOrtho\t12_FreqLemme\t13_CDOrtho\t14_IsLem\n"
        "être\tEtR\tɛtʁ\têtre\tVER\tVER\t\t\tinf\t50.0\t50.0\t100.0\t50.0\t1\n"
        "suis\tsHi\tsɥi\têtre\tVER\tVER\t\ts\tind:pre:1s\t30.0\t30.0\t100.0\t30.0\t0\n"
        "pomme\tpOm\tpɔm\tpomme\tNOM\tNOM\tf\ts\t\t15.0\t15.0\t20.0\t15.0\t1\n"
        "pommes\tpOm\tpɔm\tpomme\tNOM\tNOM\tf\tp\t\t5.0\t5.0\t20.0\t5.0\t0\n"
        "incursion\t5kyRsj5\tɛ̃kyʁsjɔ̃\tincursion\tNOM\tNOM\tf\ts\t\t2.0\t2.0\t2.0\t2.0\t0\n"
        "incursion\t5kyRsj5\tɛ̃kyʁsjɔ̃\tincursion\tNOM\tNOM\tf\ts\t\t1.5\t1.5\t2.0\t1.5\t0\n"
    )
    tsv_file = tmp_path / "Lexique_mini.tsv"
    tsv_file.write_text(tsv_content, encoding="utf-8")

    lemmas, words = parse_and_classify_lexique(str(tsv_file))

    # Expect 3 unique lemmas: être (VER), pomme (NOM), incursion (NOM - non-IsLem)
    assert len(lemmas) == 3

    # Ranking:
    # 1. être (FreqLemme 100.0) -> A1, is_base True
    assert lemmas[0]['lemma'] == 'être'
    assert lemmas[0]['pos'] == 'VER'
    assert lemmas[0]['frequency'] == 100.0
    assert lemmas[0]['level'] == 'A1'
    assert lemmas[0]['is_base'] is True
    assert lemmas[0]['metadata']['rank'] == 1
    assert lemmas[0]['metadata']['zipf'] == compute_zipf(100.0)

    # 2. pomme (FreqLemme 20.0) -> A1, is_base False
    assert lemmas[1]['lemma'] == 'pomme'
    assert lemmas[1]['pos'] == 'NOM'
    assert lemmas[1]['frequency'] == 20.0
    assert lemmas[1]['level'] == 'A1'
    assert lemmas[1]['is_base'] is False
    assert lemmas[1]['metadata']['rank'] == 2

    # 3. incursion (FreqLemme 2.0) -> A1, is_base False
    assert lemmas[2]['lemma'] == 'incursion'
    assert lemmas[2]['pos'] == 'NOM'
    assert lemmas[2]['frequency'] == 2.0
    assert lemmas[2]['level'] == 'A1'
    assert lemmas[2]['metadata']['rank'] == 3

    # Surface words:
    # 'être', 'suis', 'pomme', 'pommes', 'incursion' (deduplicated with max freq 2.0)
    word_map = {(w['word'], w['lemma'], w['pos']): w['frequency'] for w in words}
    assert len(words) == 5
    assert word_map[('être', 'être', 'VER')] == 50.0
    assert word_map[('suis', 'être', 'VER')] == 30.0
    assert word_map[('pomme', 'pomme', 'NOM')] == 15.0
    assert word_map[('pommes', 'pomme', 'NOM')] == 5.0
    assert word_map[('incursion', 'incursion', 'NOM')] == 2.0  # max of 2.0 and 1.5


def test_seed_lemmas_synthetic(tmp_path):
    # Prepare mini dataset
    tsv_content = (
        "1_Mot\t2_Phono\t3_Phono_IPA\t4_Lemme\t5_Cgram\t6_CgramOrtho\t7_Genre\t8_Nombre\t9_InfoVER\t10_FreqMot\t11_FreqOrtho\t12_FreqLemme\t13_CDOrtho\t14_IsLem\n"
        "être\tEtR\tɛtʁ\têtre\tVER\tVER\t\t\tinf\t50.0\t50.0\t100.0\t50.0\t1\n"
        "suis\tsHi\tsɥi\têtre\tVER\tVER\t\ts\tind:pre:1s\t30.0\t30.0\t100.0\t30.0\t0\n"
        "chat\tSa\tʃa\tchat\tNOM\tNOM\tm\ts\t\t12.0\t12.0\t18.0\t12.0\t1\n"
        "chats\tSa\tʃa\tchat\tNOM\tNOM\tm\tp\t\t6.0\t6.0\t18.0\t6.0\t0\n"
    )
    fake_lexique = tmp_path / "Lexique4.tsv"
    fake_lexique.write_text(tsv_content, encoding="utf-8")

    db_path = str(tmp_path / "test_app.db")

    with patch("backend.ingestion.pipeline.seed_lemmas.ROOT_DIR", str(tmp_path)):
        # Make tmp_path/data/lexique/Lexique4/Lexique4.tsv exist
        lexique_dir = tmp_path / "data" / "lexique" / "Lexique4"
        lexique_dir.mkdir(parents=True, exist_ok=True)
        (lexique_dir / "Lexique4.tsv").write_text(tsv_content, encoding="utf-8")

        success = run_seed_lemmas(db_path=db_path)
        assert success is True

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Check lemmas
        lemmas = conn.execute("SELECT * FROM lemmas ORDER BY id;").fetchall()
        assert len(lemmas) == 2
        etre_row = next(r for r in lemmas if r['lemma'] == 'être')
        assert etre_row['pos'] == 'VER'
        assert etre_row['level'] == 'A1'
        assert etre_row['is_base'] == 1

        chat_row = next(r for r in lemmas if r['lemma'] == 'chat')
        assert chat_row['pos'] == 'NOM'
        assert chat_row['level'] == 'A1'
        assert chat_row['is_base'] == 0

        # Check words & foreign key mapping
        words = conn.execute("SELECT * FROM words ORDER BY id;").fetchall()
        assert len(words) == 4
        suis_row = next(r for r in words if r['word'] == 'suis')
        assert suis_row['lemma_id'] == etre_row['id']
        assert suis_row['pos'] == 'VER'
        assert suis_row['frequency'] == 30.0

        chats_row = next(r for r in words if r['word'] == 'chats')
        assert chats_row['lemma_id'] == chat_row['id']
        assert chats_row['pos'] == 'NOM'
        assert chats_row['frequency'] == 6.0

        # Check backward compatibility wrapper run_seed_lexemas
        compat_success = run_seed_lexemas(db_path=db_path)
        assert compat_success is True

        conn.close()


def test_real_lexique_integrity():
    real_tsv = os.path.join(ROOT_DIR, 'data', 'lexique', 'Lexique4', 'Lexique4.tsv')
    if not os.path.exists(real_tsv):
        pytest.skip("Lexique4.tsv not present on filesystem")

    lemmas, words = parse_and_classify_lexique(real_tsv)

    # 1. Total count checks
    assert len(lemmas) > 65000  # 65650 IsLem==1 + ~4078 non-IsLem lemmas
    assert len(words) == 189863

    # 2. Ranking order check
    assert lemmas[0]['frequency'] >= lemmas[1]['frequency']
    assert lemmas[0]['metadata']['rank'] == 1
    assert lemmas[-1]['metadata']['rank'] == len(lemmas)

    # 3. All CEFR levels present
    levels = {l['level'] for l in lemmas}
    assert levels == {'A1', 'A2', 'B1', 'B2', 'C1', 'C2'}

    # 4. Core A1 vocabulary check
    etre_lemmas = [l for l in lemmas if l['lemma'] == 'être']
    assert len(etre_lemmas) > 0
    assert any(l['is_base'] is True for l in etre_lemmas)

    # 5. Every surface word parent (lemma, pos) is in lemmas
    lemma_keys = {(l['lemma'], l['pos']) for l in lemmas}
    sample_words = words[:1000]
    for w in sample_words:
        assert (w['lemma'], w['pos']) in lemma_keys
