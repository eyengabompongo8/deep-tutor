import pytest
from unittest.mock import patch, MagicMock
import pymupdf

from backend.ingestion.pipeline.common import (
    format_sentence,
    is_valid_sentence,
    extract_sentences_from_pdf,
    deduplicate_against_db,
    process_and_score_sentences,
)


def test_format_sentence():
    raw = "Bonjour   le monde ! Comment  ça  va ?"
    formatted = format_sentence(raw)
    assert formatted == "Bonjour le monde ! Comment ça va ?"


def test_is_valid_sentence():
    # Valid sentences
    assert is_valid_sentence("Je vais au supermarché tous les matins.")
    assert is_valid_sentence("Comment vas-tu aujourd'hui ?")
    assert is_valid_sentence("Quelle belle journée !")

    # Too short (< 3 words)
    assert not is_valid_sentence("Bonjour !")
    assert not is_valid_sentence("A.")

    # Contains digits
    assert not is_valid_sentence("J'ai 25 ans.")

    # Does not start with uppercase
    assert not is_valid_sentence("je m'appelle Paul.")

    # Missing terminal punctuation
    assert not is_valid_sentence("Je vais au marché")

    # Spaced hyphens or dashes
    assert not is_valid_sentence("Voici un exemple - et un autre.")

    # More than 50% uppercase
    assert not is_valid_sentence("JE SUIS TROP GRAND.")


def test_extract_sentences_from_pdf_with_stop_phrase():
    # Create an in-memory PDF with PyMuPDF
    doc = pymupdf.open()
    page = doc.new_page()
    text_content = (
        "Je m'appelle Jessica. Je suis une fille, je suis française.\n"
        "Questions de compréhension:\n"
        "1) Quel âge a Jessica ?\n"
    )
    page.insert_text((50, 50), text_content)
    pdf_bytes = doc.tobytes()
    doc.close()

    sentences = extract_sentences_from_pdf(pdf_bytes, stop_phrase="Questions de compréhension:")
    # Should extract text before Questions de compréhension:
    assert len(sentences) > 0
    for s in sentences:
        assert "Questions" not in s
        assert "Quel âge" not in s


def test_deduplicate_against_db():
    mock_dm = MagicMock()
    mock_dm.get_all_sentences.return_value = [
        {"fr": "Je m'appelle Jessica."}
    ]

    candidates = [
        "Je m'appelle Jessica.",  # Duplicate of DB
        "Je suis une fille très gentille.",  # Unique 1
        "Je suis une fille très gentille.",  # Duplicate in batch
        "Une autre phrase complètement nouvelle."  # Unique 2
    ]

    unique = deduplicate_against_db(candidates, data_manager=mock_dm)
    assert len(unique) == 2
    assert "Je m'appelle Jessica." not in unique
    assert unique[0] == "Je suis une fille très gentille."
    assert unique[1] == "Une autre phrase complètement nouvelle."


@patch("backend.ingestion.pipeline.common.translate_batch")
def test_process_and_score_sentences(mock_translate_batch):
    mock_translate_batch.return_value = ["Hola mundo."]

    mock_engine = MagicMock()
    mock_engine.score_sentences_ml_batch.return_value = [
        {"level": "A1", "score": 1.15}
    ]
    mock_engine.score_sentence.return_value = ("A1", 0.95, {})

    records = process_and_score_sentences(
        ["Bonjour le monde."],
        source_name="test_source",
        source_metadata={"file": "test.pdf"},
        engine=mock_engine
    )

    assert len(records) == 1
    rec = records[0]
    assert rec["fr"] == "Bonjour le monde."
    assert rec["es"] == "Hola mundo."
    assert rec["level"] == "A1"
    assert rec["score"] == 1.15
    assert rec["score_lexical"] == 0.95
    assert rec["score_ml"] == 1.15
    assert rec["source"] == "test_source"
    assert rec["metadata"] == {"file": "test.pdf"}


def test_deduplicate_candidate_records():
    mock_dm = MagicMock()
    mock_dm.get_all_sentences.return_value = [
        {"fr": "Je m'appelle Jessica."}
    ]

    candidates = [
        {"fr": "Je m'appelle Jessica.", "es": "Me llamo Jessica.", "source": "tatoeba_parallel"},
        {"fr": "Je suis une fille très gentille.", "es": None, "source": "mono"},
        # Duplicate with Spanish translation should take priority over the one without
        {"fr": "Je suis une fille très gentille.", "es": "Soy una chica muy simpática.", "source": "parallel"},
        {"fr": "Une autre phrase complètement nouvelle.", "es": None, "source": "pdf"},
    ]

    from backend.ingestion.pipeline.common import deduplicate_candidate_records
    unique = deduplicate_candidate_records(candidates, data_manager=mock_dm)

    assert len(unique) == 2
    # Ensure the duplicate that survived is the one with Spanish translation
    simp = next(r for r in unique if r["fr"] == "Je suis une fille très gentille.")
    assert simp["es"] == "Soy una chica muy simpática."
    assert simp["source"] == "parallel"

    autre = next(r for r in unique if r["fr"] == "Une autre phrase complètement nouvelle.")
    assert autre["es"] is None


@patch("backend.ingestion.pipeline.common.translate_batch")
def test_preprocess_and_score_candidates_selective_translation(mock_translate_batch):
    mock_translate_batch.return_value = ["Soy una chica muy simpática."]

    mock_engine = MagicMock()
    mock_engine.score_sentences_ml_batch.return_value = [
        {"level": "A1", "score": 1.1},
        {"level": "A2", "score": 2.2},
    ]
    mock_engine.score_sentence.side_effect = [
        ("A1", 1.0, {}),
        ("A2", 2.0, {}),
    ]

    candidates = [
        {"fr": "Bonjour le monde.", "es": "Hola mundo.", "source": "tatoeba_parallel"},
        {"fr": "Je suis une fille très gentille.", "es": None, "source": "pdf"},
    ]

    from backend.ingestion.pipeline.common import preprocess_and_score_candidates
    records = preprocess_and_score_candidates(candidates, engine=mock_engine)

    assert len(records) == 2
    # First record already had 'es', so translate_batch should not be called with it
    assert records[0]["fr"] == "Bonjour le monde."
    assert records[0]["es"] == "Hola mundo."
    assert records[0]["source"] == "tatoeba_parallel"

    # Second record lacked 'es', so it was translated
    assert records[1]["fr"] == "Je suis une fille très gentille."
    assert records[1]["es"] == "Soy una chica muy simpática."
    assert records[1]["source"] == "pdf"

    # Verify translate_batch was only called with the missing sentence
    mock_translate_batch.assert_called_once_with(["Je suis une fille très gentille."], batch_size=16)

