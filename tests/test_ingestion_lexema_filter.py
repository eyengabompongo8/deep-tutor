import os
import pytest
import sqlite3
from backend.db import DataManager
from backend.nlp.vocab_engine import VocabEngine, get_vocab_engine
from backend.ingestion.pipeline.ingest_and_link import run_ingest_and_link, run_clear_db


@pytest.fixture
def temp_db(tmp_path):
    """Creates a temporary test database with the schema."""
    db_file = tmp_path / "test_ingestion.db"
    dm = DataManager(db_path=str(db_file))
    return dm, str(db_file)


class TestIngestionLexemaFilter:

    def test_filter_empty_lexemas_table(self, temp_db):
        """When the lexemas table is completely empty, all sentences are skipped without error."""
        dm, db_path = temp_db
        engine = VocabEngine()
        
        candidates = [
            {"fr": "Bonjour le monde.", "es": "Hola mundo.", "level": "A1", "score_lexical": 1.0, "score_ml": 1.0},
            {"fr": "Le chat dort.", "es": "El gato duerme.", "level": "A1", "score_lexical": 1.0, "score_ml": 1.0},
        ]
        
        accepted, skipped = engine.filter_sentences_with_complete_lexemas(candidates, dm, show_progress=False)
        assert len(accepted) == 0
        assert skipped == 2
        
        # Test full pipeline execution with empty lexemas
        success = run_ingest_and_link(candidates, db_path=db_path)
        assert success is True
        assert len(dm.get_all_sentences()) == 0

    def test_filter_sentences_all_words_present(self, temp_db):
        """Sentences where every word has a matching lexema in the database are accepted and linked."""
        dm, db_path = temp_db
        engine = VocabEngine()

        # Seed specific lexemas
        lexemas_to_seed = [
            {"lemma": "le", "pos": "DET", "level": "A1", "is_base": 1},
            {"lemma": "chat", "pos": "NOUN", "level": "A1", "is_base": 0},
            {"lemma": "dort", "pos": "ADJ", "level": "A1", "is_base": 0},
        ]
        dm.bulk_upsert_lexemas(lexemas_to_seed)

        candidates = [
            {
                "fr": "Le chat dort.",
                "es": "El gato duerme.",
                "level": "A1",
                "score_lexical": 1.0,
                "score_ml": 1.0,
                "source": "unit_test"
            }
        ]

        accepted, skipped = engine.filter_sentences_with_complete_lexemas(candidates, dm, show_progress=False)
        assert len(accepted) == 1
        assert skipped == 0

        # Ingest and link
        success = run_ingest_and_link(candidates, db_path=db_path)
        assert success is True

        stored = dm.get_all_sentences()
        assert len(stored) == 1
        assert stored[0]["fr"] == "Le chat dort."

        # Verify relational links exist
        links = dm.get_sentence_lexemas(stored[0]["id"])
        lemmas_linked = {l["lemma"] for l in links}
        assert "le" in lemmas_linked
        assert "chat" in lemmas_linked
        assert "dort" in lemmas_linked

    def test_filter_sentences_with_missing_word_skipped(self, temp_db):
        """Sentences containing even a single word without a DB lexema are completely skipped."""
        dm, db_path = temp_db
        engine = VocabEngine()

        # Seed lexemas for "le", "chat", "dort" (dormir), but NOT "mystérieux" or "glubx"
        dm.bulk_upsert_lexemas([
            {"lemma": "le", "pos": "DET", "level": "A1", "is_base": 1},
            {"lemma": "chat", "pos": "NOUN", "level": "A1", "is_base": 0},
            {"lemma": "dort", "pos": "ADJ", "level": "A1", "is_base": 0},
        ])

        candidates = [
            # Valid: 100% words present
            {"fr": "Le chat dort.", "es": "El gato duerme.", "level": "A1", "score_lexical": 1.0},
            # Invalid: 'inconnu' / 'glubx' missing
            {"fr": "Le chat glubx dort.", "es": "El gato raro duerme.", "level": "A1", "score_lexical": 1.0},
            # Invalid: 'arbre' missing
            {"fr": "Le chat est sur un arbre.", "es": "El gato está en un árbol.", "level": "A1", "score_lexical": 1.0},
        ]

        accepted, skipped = engine.filter_sentences_with_complete_lexemas(candidates, dm, show_progress=False)
        assert len(accepted) == 1
        assert skipped == 2
        assert accepted[0]["fr"] == "Le chat dort."

        # Run full pipeline
        run_ingest_and_link(candidates, db_path=db_path)
        stored = dm.get_all_sentences()
        assert len(stored) == 1
        assert stored[0]["fr"] == "Le chat dort."

    def test_contractions_and_elisions_resolution(self, temp_db):
        """Contractions like l', j', d', c', qu' correctly resolve to base lexemas."""
        dm, db_path = temp_db
        engine = VocabEngine()

        # Seed lexemas
        dm.bulk_upsert_lexemas([
            {"lemma": "je", "pos": "PRON", "level": "A1", "is_base": 1},
            {"lemma": "aimer", "pos": "VERB", "level": "A1", "is_base": 0},
            {"lemma": "le", "pos": "DET", "level": "A1", "is_base": 1},
            {"lemma": "arbre", "pos": "NOUN", "level": "A1", "is_base": 0},
        ])

        # "J'aime l'arbre." -> j' (je), aime (aimer), l' (le), arbre (arbre)
        sentence = {"fr": "J'aime l'arbre.", "es": "Me gusta el árbol.", "level": "A1", "score_lexical": 1.0}
        accepted, skipped = engine.filter_sentences_with_complete_lexemas([sentence], dm, show_progress=False)
        assert len(accepted) == 1
        assert skipped == 0

        run_ingest_and_link([sentence], db_path=db_path)
        stored = dm.get_all_sentences()
        assert len(stored) == 1
        links = dm.get_sentence_lexemas(stored[0]["id"])
        lemmas_linked = {l["lemma"] for l in links}
        assert "je" in lemmas_linked
        assert "aimer" in lemmas_linked
        assert "le" in lemmas_linked
        assert "arbre" in lemmas_linked

    def test_punctuation_numbers_spaces_ignored(self, temp_db):
        """Punctuation, digits, quotes, and whitespace do not require lexemas."""
        dm, db_path = temp_db
        engine = VocabEngine()

        dm.bulk_upsert_lexemas([
            {"lemma": "oui", "pos": "ADV", "level": "A1", "is_base": 1},
            {"lemma": "merci", "pos": "NOUN", "level": "A1", "is_base": 1},
        ])

        sentence = {"fr": "« Oui, merci ! » 42...", "es": "¡Sí, gracias! 42...", "level": "A1", "score_lexical": 1.0}
        accepted, skipped = engine.filter_sentences_with_complete_lexemas([sentence], dm, show_progress=False)
        assert len(accepted) == 1
        assert skipped == 0

    def test_run_clear_db(self, temp_db):
        """run_clear_db clears sentences and junction links."""
        dm, db_path = temp_db
        dm.bulk_upsert_lexemas([
            {"lemma": "bonjour", "pos": "INTJ", "level": "A1", "is_base": 1}
        ])
        run_ingest_and_link([{"fr": "Bonjour !", "es": "¡Hola!", "level": "A1", "score_lexical": 1.0}], db_path=db_path)
        assert len(dm.get_all_sentences()) == 1
        
        run_clear_db(db_path=db_path)
        assert len(dm.get_all_sentences()) == 0

    def test_is_sentence_fully_covered_direct(self, temp_db):
        """Direct unit test for VocabEngine.is_sentence_fully_covered."""
        dm, _ = temp_db
        engine = VocabEngine()

        lemma_pos_to_id = {("maison", "NOUN"): 1, ("grand", "ADJ"): 2, ("le", "DET"): 3, ("être", "AUX"): 4}
        lemma_fallback_to_id = {"maison": 1, "grand": 2, "le": 3, "être": 4}

        # Empty string
        valid, ids = engine.is_sentence_fully_covered("", lemma_pos_to_id, lemma_fallback_to_id)
        assert valid is False
        assert ids == []

        # Only punctuation / digits
        valid, ids = engine.is_sentence_fully_covered("123 ... ???", lemma_pos_to_id, lemma_fallback_to_id)
        assert valid is False
        assert ids == []

        # Fully covered
        valid, ids = engine.is_sentence_fully_covered("La maison est grande.", lemma_pos_to_id, lemma_fallback_to_id)
        assert valid is True
        assert len(ids) == 4

        # Partially covered
        valid, ids = engine.is_sentence_fully_covered("La maison rouge est grande.", lemma_pos_to_id, lemma_fallback_to_id)
        assert valid is False

    def test_batch_processing_lexema_filter(self, temp_db):
        """Validates that chunked batch processing preserves correct indexing and filtering."""
        dm, db_path = temp_db
        engine = VocabEngine()

        dm.bulk_upsert_lexemas([
            {"lemma": "le", "pos": "DET", "level": "A1", "is_base": 1},
            {"lemma": "chat", "pos": "NOUN", "level": "A1", "is_base": 0},
            {"lemma": "chien", "pos": "NOUN", "level": "A1", "is_base": 0},
            {"lemma": "dort", "pos": "ADJ", "level": "A1", "is_base": 0},
        ])

        candidates = [
            {"fr": "Le chat dort.", "es": "El gato duerme.", "level": "A1", "score_lexical": 1.0},
            {"fr": "Le chien dort.", "es": "El perro duerme.", "level": "A1", "score_lexical": 1.0},
            {"fr": "Le crocodile dort.", "es": "El cocodrilo duerme.", "level": "A1", "score_lexical": 1.0},
            {"fr": "Le chat et le chien dorment.", "es": "El gato y el perro duermen.", "level": "A1", "score_lexical": 1.0}, # 'et' missing
        ]

        # Use batch_size=2 to test boundary conditions
        accepted, skipped = engine.filter_sentences_with_complete_lexemas(candidates, dm, batch_size=2, show_progress=False)
        assert len(accepted) == 2
        assert skipped == 2
        assert [s["fr"] for s in accepted] == ["Le chat dort.", "Le chien dort."]

    def test_seeded_catalog_integration(self, temp_db):
        """Integration test with the real seeded lexemas from FLELex."""
        from backend.ingestion.pipeline.seed_lexemas import run_seed_lexemas
        dm, db_path = temp_db
        
        # Seed the temporary database using the actual FLELex catalog
        seeded = run_seed_lexemas(db_path=db_path)
        assert seeded is True

        candidates = [
            # Common everyday French sentences
            {"fr": "Je mange une pomme.", "es": "Como una manzana.", "level": "A1", "score_lexical": 1.0},
            {"fr": "Je voudrais un café s'il vous plaît.", "es": "Quisiera un café por favor.", "level": "A1", "score_lexical": 1.0},
            # Sentence with invented / non-existent French words
            {"fr": "Le blubzorp florpait tranquillement.", "es": "El alien caminaba tranquilamente.", "level": "A1", "score_lexical": 1.0},
        ]

        success = run_ingest_and_link(candidates, db_path=db_path)
        assert success is True

        stored = dm.get_all_sentences()
        stored_texts = [s["fr"] for s in stored]
        assert "Je mange une pomme." in stored_texts
        assert "Je voudrais un café s'il vous plaît." in stored_texts
        assert "Le blubzorp florpait tranquillement." not in stored_texts

