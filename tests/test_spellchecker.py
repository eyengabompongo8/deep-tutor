import pytest
from server import autocorrect_text, remove_accents, levenshtein


class TestSpellcheckerAndAccents:
    def test_remove_accents(self):
        assert remove_accents("élève") == "eleve"
        assert remove_accents("garçon") == "garcon"
        assert remove_accents("ça va bien où ?") == "ca va bien ou ?"

    def test_levenshtein_distance(self):
        assert levenshtein("chat", "chat") == 0
        assert levenshtein("chat", "chatte") == 2
        assert levenshtein("garcon", "garçon") == 1
        assert levenshtein("", "abc") == 3

    def test_autocorrect_empty_or_none(self):
        corrected, msgs = autocorrect_text("", "Bonjour")
        assert corrected == ""
        assert msgs == []

    def test_autocorrect_accents_from_reference(self):
        student = "J'aime l'eleve et le garcon."
        ref = "J'aime l'élève et le garçon."
        corrected, msgs = autocorrect_text(student, ref)
        assert corrected == "J'aime l'élève et le garçon."
        assert "eleve -> élève" in msgs
        assert "garcon -> garçon" in msgs

    def test_autocorrect_preserves_capitalization(self):
        student = "Eleve et Garcon"
        ref = "Élève et Garçon"
        corrected, msgs = autocorrect_text(student, ref)
        assert corrected == "Élève et Garçon"
        assert any("Eleve" in m for m in msgs)
        assert any("Garcon" in m for m in msgs)

    def test_autocorrect_ignores_french_contractions(self):
        student = "j'ai vu l'arbre"
        ref = "j'ai vu l'arbre"
        corrected, msgs = autocorrect_text(student, ref)
        assert corrected == student
        assert msgs == []

    def test_autocorrect_no_corrections_needed(self):
        student = "Bonjour le monde"
        ref = "Bonjour le monde"
        corrected, msgs = autocorrect_text(student, ref)
        assert corrected == student
        assert msgs == []
