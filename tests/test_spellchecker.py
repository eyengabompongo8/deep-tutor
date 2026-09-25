from backend.nlp.text_utils import autocorrect_text, remove_accents, levenshtein


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

    def test_autocorrect_reconciles_terminal_punctuation(self):
        student = "Comment vas-tu"
        ref = "Comment vas-tu ?"
        corrected, msgs = autocorrect_text(student, ref)
        assert corrected == "Comment vas-tu ?"
        assert "added '?'" in msgs

    def test_autocorrect_french_punctuation_spacing(self):
        student = "C'est bien!"
        ref = "C'est bien !"
        corrected, msgs = autocorrect_text(student, ref)
        assert corrected == "C'est bien !"

    def test_autocorrect_does_not_override_words_from_reference(self):
        student = "Elle a appris a nager."
        ref = "Il a appris à nager."
        corrected, msgs = autocorrect_text(student, ref)
        # Without LanguageTool, only punctuation reconciliation happens, words/accents are not blindly rewritten from ref
        assert corrected == "Elle a appris a nager."
        assert msgs == []

    def test_autocorrect_ignores_french_contractions(self):
        student = "j'ai vu l'arbre."
        ref = "j'ai vu l'arbre."
        corrected, msgs = autocorrect_text(student, ref)
        assert corrected == student
        assert msgs == []

    def test_autocorrect_with_mock_languagetool_typos(self):
        class MockMatch:
            def __init__(self, offset, error_length, replacements, category, rule_issue_type, rule_id):
                self.offset = offset
                self.error_length = error_length
                self.replacements = replacements
                self.category = category
                self.rule_issue_type = rule_issue_type
                self.rule_id = rule_id

        class MockLanguageTool:
            def check(self, text):
                matches = []
                idx = text.find("pome")
                if idx != -1:
                    matches.append(MockMatch(idx, 4, ["pomme"], "TYPOS", "misspelling", "MORFOLOGIK_RULE_FR"))
                # Grammar error: should NOT be touched by Tier 1
                idx_g = text.find("mange")
                if idx_g != -1 and "Ils" in text:
                    matches.append(MockMatch(idx_g, 5, ["mangent"], "GRAMMAR", "grammar", "ACCORD_SUJET_VERBE"))
                return matches

        tool = MockLanguageTool()
        student = "Ils mange une pome."
        ref = "Ils mangent une pomme."
        corrected, msgs = autocorrect_text(student, ref, lang_tool=tool)
        # Typo is corrected
        assert "pomme" in corrected
        # Grammar mistake is preserved so Tier 2 can penalize it
        assert "Ils mange" in corrected
        assert "pome -> pomme" in msgs

    def test_autocorrect_no_corrections_needed(self):
        student = "Bonjour le monde."
        ref = "Bonjour le monde."
        corrected, msgs = autocorrect_text(student, ref)
        assert corrected == student
        assert msgs == []

    def test_autocorrect_with_mock_languagetool_capitalization(self):
        class MockMatch:
            def __init__(self, offset, error_length, replacements, category, rule_issue_type, rule_id):
                self.offset = offset
                self.error_length = error_length
                self.replacements = replacements
                self.category = category
                self.rule_issue_type = rule_issue_type
                self.rule_id = rule_id

        class MockLanguageTool:
            def check(self, text):
                matches = []
                if text.startswith("bonjour"):
                    matches.append(MockMatch(0, 7, ["Bonjour"], "CASING", "typographical", "UPPERCASE_SENTENCE_START"))
                if "j'" in text:
                    idx = text.find("j'")
                    matches.append(MockMatch(idx, 2, ["J'"], "CASING", "typographical", "UPPERCASE_SENTENCE_START"))
                if "paris" in text:
                    idx = text.find("paris")
                    matches.append(MockMatch(idx, 5, ["Paris"], "CASING", "typographical", "PROPER_NOUN_CASING"))
                return matches

        tool = MockLanguageTool()
        student = "bonjour le monde. j'habite à paris"
        ref = "Bonjour le monde. J'habite à Paris."
        corrected, msgs = autocorrect_text(student, ref, lang_tool=tool)
        assert corrected == "Bonjour le monde. J'habite à Paris."
        assert "bonjour -> Bonjour" in msgs
        assert "j' -> J'" in msgs
        assert "paris -> Paris" in msgs
        assert "added '.'" in msgs

