import os

# App launch settings
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", os.environ.get("BACKEND_PORT", 8000)))
RELOAD = os.environ.get("RELOAD", "false").lower() in ("1", "true", "yes")

# Default user settings
DEFAULT_CEFR_LEVEL = "A1"
STARTING_SCORE = 1.20

# Exercise & session limits
DEFAULT_CARDS_PER_EXERCISE = 3
MAX_UNKNOWN_WORDS = 2

# Evaluation thresholds
DEFAULT_CORRECTION_THRESHOLD = 75.0
NMT_DELTA_L_THRESHOLD = 8.0
VOCABULARY_UNLOCK_THRESHOLD = 2

# Database & history limits
HISTORY_LIMIT = 50

# Default unlocked foundational vocabulary (mots débloqués par défaut)
DEFAULT_UNLOCKED_WORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'au', 'aux', 'ce', 'cet', 'cette', 'ces',
    'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se', 'moi', 'toi', 'lui',
    'y', 'en', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
    'à', 'de', 'pour', 'dans', 'sur', 'avec', 'sans', 'sous', 'chez', 'vers', 'par', 'entre',
    'et', 'ou', 'mais', 'car', 'si', 'que', 'qui', 'quand', 'où', 'comme', 'donc',
    'être', 'avoir', 'aller', 'faire', 'dire', 'pouvoir', 'vouloir', 'savoir', 'voir', 'devoir', 'venir', 'falloir',
    'ne', 'pas', 'plus', 'très', 'bien', 'oui', 'non', 'merci', 'bonjour', 'bonsoir', 'au revoir', 'salut',
    "m'", "t'", "s'", "c'", "j'", "l'", "d'", "n'", "qu'", "aujourd'hui", "hier", "demain"
}

# Common functional words used for grammatical sound matching
FUNCTION_WORDS = {
    "le", "la", "les", "l'", "un", "une", "des", "du", "de", "d'",
    "ce", "cet", "cette", "ces", "c'", "est", "sont", "a", "ont",
    "que", "qui", "qu'", "ne", "pas", "n'", "se", "s'", "en", "y"
}

# Common short French words for language identification gatekeeper
COMMON_FRENCH_WORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'au', 'aux',
    'ce', 'cet', 'cette', 'ces', 'je', 'tu', 'il', 'elle', 'on',
    'nous', 'vous', 'ils', 'elles', 'j', 't', 's', 'l', 'd', 'c', 'm', 'n', 'qu',
    'me', 'te', 'se', 'lui', 'leur', 'y', 'en', 'moi', 'toi',
    'est', 'ai', 'suis', 'es', 'sommes', 'êtes', 'sont', 'a', 'ont', 'va', 'vais', 'vas', 'vont',
    'bonjour', 'merci', 'oui', 'non', 'salut', 'revoir', 'bien', 'sûr', 'faim', 'soif', 'peur',
    'très', 'plus', 'pas', 'ne', 'où', 'que', 'qui', 'quoi', 'comment', 'pourquoi', 'avec', 'sans', 'dans', 'pour', 'sur'
}
