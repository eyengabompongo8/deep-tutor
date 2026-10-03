import os

# App launch settings
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", os.environ.get("BACKEND_PORT", 8000)))
RELOAD = os.environ.get("RELOAD", "false").lower() in ("1", "true", "yes")

# Default user settings
DEFAULT_CEFR_LEVEL = "A1"
STARTING_SCORE = 1.20
DEFAULT_AUTO_JUMP = True

# Exercise & session limits
DEFAULT_CARDS_PER_EXERCISE = 3
MAX_UNKNOWN_WORDS = 2
DEFAULT_MIXED_TRANSLATION_RATIO = 50

# Evaluation thresholds
DEFAULT_CORRECTION_THRESHOLD = 75.0
NMT_DELTA_L_THRESHOLD = 8.0
VOCABULARY_UNLOCK_THRESHOLD = 1

# Database & history limits
HISTORY_LIMIT = 50

# Ingestion sentence limits
INGESTION_MIN_WORDS = 3
INGESTION_MAX_WORDS = 40

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
