from backend.db.connection import DBConnection, ROOT_DIR, DB_PATH, SYSTEM_COLUMNS
from backend.db.sentences import SentencesMixin
from backend.db.users import UsersMixin
from backend.db.history import HistoryMixin
from backend.db.audio import AudioMixin
from backend.db.vocabulary import VocabularyMixin


class DataManager(DBConnection, SentencesMixin, UsersMixin, HistoryMixin, AudioMixin, VocabularyMixin):
    pass
