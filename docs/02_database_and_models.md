# Chapter 2: Database Schema, Lexical Models & Repositories

[← Previous: 01 Architecture Overview](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/01_architecture_overview.md) | [Documentation Hub (README)](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/README.md) | [Next: 03 NLP & Scoring Pipeline →](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/03_nlp_and_scoring_pipeline.md)

---

## 1. Foundations of the Two-Tier Lexical Architecture

A core design challenge in intelligent language tutoring systems is bridging the gap between dictionary abstractions (lemmas) and actual textual exposure (surface word inflections). In earlier versions of FrenchTutor, word roots were referred to as generic *lexemas*, and all entries were assigned a static default level (`A1`). The lexical subsystem was subsequently re-architected around **Lexique 4** (`data/lexique/Lexique4/Lexique4.tsv`), establishing a **Two-Tier Lexical Architecture**:

1. **Tier 1: Canonical Base Lemmas (`lemmas`)**  
   Represents canonical dictionary headwords associated with syntactic categories (e.g., the infinitive verb *manger* with POS `VER`, or the singular masculine noun *chat* with POS `NOM`). Each lemma entry encapsulates corpus-level lexical frequency (`12_FreqLemme`) and a psycholinguistically validated CEFR proficiency level (A1 to C2).
2. **Tier 2: Inflected Surface Words (`words`)**  
   Represents surface tokens as they appear in natural language sentences (e.g., *mange*, *mangeons*, *mangeraient*). Every surface word record retains its specific surface frequency (`10_FreqMot`) and points back to its canonical parent lemma via a relational foreign key (`lemma_id`).

```
┌────────────────────────────────────────────────────────────┐
│                    Canonical Lemma                         │
│             [id: 42, lemma: 'manger', pos: 'VER', level: 'A1']    │
└─────────────────────────────┬──────────────────────────────┘
                              │ 1:N Relationship
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ Surface Word │       │ Surface Word │       │ Surface Word │
│ 'mange'      │       │ 'mangeons'   │       │ 'mangeraient'│
│ (Freq: 245.2)│       │ (Freq: 18.4) │       │ (Freq: 4.1)  │
└──────────────┘       └──────────────┘       └──────────────┘
```

### Design Rationale: Why Two Tiers?

* **Psycholinguistic Pedagogical Validity**: Language learners acquire grammatical paradigms through base roots while encountering inflected morphology. Assessing learner proficiency strictly on surface forms would fragment mastery metrics (e.g., treating *parle*, *parles*, and *parlons* as three unrelated vocabulary challenges). Conversely, tracking only lemmas obscures morphological stumbling blocks.
* **Separation of Exercise Generation & Granular Analytics**: Exercise progression relies on lemma-level novelty (e.g., ensuring a candidate sentence introduces at most $i+1$ unknown lemmas). Meanwhile, surface exposure tracking captures whether a student has met specific irregular verb inflections or feminine/plural agreements.
* **Referential Integrity for Inflected-Only Entries**: In natural language datasets such as Lexique 4, approximately 3,696 `(4_Lemme, 5_Cgram)` pairs appear in surface form without an explicit base-row entry where `14_IsLem == '1'` (for instance, the feminine adjective *applicative* appears, while its canonical masculine form *applicatif* lacked a standalone `IsLem == 1` row in the raw corpus table). By modeling lemmas and words as distinct relational entities, FrenchTutor seeds referenced lemmas with their observed corpus frequency, establishing 100% referential integrity across all 69,728 unique lemmas and 189,863 surface words.
* **Surrogate Keys vs. Natural Uniqueness Constraints**: Integer surrogate keys (`id INTEGER PRIMARY KEY AUTOINCREMENT`) are used across both tables for fast, compact foreign-key joining and B-tree indexing in SQLite. Natural uniqueness constraints (`UNIQUE(lemma, pos)` on `lemmas` and `UNIQUE(word, lemma_id, pos)` on `words`) prevent duplicate entries and enforce unambiguous token resolution.

---

## 2. Complete Entity-Relationship (ER) Diagram

The FrenchTutor relational database consists of 10 primary tables and 1 compatibility view, maintaining data across sentences, vocabulary, learner progress, user configurations, and media caches.

```mermaid
erDiagram
    USERS ||--o{ USER_MODES : "configures"
    USERS ||--o| USER_SETTINGS : "customizes"
    USERS ||--o{ EXERCISE_HISTORY : "logs attempts"
    USERS ||--o{ USER_LEMMAS : "masters"
    USERS ||--o{ USER_WORDS : "encounters"

    LEMMAS ||--o{ WORDS : "has inflections (lemma_id)"
    LEMMAS ||--o{ USER_LEMMAS : "tracked per user"
    WORDS ||--o{ USER_WORDS : "tracked per user"

    SENTENCES ||--o{ SENTENCE_LEMMAS : "links base roots"
    LEMMAS ||--o{ SENTENCE_LEMMAS : "referenced by"
    SENTENCES ||--o{ SENTENCE_WORDS : "links surface tokens"
    WORDS ||--o{ SENTENCE_WORDS : "referenced by"

    SENTENCES ||--o{ EXERCISE_HISTORY : "evaluated in"
    SENTENCES ||--o{ AUDIO_CACHE : "synthesized for"

    USERS {
        text id PK "Slug or UUID identifier"
        text name "Display name"
    }

    USER_SETTINGS {
        text user_id PK,FK "References users(id)"
        text theme "UI theme (system/light/dark)"
        text preferred_voice "Preferred TTS voice"
        real correction_threshold "Passing quality threshold"
        int cards_per_exercise "Batch size per exercise"
        int auto_jump "Boolean auto-advance flag"
        int auto_play_audio "Boolean audio playback flag"
        int transcription_retry "Boolean retry on mistake"
        int blind_retry "Boolean hide text on retry"
        text active_mode "translation | transcription | mixed"
        int mixed_translation_ratio "Percentage of translation cards"
        timestamp updated_at "Last update timestamp"
    }

    USER_MODES {
        text user_id PK,FK "References users(id)"
        text mode PK "translation | transcription | mixed"
        text level "Current CEFR level (A1..C2)"
        int streak "Consecutive correct exercises"
        int total_completed "Total completed cards"
        int total_skipped "Total skipped cards"
        timestamp updated_at "Last update timestamp"
    }

    SENTENCES {
        int id PK "Autoincrement surrogate ID"
        text fr "French target text"
        text es "Spanish reference text"
        text level "CEFR level (A1..C2 NOCASE)"
        real score_ml "Composite model difficulty score"
        text source "Corpus source (tatoeba/custom/...)"
        text metadata "JSON string with linguistic features"
    }

    LEMMAS {
        int id PK "Autoincrement surrogate ID"
        text lemma "Base canonical headword"
        text pos "Lexique Cgram POS tag"
        text level "Rank-derived CEFR level (A1..C2)"
        real frequency "Occurrences per million words"
        int is_base "Boolean foundational function word"
        text metadata "JSON string (Zipf score, rank, etc.)"
        timestamp created_at "Timestamp of insertion"
    }

    WORDS {
        int id PK "Autoincrement surrogate ID"
        text word "Surface inflected word"
        int lemma_id FK "References lemmas(id)"
        text pos "Lexique Cgram POS tag"
        real frequency "Occurrences per million words"
        text metadata "JSON string"
        timestamp created_at "Timestamp of insertion"
    }

    SENTENCE_LEMMAS {
        int sentence_id PK,FK "References sentences(id)"
        int lemma_id PK,FK "References lemmas(id)"
    }

    SENTENCE_WORDS {
        int sentence_id PK,FK "References sentences(id)"
        int word_id PK,FK "References words(id)"
    }

    USER_LEMMAS {
        text user_id PK,FK "References users(id)"
        int lemma_id PK,FK "References lemmas(id)"
        text level "Cached CEFR level"
        int appearance_count "Total times lemma was seen"
        int success_count "Times answered correctly"
    }

    USER_WORDS {
        text user_id PK,FK "References users(id)"
        int word_id PK,FK "References words(id)"
        int appearance_count "Total times surface word was seen"
        int success_count "Times answered correctly"
    }

    EXERCISE_HISTORY {
        int id PK "Autoincrement surrogate ID"
        text user_id FK "References users(id)"
        text mode "Exercise mode name"
        int sentence_id FK "References sentences(id)"
        real quality "Evaluation quality score (0.0..1.0)"
        int is_skipped "Boolean skip flag"
        timestamp created_at "Attempt timestamp"
    }

    AUDIO_CACHE {
        int id PK "Autoincrement surrogate ID"
        int sentence_id FK "References sentences(id)"
        text text_hash UK "MD5/SHA256 text fingerprint"
        text text "Exact French utterance text"
        text file_path "Absolute path to cached audio file"
        timestamp created_at "Synthesis timestamp"
    }
```

---

## 3. SQL Table DDL & Indexing Specifications

The schema is created and migrated automatically during database initialization in [`backend/db/connection.py`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/connection.py).

### 3.1 Sentences Table
Stores bilingual (and multilingual) sentence pairs evaluated by the system. Additional language columns (such as `en`) can be added dynamically at runtime via `add_language_column()`.
```sql
CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fr TEXT NOT NULL,
    es TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'A1' COLLATE NOCASE,
    score_ml REAL DEFAULT NULL,
    source TEXT NOT NULL DEFAULT 'custom',
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sentences_level ON sentences(level COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_sentences_score_ml ON sentences(score_ml);
```

### 3.2 Canonical Lemmas Table
Stores unique canonical base lemmas with their psycholinguistic CEFR levels and corpus frequencies.
```sql
CREATE TABLE IF NOT EXISTS lemmas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lemma TEXT NOT NULL,
    pos TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'A1',
    frequency REAL DEFAULT 0.0,
    is_base BOOLEAN NOT NULL DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(lemma, pos)
);

CREATE INDEX IF NOT EXISTS idx_lemmas_pos ON lemmas(pos);
CREATE INDEX IF NOT EXISTS idx_lemmas_level ON lemmas(level);
CREATE INDEX IF NOT EXISTS idx_lemmas_is_base ON lemmas(is_base);
```

### 3.3 Inflected Words Table
Stores all surface word forms, bound to their parent canonical lemma through a cascading foreign key.
```sql
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    lemma_id INTEGER NOT NULL,
    pos TEXT NOT NULL,
    frequency REAL DEFAULT 0.0,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (lemma_id) REFERENCES lemmas(id) ON DELETE CASCADE,
    UNIQUE(word, lemma_id, pos)
);

CREATE INDEX IF NOT EXISTS idx_words_pos ON words(pos);
CREATE INDEX IF NOT EXISTS idx_words_lemma_id ON words(lemma_id);
```

### 3.4 Relational Junction Tables (`sentence_lemmas` & `sentence_words`)
Pre-indexed M:N relational maps connecting sentences with their extracted lemmas and surface tokens.
```sql
CREATE TABLE IF NOT EXISTS sentence_lemmas (
    sentence_id INTEGER NOT NULL,
    lemma_id INTEGER NOT NULL,
    PRIMARY KEY (sentence_id, lemma_id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE,
    FOREIGN KEY (lemma_id) REFERENCES lemmas(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sentence_lemmas_lemma ON sentence_lemmas(lemma_id);

CREATE TABLE IF NOT EXISTS sentence_words (
    sentence_id INTEGER NOT NULL,
    word_id INTEGER NOT NULL,
    PRIMARY KEY (sentence_id, word_id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE,
    FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sentence_words_word ON sentence_words(word_id);
```

### 3.5 Dual User Vocabulary Tracking Tables
Tracks user interaction statistics independently at the lemma level (canonical mastery) and word level (surface form exposures).
```sql
CREATE TABLE IF NOT EXISTS user_lemmas (
    user_id TEXT NOT NULL,
    lemma_id INTEGER NOT NULL,
    level TEXT DEFAULT 'A1',
    appearance_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, lemma_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (lemma_id) REFERENCES lemmas(id) ON DELETE CASCADE
);

-- Composite covering index optimized for candidate sentence CTE lookups
CREATE INDEX IF NOT EXISTS idx_user_lemmas_user_success 
ON user_lemmas(user_id, success_count, lemma_id);

CREATE TABLE IF NOT EXISTS user_words (
    user_id TEXT NOT NULL,
    word_id INTEGER NOT NULL,
    appearance_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, word_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE
);
```

### 3.6 Users, Modes, Settings & Telemetry Tables
```sql
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'Default User'
);

CREATE TABLE IF NOT EXISTS user_modes (
    user_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'A1',
    streak INTEGER NOT NULL DEFAULT 0,
    total_completed INTEGER NOT NULL DEFAULT 0,
    total_skipped INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, mode),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id TEXT PRIMARY KEY,
    theme TEXT DEFAULT 'system',
    preferred_voice TEXT DEFAULT 'default',
    correction_threshold REAL DEFAULT 0.75,
    cards_per_exercise INTEGER DEFAULT 7,
    auto_jump INTEGER DEFAULT 1,
    auto_play_audio INTEGER DEFAULT 1,
    transcription_retry INTEGER DEFAULT 1,
    blind_retry INTEGER DEFAULT 0,
    active_mode TEXT DEFAULT 'translation',
    mixed_translation_ratio INTEGER DEFAULT 50,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS exercise_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'translation',
    sentence_id INTEGER,
    quality REAL,
    is_skipped BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_history_user_mode ON exercise_history(user_id, mode);

CREATE TABLE IF NOT EXISTS audio_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence_id INTEGER,
    text_hash TEXT UNIQUE NOT NULL,
    text TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_audio_cache_hash ON audio_cache(text_hash);
CREATE INDEX IF NOT EXISTS idx_audio_cache_sentence ON audio_cache(sentence_id);
```

### 3.7 Compatibility Views
To maintain backwards compatibility for legacy components referencing the prior lexema naming conventions:
```sql
CREATE VIEW IF NOT EXISTS user_lexema_counts AS
SELECT user_id, lemma_id AS lexema_id, level, appearance_count, success_count
FROM user_lemmas;
```

---

## 4. Dual User Tracking System

The dual tracking mechanism is implemented in [`record_token_appearances(...)`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/vocabulary.py#L245-L320). When a learner completes or skips an exercise card, the system executes simultaneous updates across both layers:

```mermaid
sequenceDiagram
    autonumber
    actor Learner
    participant API as /api/exercises/evaluate
    participant Vocab as VocabularyMixin
    participant DB as SQLite (WAL)

    Learner->>API: Submit translation / transcription response
    API->>Vocab: record_token_appearances(user_id, sentence_id, is_success)
    Vocab->>DB: Query lemma_ids FROM sentence_lemmas WHERE sentence_id = ?
    Vocab->>DB: Query word_ids FROM sentence_words WHERE sentence_id = ?
    
    alt is_retry == False
        Vocab->>DB: UPSERT user_lemmas (appearance_count + 1, success_count + success_inc)
    end
    Vocab->>DB: UPSERT user_words (appearance_count + 1, success_count + success_inc)
    
    Vocab->>DB: Fetch updated counts & evaluate is_known status
    Vocab-->>API: Return token feedback & progression diagnostics
    API-->>Learner: Response with score, corrections & vocabulary delta
```

### 4.1 Canonical Root Mastery (`user_lemmas`)
`user_lemmas` tracks the conceptual mastery of lexical roots.
* **Mastery Definition**: A lemma is considered *known* (`is_known = 1`) by a user if:
  1. It is marked as a foundational base word (`is_base = 1`), such as pronouns, common prepositions, or essential auxiliary verbs.
  2. Its CEFR level is strictly below the user's current target level (e.g., all A1 lemmas are automatically treated as known for an A2 learner).
  3. The user has practiced the lemma and achieved `success_count >= VOCABULARY_UNLOCK_THRESHOLD` (default is 1).
* **Algorithmic Role**: Candidate sentence generation (`get_progressive_sentences`) operates **strictly** over `user_lemmas` and `sentence_lemmas`. This ensures sentence difficulty is governed purely by root concepts rather than inflected variations.

### 4.2 Surface Exposure Tracking (`user_words`)
`user_words` records empirical exposure to specific inflected forms.
* **Pedagogical Purpose**: While a user may understand the lemma *devoir*, encountering rare subjunctive forms (e.g., *dussiez*) represents distinct morphosyntactic friction.
* **Data Granularity**: Tracks specific tense, agreement, and phonetic contractions seen in practice.
* **Engine Isolation**: Changes or extensions to `user_words` tracking do not affect candidate sentence selection, preserving predictable progression while supplying high-resolution analytics for dashboards and spaced-repetition diagnostics.

---

## 5. Database Connection Management & SQLite Optimization

All database interactions pass through [`DBConnection`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/connection.py#L23-L73).

### 5.1 Write-Ahead Logging (WAL) Mode
During initialization, [`DBConnection._set_wal_mode()`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/connection.py#L86-L94) configures SQLite pragma flags:
* `PRAGMA journal_mode = WAL;`: Enables SQLite Write-Ahead Logging. Readers never block writers, and writers do not block readers. This is critical for FrenchTutor, where background candidate precomputation threads continuously read the database while active user exercises write telemetry and history records.
* `PRAGMA synchronous = NORMAL;`: Reduces disk I/O fsync overhead while maintaining crash resilience under WAL mode.
* `PRAGMA busy_timeout = 10000;`: Waits up to 10 seconds to resolve transient locks before throwing a busy operational error.

### 5.2 Context Manager & Transaction Lifecycle
Connections are obtained via the [`_get_connection()`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/connection.py#L43-L73) context manager:

```python
@contextmanager
def _get_connection(self):
    conn = None
    try:
        try:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
        except sqlite3.OperationalError as e:
            if os.path.exists(self.db_path):
                uri_path = f"file:{os.path.abspath(self.db_path)}?immutable=1"
                conn = sqlite3.connect(uri_path, uri=True, timeout=15.0)
            else:
                raise e
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    except Exception:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
```

Key operational traits:
* **Atomic Unit of Work**: Every block automatically commits on normal completion or executes a full `rollback()` if an unhandled exception occurs.
* **Strict Foreign Key Enforcement**: SQLite disables foreign key validation by default. FrenchTutor explicitly executes `PRAGMA foreign_keys = ON;` on every single opened connection to guarantee cascading deletes across junction tables.
* **Row Factory Mapping**: Sets `conn.row_factory = sqlite3.Row` for fast, dict-like key-value column indexing.
* **Immutable Read-Only Fallback**: If file permission errors prevent standard opening on an existing database, the connection falls back to `?immutable=1` URI mode.

---

## 6. Repository Architecture & The Aggregation Pattern

FrenchTutor employs a modular **Repository Aggregation Pattern** using Python multiple-inheritance mixins. Rather than placing all SQL operations inside a monolithic class, queries and mutations are isolated into focused domain modules.

```mermaid
classDiagram
    class DBConnection {
        +db_path: str
        -_lemmas_cache: Dict
        -_lemmas_cache_lock: Lock
        +_get_connection() ContextManager
        -_init_db()
        -_set_wal_mode()
        -_create_tables(conn)
        -_run_migrations(conn)
        -_seed_defaults(conn)
        -_row_to_dict(row) Dict
    }

    class SentencesMixin {
        +get_all_sentences(level) List
        +get_sentence_by_id(item_id) Dict
        +add_sentence(es, fr, ...) Dict
        +bulk_insert_sentences(sentences) int
        +update_sentence(...) bool
        +delete_sentence(item_id) bool
        +add_language_column(lang_code)
    }

    class VocabularyMixin {
        +get_lemmas_dict() Dict
        +invalidate_lemmas_cache()
        +upsert_lemma(...) int
        +bulk_upsert_lemmas(...) Dict
        +upsert_word(...) int
        +bulk_upsert_words(...) Dict
        +link_sentence_lemmas(sentence_id, lemma_ids)
        +link_sentence_words(sentence_id, word_ids)
        +get_sentence_lemmas(sentence_id) List
        +get_sentence_words(sentence_id) List
        +get_user_known_lemma_ids(...) Set
        +record_token_appearances(...) List
        +get_user_vocabulary_stats(...) Dict
        +get_progressive_sentences(...) List
    }

    class UsersMixin {
        +get_or_create_user(user_id, name) Dict
        +get_user(user_id) Dict
        +list_users() List
        +create_user(name, user_id) Dict
        +delete_user(user_id) bool
        +get_user_mode(user_id, mode) Dict
        +update_user_mode(...) Dict
        +reset_user_mode(...) Dict
        +get_user_modes_profile(user_id) Dict
        +get_user_settings(user_id) Dict
        +update_user_settings(user_id, **kwargs) Dict
    }

    class HistoryMixin {
        +record_exercise_history(...)
        +get_exercise_history(...) List
    }

    class AudioMixin {
        +get_audio_cache(text_hash) Dict
        +save_audio_cache(...)
    }

    class DataManager {
    }

    DBConnection <|-- DataManager
    SentencesMixin <|-- DataManager
    UsersMixin <|-- DataManager
    HistoryMixin <|-- DataManager
    AudioMixin <|-- DataManager
    VocabularyMixin <|-- DataManager
```

### 6.1 The Unified `DataManager` Interface
As defined in [`backend/db/manager.py`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/manager.py#L9-L10):
```python
class DataManager(DBConnection, SentencesMixin, UsersMixin, HistoryMixin, AudioMixin, VocabularyMixin):
    pass
```
The FastAPI dependency injection container instantiates a single `DataManager` instance, exposing the full breadth of domain operations across all endpoints without needing multiple disparate database clients:
* [`SentencesMixin`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/sentences.py): Sentence pair persistence, batch ingestion, dynamic multilingual column generation, and level distribution aggregations.
* [`VocabularyMixin`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/vocabulary.py): Lexique lemma/word upserts, sentence linking, dual tracking counters, in-memory lemma cache management, and the $i+1$ progressive sentence selection query.
* [`UsersMixin`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/users.py): User profiles, mode-specific streaks and completions (`translation`, `transcription`, `mixed`), and configuration flags (`user_settings`).
* [`HistoryMixin`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/history.py): Audit logging of exercise attempts, evaluation scores, and skip actions.
* [`AudioMixin`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/audio.py): Text hash indexing and file path caching for Edge-TTS and Piper audio synthesis.

---

## 7. Indexing Strategies for High-Frequency Lookups & Candidate Sampling

Candidate sentence selection operates under tight interactive latency constraints. To guarantee sub-15ms generation times, FrenchTutor employs a multi-tiered query and caching architecture:

### 7.1 Composite Covering Index on `user_lemmas`
The core progression query uses a Common Table Expression (CTE) to compute which lemmas are already known by the learner:
```sql
CREATE INDEX IF NOT EXISTS idx_user_lemmas_user_success 
ON user_lemmas(user_id, success_count, lemma_id);
```
By indexing `(user_id, success_count, lemma_id)` in that order, SQLite performs an index-only scan to evaluate `WHERE user_id = ? AND success_count >= ?`, retrieving all known `lemma_id`s directly from the B-tree leaf pages without touching table data blocks.

### 7.2 The $i+1$ Candidate Aggregation Query
In [`VocabularyMixin.get_progressive_sentences(...)`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/vocabulary.py#L484-L501), sentences matching the user's level are evaluated against known vocabulary:

```sql
WITH known_lemmas AS (
    SELECT id FROM lemmas WHERE is_base = 1 OR level IN ('A1')
    UNION
    SELECT lemma_id FROM user_lemmas WHERE user_id = 'user_123' AND success_count >= 1
)
SELECT 
    s.*, 
    GROUP_CONCAT(sl.lemma_id) as all_lemma_ids,
    SUM(CASE WHEN kl.id IS NULL THEN 1 ELSE 0 END) as unknown_count,
    GROUP_CONCAT(CASE WHEN kl.id IS NULL THEN sl.lemma_id ELSE NULL END) as unknown_lemma_ids
FROM sentences s
JOIN sentence_lemmas sl ON s.id = sl.sentence_id
LEFT JOIN known_lemmas kl ON sl.lemma_id = kl.id
WHERE s.level = 'A2'
GROUP BY s.id
HAVING unknown_count <= 2;
```

**Index Acceleration**:
1. `s.level = 'A2'`: Filtered via `idx_sentences_level` (`sentences(level COLLATE NOCASE)`).
2. `JOIN sentence_lemmas sl ON s.id = sl.sentence_id`: Uses primary key `(sentence_id, lemma_id)`.
3. `LEFT JOIN known_lemmas kl ON sl.lemma_id = kl.id`: Fast hash lookup against CTE results.
4. `HAVING unknown_count <= 2`: Filters out sentences that introduce more than the allowed threshold of unmastered concepts.

### 7.3 In-Memory Candidate Buffering
To prevent repetitive SQL execution during active study sessions, `VocabularyMixin` maintains a per-user candidate buffer partitioned by mode and target level:
```python
buffer_key = f"{user_id}_{target_level}_{mode}"
buffer = self._user_candidates_buffer[buffer_key]  # {'reinforcement': [...], 'progression': [...]}
```
* **Progression Cards**: Sentences containing exactly $1 \le \text{unknown\_count} \le \text{max\_unknown}$.
* **Reinforcement Cards**: Sentences where $\text{unknown\_count} == 0$ (all lemmas already mastered).
* **Buffer Invalidation**: Any upsert or mutation of lemmas immediately flushes the candidate buffer via [`invalidate_lemmas_cache()`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/vocabulary.py#L32-L38).

### 7.4 In-Memory Lemma Metadata Cache
The static properties of all 69,728 lemmas (`lemma`, `pos`, `level`, `is_base`) are held in thread-safe memory via [`get_lemmas_dict()`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/db/vocabulary.py#L13-L28). When cards are popped from the candidate buffer, surface token pairing and known/new classifications are resolved entirely in-memory without issuing secondary SQL queries.

---

## 8. Summary Table of Indexes & Storage Optimization

| Table | Index Name | Indexed Columns | Design Target |
| :--- | :--- | :--- | :--- |
| `sentences` | `idx_sentences_level` | `level COLLATE NOCASE` | Fast level-based candidate filtering |
| `sentences` | `idx_sentences_score_ml` | `score_ml` | Quality/difficulty range queries |
| `lemmas` | `idx_lemmas_pos` | `pos` | POS-specific lookup & disambiguation |
| `lemmas` | `idx_lemmas_level` | `level` | Bulk filtering by CEFR rank level |
| `lemmas` | `idx_lemmas_is_base` | `is_base` | Foundational vocabulary identification |
| `words` | `idx_words_pos` | `pos` | Surface token grammatical classification |
| `words` | `idx_words_lemma_id` | `lemma_id` | Relational join from lemma to inflections |
| `sentence_lemmas` | `idx_sentence_lemmas_lemma` | `lemma_id` | Reverse lookup (finding sentences by lemma) |
| `sentence_words` | `idx_sentence_words_word` | `word_id` | Reverse lookup (finding sentences by word) |
| `user_lemmas` | `idx_user_lemmas_user_success` | `(user_id, success_count, lemma_id)` | Covering index for candidate CTE lookups |
| `exercise_history`| `idx_history_user_mode` | `(user_id, mode)` | User performance audit trail pagination |
| `audio_cache` | `idx_audio_cache_hash` | `text_hash UNIQUE` | $O(1)$ audio file retrieval by MD5 text key |
| `audio_cache` | `idx_audio_cache_sentence` | `sentence_id` | Sentence-to-audio cache invalidation |

---

[← Previous: 01 Architecture Overview](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/01_architecture_overview.md) | [Documentation Hub (README)](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/README.md) | [Next: 03 NLP & Scoring Pipeline →](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/03_nlp_and_scoring_pipeline.md)
