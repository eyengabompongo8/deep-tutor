# Testing Rules for FrenchTutor

## Cardinal Rule
**Never modify an existing test to make a broken implementation pass. Fix the source code instead.**
If a test is genuinely flawed (wrong expectation, broken mock, invalid test contract), document the rationale clearly in a comment and explain it to the user.

## Test Framework & Style
- All tests must use **pure pytest style** (standard `assert`, pytest fixtures).
- **Do not inherit from `unittest.TestCase`**. Using `unittest.TestCase` breaks pytest marker filtering (such as `@pytest.mark.ml`) on test methods.
- Do not add manual `sys.path.insert` manipulations. `pyproject.toml` configures `pythonpath = [".", "backend"]`.
- Imports should be clean and absolute, e.g., `from backend.scoring_engine import ScoringEngine`.

## Fixtures & Database Isolation
Use the shared fixtures from `tests/conftest.py`:
- `temp_db_path`: Yields an isolated temporary SQLite database file path, automatically cleaned up.
- `isolated_data_manager`: Provides a fresh `DataManager` instance seeded with canonical sentences (`SEED_SENTENCES`).
- `mock_tts`: Provides a `MagicMock` TTS engine to prevent network calls to external TTS services.
- `test_client`: Provides a FastAPI `TestClient` pre-wired with `isolated_data_manager`, `scoring_engine`, and `mock_tts`.

**Never** mutate module globals directly without restoring them or using pytest's `monkeypatch` fixture.

## Markers & Test Suites
- `@pytest.mark.ml`: Heavy tests loading PyTorch / Hugging Face models (excluded by default via `addopts = "-m 'not ml'"` in `pyproject.toml`).
- `@pytest.mark.slow`: Long-running integration benchmarks.
- Fast tests must complete in < 2 seconds without external network dependencies.

## Network & I/O Isolation
- Unit and API tests must run completely offline without hitting Hugging Face, LanguageTool remote endpoints, or external TTS servers.
- Always mock network/disk-heavy subsystems when testing upper layers (API endpoints, scoring wrappers).

## Commands Reference
```bash
make test               # Run fast unit and API tests
uv run pytest           # Run fast tests
uv run pytest -m ml     # Run ML model tests
uv run pytest --cov     # Run with coverage report
```
