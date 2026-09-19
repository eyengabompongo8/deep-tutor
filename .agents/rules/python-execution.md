# Python Execution Rules

## The Golden Rule: Exclusively Use `uv run` or `make`

**Never call Python directly.** Every Python invocation in this project must go through one of these two forms:

```bash
uv run python <args>        # Run a script or module
uv run pytest <args>        # Run tests
uv run <tool> <args>        # Run any tool installed in the project env
make <target>               # Use a Makefile target when available
```

### What you must NEVER do

| ❌ Forbidden | Why |
| :--- | :--- |
| `python script.py` | Calls the system Python, not the project environment |
| `.venv/bin/python script.py` | Bypasses uv's dependency sync and env management |
| `.venv/bin/pytest` | Same — never call `.venv/bin/` binaries directly |
| `source .venv/bin/activate && python ...` | Shell activation is fragile and unreliable in agentic contexts |
| `export PATH=".venv/bin:$PATH"` | PATH manipulation causes environment pollution and unexpected behavior |
| `export VIRTUAL_ENV=...` | Same — uv handles the venv internally; do not pre-set env vars |
| `PYTHONPATH=... python ...` | Do not inject env vars to try to make a bare Python call work |
| `/usr/bin/python3 ...` | Never use absolute paths to system Python |

These patterns are anti-patterns. They **do not grant extra access** and **do not escape the sandbox**. They only cause subtly broken behavior, wrong dependency resolution, and hard-to-debug failures.

---

## Sandbox: Do Not Try to Escape It

The `run_command` tool runs in a secure sandbox by default. This is intentional.

**Do not attempt to work around the sandbox** by:
- Manipulating `PATH`, `VIRTUAL_ENV`, `PYTHONPATH`, or other environment variables
- Using absolute paths to binaries outside the workspace
- Chaining shell tricks (`eval`, `env -i`, etc.) to alter the execution environment
- Using `BypassSandbox: true` without a legitimate reason

> [!IMPORTANT]
> If a command genuinely requires network access or elevated privileges (e.g., downloading a model, installing a system package), **use the `run_command` tool with `BypassSandbox: true` and wait for the user to grant permission**. Do not attempt to work around the restriction yourself.

The correct flow when you need out-of-sandbox execution:
1. Identify the single command that truly needs it.
2. Call `run_command` with `BypassSandbox: true`.
3. The user will be prompted for approval — **wait for their decision**.
4. All other preparation and processing steps should remain sandboxed.

---

## Quick Reference

```bash
# ✅ Correct — always use uv run
uv run python -m backend.server
uv run python run_tests.py
uv run pytest tests/
uv run python -m spacy download fr_core_news_sm
uv add <package>
uv sync --extra cpu

# ✅ Correct — use make targets
make test
make run
```
