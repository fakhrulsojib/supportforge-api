# Agent Instructions — SupportForge API

## Project Overview

SupportForge is a production-grade, multi-tenant AI customer support agent. This is the **backend API** repository built with FastAPI, using hexagonal architecture (ports & adapters) and SOLID principles.

**Full implementation plan:** See [ROADMAP.md](ROADMAP.md) for the phased roadmap and task breakdown.

---

## Branch Strategy

**Every implementation phase MUST be developed in its own dedicated git branch.** The completed phase branch is then submitted as a Pull Request to `main`.

| Phase | Branch Name | Description |
|---|---|---|
| Phase 0 | `phase-0/repository-bootstrap` | Repo setup, docs, env config |
| Phase 1 | `phase-1/core-rag-engine` | FastAPI scaffold, DB, Ollama, ChromaDB, RAG pipeline, chat endpoint, Docker |
| Phase 2 | `phase-2/realtime-admin` | WebSocket streaming, doc upload, auth, RBAC, Redis cache |
| Phase 3 | _(frontend repo)_ | React + Vite frontend — see `supportforge-ui` |
| Phase 4 | `phase-4/production-polish` | A/B testing, rate limiting, widget, webhooks, email digest, E2E tests |

### Branch Rules

1. **Create the branch from `main`** before starting any phase work
2. **All commits within a phase go to its branch** — never commit phase work directly to `main`
3. **When a phase is complete**, open a PR from the phase branch → `main`
4. **Do NOT start the next phase** until the previous phase's PR is merged to `main`
5. **Use conventional commit messages** within each branch: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`

---

## Task Execution Pipeline

> **This is the exact sequence an agent MUST follow for every task.** No step may be skipped. Each step has a gate condition that must be satisfied before proceeding to the next.

### Step 1 — Orient

1. Read `ROADMAP.md` and identify the current phase and the specific task to work on
2. Read all files in the directories that will be affected by the task
3. If this task depends on work from a previous task, verify that work exists and is correct

> **Gate:** You can describe what you're about to build, which files you'll create or modify, and how it fits into the existing codebase.

### Step 2 — Write Tests

1. Create the test file(s) for the new code following the naming convention: `app/path/to/module.py` → `tests/unit/path/to/test_module.py`
2. Write tests for:
   - Happy path (expected inputs → expected outputs)
   - Edge cases (empty inputs, boundary values, None/null)
   - Error cases (invalid input → correct exception raised)
   - If fixing a bug: add a regression test tagged `@pytest.mark.regression`
3. Run the new tests to confirm they **fail** (since implementation doesn't exist yet): `pytest tests/path/to/test_module.py -v`

> **Gate:** New tests exist and fail with `ImportError` or `NotImplementedError`, NOT with syntax errors.

### Step 3 — Implement

1. Write the implementation code
2. Follow these constraints:
   - Domain layer (`app/domain/`) — **ZERO** imports from `fastapi`, `sqlalchemy`, `chromadb`, `openai`, `redis`
   - Infrastructure adapters — **MUST** implement the corresponding ABC from `app/domain/interfaces/`
   - All I/O operations — **MUST** use `async/await`
   - All function signatures — **MUST** have complete type hints
3. Run the new tests again to confirm they **pass**: `pytest tests/path/to/test_module.py -v`

> **Gate:** All new tests pass.

### Step 4 — Validate

Run each of these commands in order. **If any command fails, fix the issue before proceeding.**

```bash
# 4a. Full test suite (all existing + new tests must pass)
pytest --cov --cov-branch --cov-fail-under=95

# 4b. Type checking (zero errors)
mypy app/ --strict

# 4c. Linting (zero warnings)
ruff check app/

# 4d. Formatting (zero diffs)
ruff format --check app/
```

> **Gate:** All four commands exit with code 0.

### Step 5 — Update Documentation

For each markdown file, check if this task requires an update:

| File | Update if... |
|---|---|
| `ROADMAP.md` | Task completed → mark `[x]`. New tasks discovered → add them. |
| `README.md` | New endpoint, dependency, setup step, or project structure change. |
| `.env.example` | New environment variable introduced. |
| `AGENTS.md` | New architecture rule, workflow change, or convention established. |

> **Gate:** All affected markdown files are updated. No stale information remains.

### Step 6 — Commit

1. Stage all changes: `git add -A`
2. Review staged changes: `git diff --cached --stat`
3. Commit with a conventional commit message:
   - `feat: <description>` — new feature or functionality
   - `fix: <description>` — bug fix
   - `test: <description>` — test-only changes
   - `refactor: <description>` — code restructuring, no behavior change
   - `docs: <description>` — documentation-only changes
   - `chore: <description>` — tooling, config, dependencies

> **Gate:** `git status` shows a clean working tree.

### Step 7 — Verify (if browser available)

1. Start the server: `uvicorn app.main:app --reload`
2. Navigate to `http://localhost:8000/docs` — verify OpenAPI docs render
3. Test the endpoint(s) you just created via the interactive docs
4. Verify `GET /health` returns 200
5. If frontend is running at `http://localhost:5173`, verify integration

> **Gate:** Manual verification confirms the feature works as expected.

### Step 8 — Update Master Plan

1. Open `../supportforge_plan.md` (the master implementation plan in the parent directory)
2. Find the checklist items that correspond to the task(s) you just completed
3. Mark them as done: `- [ ]` → `- [x]`
4. Do **not** modify any other content in the plan

> **Gate:** Every task you completed in this session is marked `[x]` in `supportforge_plan.md`.

---

## Architecture Reference

### Hexagonal Architecture

```
Domain Core (ZERO framework imports)
├── domain/models/       → Pure Python dataclasses / Pydantic models
├── domain/services/     → Business logic orchestration
└── domain/interfaces/   → Abstract Base Classes (ports)

Infrastructure (adapters — implement the ports)
├── infrastructure/database/       → SQLAlchemy ORM + repositories
├── infrastructure/llm/            → Ollama adapter
├── infrastructure/vectorstore/    → ChromaDB adapter
├── infrastructure/cache/          → Redis adapter
└── infrastructure/websocket/      → Connection manager

API Layer (FastAPI routes + Pydantic schemas)
├── api/v1/       → Route handlers
└── api/schemas/  → Request/response DTOs
```

### Test File Mapping

| Source | Test |
|---|---|
| `app/domain/services/chat_service.py` | `tests/unit/domain/test_chat_service.py` |
| `app/infrastructure/llm/ollama_adapter.py` | `tests/integration/infrastructure/test_ollama_adapter.py` |
| `app/api/v1/chat.py` | `tests/integration/api/test_chat.py` |
| `app/rag/nodes/retriever.py` | `tests/unit/rag/test_retriever.py` |

### Testing Standards

- **Fixtures over mocks** — use factories for test data, mock only external I/O
- **Every module gets a test file** — no exceptions
- **Negative tests are mandatory** — every happy path has a matching error/rejection test
- **Multi-tenant isolation** — Tenant A must never see Tenant B's data

---

## Environment Setup

```bash
# Required env vars (see .env.example for full list):
OLLAMA_BASE_URL=https://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CF_OLLAMA_ID=<cloudflare-service-token-id>
CF_OLLAMA_SECRET=<cloudflare-service-token-secret>
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://...
```
