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

## Phase Awareness

- Before starting work, **read ROADMAP.md** to identify the current phase and its status markers
- Only work on tasks within the **current phase** unless explicitly told otherwise
- Mark completed tasks with `[x]` in ROADMAP.md as you finish them

---

## Documentation Sync (Mandatory)

**Every implementation change MUST be reflected in the relevant markdown files** when applicable. This includes but is not limited to:

- **ROADMAP.md** — Update task checkboxes, add new tasks if scope expanded, mark phases complete
- **README.md** — Update setup instructions, API routes, project structure, tech stack if anything changes
- **.env.example** — Add new environment variables as they are introduced
- **AGENTS.md** — Update rules or checklists if architecture or workflow patterns change

> If you add a new endpoint, service, adapter, migration, or configuration — ask yourself: "Does any markdown file need to reflect this?" If yes, update it in the same commit.

---

## Mandatory Steps Per Task

1. **Read context** — review related existing files before writing new code
2. **Write tests FIRST** — TDD required. Write failing test → implement → pass
3. **Run full test suite** — `pytest --cov --cov-branch` must pass before committing
4. **Check coverage** — coverage must not drop below 95%. Run `pytest --cov-fail-under=95`
5. **Type check** — `mypy app/ --strict` must pass with zero errors
6. **Lint** — `ruff check app/` must pass with zero warnings
7. **Regression test** — if fixing a bug, add test tagged `@pytest.mark.regression`
8. **Manual browser test** — if agent has browser access, visually verify UI changes
9. **Commit message** — use conventional commits: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`

---

## Architecture Rules

### Hexagonal Architecture Enforcement

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

### Hard Rules

- **Domain layer** (`app/domain/`) must have **ZERO** imports from `fastapi`, `sqlalchemy`, `chromadb`, `openai`, `redis`, or any framework
- All new infrastructure adapters **must** implement the corresponding ABC from `app/domain/interfaces/`
- All API endpoints **must** have corresponding integration tests
- All Pydantic schemas **must** have validation tests for edge cases
- All I/O operations **must** use `async/await`
- Every function signature **must** be fully type-hinted

---

## Testing Checklist (Every PR)

- [ ] Unit tests for all new domain logic
- [ ] Integration tests for all new endpoints
- [ ] Negative tests (invalid input, auth failures, not-found cases)
- [ ] Multi-tenant isolation verified (Tenant A cannot access Tenant B data)
- [ ] Coverage ≥ 95% line + branch
- [ ] No type errors (`mypy --strict`)
- [ ] No lint warnings (`ruff check`)

### Test Standards

- **Every module gets a corresponding test file** — `app/domain/services/chat_service.py` → `tests/unit/domain/test_chat_service.py`
- **All edge cases covered** — null inputs, empty collections, auth failures, rate limits, malformed data
- **Negative tests mandatory** — every happy path has a matching error/rejection test
- **Fixtures over mocks** — use factories for test data, mock only external I/O
- **Regression tests** — every bug fix must include a test tagged `@pytest.mark.regression`

---

## Code Quality Gates

```bash
# All of these must pass before committing:
pytest --cov --cov-branch --cov-fail-under=95    # Tests + coverage
mypy app/ --strict                                # Type checking
ruff check app/                                   # Linting
ruff format --check app/                          # Formatting
```

---

## When Browser Is Available

- Navigate to `http://localhost:8000/docs` and verify OpenAPI docs render
- Test chat endpoint via interactive docs
- Verify health endpoint returns 200
- If frontend is running, test integration at `http://localhost:5173`

---

## Environment Setup

```bash
# Required env vars (see .env.example for full list):
OLLAMA_BASE_URL=https://localhost:11434
OLLAMA_CHAT_MODEL=<model-name>
OLLAMA_EMBEDDING_MODEL=<model-name>
CF_OLLAMA_ID=<cloudflare-service-token-id>
CF_OLLAMA_SECRET=<cloudflare-service-token-secret>
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://...
```
