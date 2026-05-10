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
| Phase 4 | `phase-4/response-timeout` | First-token timeout, streaming fallback, retry logic |
| Phase 5 | `phase-5/output-validation` | Anti-hallucination guard, post-generation rule checks |
| Phase 6 | `phase-6/content-moderation` | Input/output moderation, jailbreak detection, blocklist |
| Phase 7 | `phase-7/smart-escalation` | Sentiment + repetition + explicit request escalation |
| Phase 8 | `phase-8/feedback-review-queue` | Admin review endpoints + UI for negative feedback, escalations, flagged messages |
| Phase 9 | `phase-9/platform-superadmin` | Platform superadmin role, JWT claims, cross-tenant access |
| Phase 10 | `phase-10/tenant-provisioning-api` | Tenant CRUD, status lifecycle, chat gate for suspended tenants |
| Phase 11 | `phase-11/failed-query-logging` | Failed query model, persistence, admin API, analytics integration |
| Phase 12 | `phase-12/tenant-provisioning-ui` | Superadmin frontend for tenant management |
| Phase 13 | `phase-13/rate-limiting` | Redis sliding window middleware, fail-closed, token blacklist |
| Phase 14 | `phase-14/pii-masking` | PII detection (CC, SSN, phone, email), masking before LLM + storage |
| Phase 15 | `phase-15/user-approval` | User registration approval workflow (backend) |
| Phase 16 | `phase-16/role-management` | Role management API (promote, demote, last-admin protection) |
| Phase 17 | `phase-17/user-management-ui` | User management + approval frontend |
| Phase 18 | `phase-18/moderation-dashboard-api` | Cross-tenant moderation query API |
| Phase 19 | `phase-19/moderation-dashboard-ui` | Moderation dashboard frontend |
| Phase 20 | `phase-20/ab-testing-config` | Tenant config (model, temperature, prompt variant), admin settings UI |
| Phase 21 | `phase-21/webhook-integration` | Webhook service for escalation/feedback/new conversation events |
| Phase 22 | `phase-22/deployment-e2e` | Docker prod, deployment guides, E2E test suite, tech debt cleanup |

### Branch Rules

1. **Create the branch from `main`** before starting any phase work
2. **All commits within a phase go to its branch** — never commit phase work directly to `main`
3. **When a phase is complete**, open a PR from the phase branch → `main`
4. **Do NOT start the next phase** until the previous phase's PR is merged to `main`
5. **Use conventional commit messages** within each branch: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`

### Scope Rules

> **One sub-phase per conversation.** Large phases (e.g., Phase 2 has 7 sub-phases) MUST be implemented one sub-phase at a time. Commit after each sub-phase. This prevents context loss and ensures each unit of work receives full validation. Do NOT implement multiple sub-phases in a single session.

### Cross-Repo Phases

Some phases (8, 20, 22) span both `supportforge-api` and `supportforge-ui`. When working on a cross-repo phase:
1. The **backend portion** follows this file's pipeline (Steps 1–9)
2. The **frontend portion** follows `supportforge-ui/AGENTS.md`'s pipeline (Steps 1–8)
3. Execute backend tasks first, then frontend tasks (frontend depends on API endpoints)
4. Commit in **each repo separately** with its own conventional commit message
5. Use the **same branch name** in both repos

---

## Task Execution Pipeline

> **This is the exact sequence an agent MUST follow for every task.** No step may be skipped. Each step has a gate condition that must be satisfied before proceeding to the next.
>
> **Failure protocol:** If a gate fails, fix the issue and re-run. **Maximum 3 attempts per gate.** After 3 failures on the same gate, STOP and report the error with full context (command output, file paths, error messages). Do NOT continue to the next step.

### Step 1 — Orient

1. Read `ROADMAP.md` and identify the current phase and the specific task to work on
2. Read all files in the directories that will be affected by the task
3. If this task depends on work from a previous task, verify that work exists and is correct
4. **Cross-cutting audit:** If this task introduces or modifies a cross-cutting concern (auth, caching, tenant isolation, logging, error handling), identify **ALL existing endpoints and modules** that must also be updated. List them explicitly before writing any code.
5. **Impact analysis:** For any module being moved, renamed, or refactored — grep for all import references and list every file that will need updating.

> **Gate:** You can describe what you're about to build, which files you'll create or modify, which existing endpoints need updating for cross-cutting concerns, and how it fits into the existing codebase.

### Step 2 — Write Tests

1. Create the test file(s) for the new code following the naming convention: `app/path/to/module.py` → `tests/unit/path/to/test_module.py`
2. Write tests for:
   - Happy path (expected inputs → expected outputs)
   - Edge cases (empty inputs, boundary values, None/null, empty strings)
   - Error cases (invalid input → correct exception raised)
   - If fixing a bug: add a regression test tagged `@pytest.mark.regression`
   - **If the feature involves streaming (WebSocket/SSE):** write at least one test verifying that the data the client receives (streamed frames) is consistent with the data persisted to the database. Any post-stream mutation (appended text, modified status) must either be streamed to the client OR documented as a known divergence.
3. **Security negative tests (mandatory for any auth or data endpoint):**
   - Missing auth token → 401
   - Wrong tenant's data → 403 or empty result
   - Empty/missing required fields → 422
   - Invalid token claims (empty `user_id`, empty `tenant_id`, expired token) → 401
   - Malformed inputs that could bypass validation (empty strings vs None)
4. Run the new tests to confirm they **fail** (since implementation doesn't exist yet): `pytest tests/path/to/test_module.py -v`
5. **Paste the failure output** to confirm tests fail for the right reason (`ImportError` or `NotImplementedError`, NOT `SyntaxError`, and NOT passing)

> **Gate:** New tests exist, fail correctly, AND include at least one negative/error test per endpoint or public method.

### Step 3 — Implement

1. Write the implementation code
2. Follow these constraints:
   - Domain layer (`app/domain/`) — **ZERO** imports from `fastapi`, `sqlalchemy`, `chromadb`, `openai`, `redis`
   - Infrastructure adapters — **MUST** implement the corresponding ABC from `app/domain/interfaces/`
   - All I/O operations — **MUST** use `async/await`
   - All function signatures — **MUST** have complete type hints
   - All `datetime` usage — **MUST** use `datetime.now(timezone.utc)` — **NEVER** `datetime.utcnow()` or naive datetimes
   - All Pydantic response models — **MUST NOT** include sensitive fields (password hashes, tokens, connection strings)
   - If any ORM model (`app/infrastructure/database/models.py`) is modified — **MUST** generate an Alembic migration: `alembic revision --autogenerate -m "<description>"` and verify with `alembic upgrade head`
   - If a new column has a `default=` value, verify existing rows will be migrated correctly (nullable vs server_default)
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

### Step 5 — Self-Review

> **This step exists because automated validation (Step 4) only catches syntax, types, and test failures — it does NOT catch design flaws, security gaps, or cross-cutting inconsistencies. Those are the issues that code reviewers find.**

1. Stage all changes first: `git add -A`, then run `git diff --cached` (full diff — **NOT** `--stat`) and read through every changed line as if you are an independent reviewer seeing this code for the first time.
2. For **every file**, systematically ask:
   - **Security:** Does this endpoint have auth? Does it validate tenant ownership? Are credentials masked?
   - **Consistency:** Does this code follow the same patterns as other files in the same layer? (Same error handling, same response format, same naming)
   - **Error paths:** What happens when this input is `None`? Empty string `""`? Missing key? Negative number? Do these all produce the correct error response?
   - **Cross-cutting:** Does this change affect any OTHER endpoint or module? Did I update all callers/importers?
   - **Cleanup:** Are there deprecated shims, unused imports, or stale re-exports left from refactoring?
3. Check all endpoints against the **Security Checklist** (below).
4. Check against the **Consistency Checklist** (below).
5. If you find ANY issue, go back to Step 3 and fix before continuing.

> **Gate:** You can explain why every endpoint is secure, every boundary validated, every cross-cutting concern addressed, and no deprecated code remains.

### Step 6 — Update Documentation

For each markdown file, check if this task requires an update:

| File | Update if... |
|---|---|
| `ROADMAP.md` | Task completed → mark `[x]`. New tasks discovered → add them. |
| `README.md` | New endpoint, dependency, setup step, or project structure change. |
| `.env.example` | New environment variable introduced. |
| `AGENTS.md` | New architecture rule, workflow change, or convention established. |

> **Gate:** All affected markdown files are updated. No stale information remains.

### Step 7 — Commit

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

### Step 8 — Verify (if browser available)

1. Start the server: `uvicorn app.main:app --reload`
2. Navigate to `http://localhost:8000/docs` — verify OpenAPI docs render
3. Test the endpoint(s) you just created via the interactive docs
4. Verify `GET /health` returns 200
5. If frontend is running at `http://localhost:5173`, verify integration

> **Gate:** Manual verification confirms the feature works as expected.

### Step 9 — Update Master Plan

1. Open `../supportforge_plan.md` (the master implementation plan in the parent directory)
2. Find the checklist items that correspond to the task(s) you just completed
3. Mark them as done: `- [ ]` → `- [x]`
4. Do **not** modify any other content in the plan

> **Gate:** Every task you completed in this session is marked `[x]` in `supportforge_plan.md`.

---

## Security Checklist

> **Check EVERY item before committing ANY endpoint or auth-related code.** This checklist exists because security gaps were the #1 category of code review findings.

### Authentication & Authorization
- [ ] Every endpoint has explicit auth (`Depends(get_current_user)` or `Depends(require_role(...))`)
- [ ] The ONLY exceptions to auth are: `POST /auth/register`, `POST /auth/login`, `POST /auth/refresh`, `GET /health`
- [ ] Token verification rejects missing AND empty required claims (`user_id`, `tenant_id`, `role`)
- [ ] Refresh token rotation: old refresh token is invalidated after use

### Multi-Tenant Isolation
- [ ] Every read query filters by `tenant_id` from the authenticated user's token
- [ ] Every write operation validates the target resource belongs to the authenticated user's tenant
- [ ] Cross-tenant access tests exist (Tenant A cannot see/modify Tenant B's data)

### Sensitive Data
- [ ] No plaintext passwords in DTOs, logs, error messages, or API responses
- [ ] Password hash fields are excluded from all response schemas
- [ ] All log messages mask sensitive data (passwords, tokens, connection strings, API keys)
- [ ] Error messages don't leak internal state (no stack traces, no SQL, no tenant IDs of other tenants)

### Input Validation
- [ ] All datetime values use `datetime.now(timezone.utc)` — never `datetime.utcnow()` or naive datetimes
- [ ] Default/fallback values are fail-secure (no empty strings that bypass validation)
- [ ] Enum values are validated (not just accepted as raw strings)
- [ ] Pagination has sensible max limits to prevent resource exhaustion

### Configuration
- [ ] No default secrets in production (`JWT_SECRET`, `DATABASE_URL`, etc. must be explicitly set)
- [ ] Startup fails fast if required secrets are missing or empty

---

## Consistency Checklist

> **Check EVERY item to ensure the codebase is internally consistent.** Inconsistency was the #2 category of code review findings.

### Patterns
- [ ] All routers use the same error response format
- [ ] All services follow the same dependency injection pattern
- [ ] Stateless domain services are injected via constructor (`__init__`) or used as module-level singletons — never instantiated per-request inside a method
- [ ] All repository methods follow the same naming convention (`get_by_*`, `create`, `update`, `delete`)
- [ ] All Pydantic schemas follow the same field naming convention (snake_case, consistent `Field(...)` usage)

### Imports & Structure
- [ ] No deprecated import shims left after refactoring — all callers use the canonical path
- [ ] No re-export files that exist only for backward compatibility — migrate all consumers
- [ ] Domain-to-domain imports are at module top-level (never inline). Inline imports are ONLY justified when crossing hexagonal boundaries (domain→infrastructure) to avoid circular dependencies.
- [ ] All `__init__.py` files have module docstrings

### Type Safety
- [ ] All function signatures have full type annotations
- [ ] All `Any` usage is justified with a comment explaining why a specific type cannot be used
- [ ] ABCs have `close()` / cleanup methods for resources that need lifecycle management

---

## Phase Completion Checklist

> **Before marking ANY phase as complete, verify EVERY item below.** This is the final gate before a phase branch can be submitted as a PR.

- [ ] Every endpoint in this phase has auth protection (checked against Security Checklist)
- [ ] Every data endpoint has tenant isolation (verified with cross-tenant test)
- [ ] All deprecated shims from refactoring are cleaned up — no backward-compat re-exports remain
- [ ] All new config values are in `.env.example` with documentation
- [ ] All new dependencies are in `pyproject.toml`
- [ ] `git diff main --name-only` shows no unexpected files
- [ ] Full self-review of `git diff main` completed (not just `--stat`)
- [ ] All items in `ROADMAP.md` for this phase are marked `[x]`
- [ ] All items in `../supportforge_plan.md` for this phase are marked `[x]`
- [ ] Coverage is ≥ 95% with `--cov-branch`
- [ ] `mypy --strict` passes with zero errors
- [ ] `ruff check` + `ruff format --check` pass with zero issues
- [ ] If any ORM model changed: Alembic migration exists and `alembic upgrade head` succeeds on a clean DB
- [ ] No pending `alembic revision --autogenerate` changes (run and verify "No changes detected" or new migration is committed)

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

### Test File Naming Convention

For **any** module not listed below, derive the test path using this rule:
- `app/{layer}/{path}/module.py` → `tests/unit/{path}/test_module.py` (domain services, utilities, workers)
- `app/api/v1/endpoint.py` → `tests/integration/api/test_endpoint.py` (API routes)
- `app/infrastructure/{adapter}/adapter.py` → `tests/unit/infrastructure/test_adapter.py` (infra adapters)

**Every new module MUST have a corresponding test file.** No exceptions.

### Test File Mapping (existing modules)

| Source | Test |
|---|---|
| `app/domain/services/chat_service.py` | `tests/unit/test_chat.py` |
| `app/domain/services/tenant_service.py` | `tests/unit/domain/test_tenant_service.py` |
| `app/domain/services/ingestion_service.py` | `tests/unit/domain/test_ingestion_service.py` |
| `app/infrastructure/llm/ollama_adapter.py` | `tests/integration/infrastructure/test_ollama_adapter.py` |
| `app/infrastructure/cache/redis_adapter.py` | `tests/unit/infrastructure/test_redis_adapter.py` |
| `app/api/v1/auth.py` | `tests/integration/api/test_auth.py` |
| `app/api/v1/tenants.py` | `tests/integration/api/test_tenants.py` |
| `app/api/v1/conversations.py` | `tests/integration/api/test_conversations.py` |
| `app/api/v1/chat_router.py` | `tests/integration/api/test_chat.py` |
| `app/api/v1/ingest.py` | `tests/integration/api/test_ingest.py` |
| `app/api/schemas/*.py` | `tests/unit/schemas/test_*.py` |
| `app/core/security.py` | `tests/unit/test_security.py` |
| `app/core/events.py` | `tests/unit/test_events.py` |
| `app/core/dependencies.py` | `tests/unit/test_dependencies.py` |
| `app/config.py` | `tests/unit/test_config.py` |
| `app/rag/nodes/retriever.py` | `tests/unit/rag/test_retriever.py` |
| `app/rag/text_extractor.py` | `tests/unit/workers/test_text_extractor.py` |
| `app/workers/ingestion_worker.py` | `tests/unit/workers/test_ingestion_worker.py` |
| `app/domain/services/output_validator.py` | `tests/unit/domain/test_output_validator.py` |
| `app/domain/services/content_moderator.py` | `tests/unit/domain/test_content_moderator.py` |
| `app/domain/services/escalation_detector.py` | `tests/unit/domain/test_escalation_detector.py` |
| `app/api/v1/review.py` | `tests/integration/api/test_review.py` |
| `app/domain/models/enums.py` (SUPERADMIN) | `tests/unit/test_superadmin.py` |
| `app/domain/models/user.py` (is_superadmin) | `tests/unit/test_superadmin.py` |
| `app/core/security.py` (is_superadmin JWT) | `tests/unit/test_superadmin.py` |
| `app/core/dependencies.py` (require_superadmin) | `tests/unit/test_superadmin.py` |
| `app/domain/models/failed_query.py` | `tests/unit/domain/test_failed_query.py` |
| `app/domain/models/enums.py` (FailureReason) | `tests/unit/domain/test_failed_query.py` |
| `app/api/schemas/failed_query.py` | `tests/unit/schemas/test_failed_query_schemas.py` |
| `app/api/v1/failed_queries.py` | `tests/integration/api/test_failed_queries.py` |

### Testing Standards

- **Fixtures over mocks** — use factories for test data, mock only external I/O
- **Every module gets a test file** — no exceptions
- **Negative tests are mandatory** — every happy path has a matching error/rejection test
- **Security tests are mandatory** — every auth endpoint tests missing/invalid/expired tokens
- **Multi-tenant isolation** — Tenant A must never see Tenant B's data
- **Test environment isolation** — tests must not leak environment variables (`APP_ENV`, secrets) between test functions; use `monkeypatch` or `unittest.mock.patch.dict(os.environ)`
- **Edge case coverage** — test empty strings, `None`, zero-length lists, boundary values for every validated field
- **Pattern/regex coverage** — when implementing regex-based detection, test with: trailing punctuation (`.`, `,`, `)`), embedded in sentences, case variations, overlapping/substring matches, and boundary characters

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
