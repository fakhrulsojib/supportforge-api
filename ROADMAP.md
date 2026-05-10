# SupportForge API — Roadmap

> Implementation phases for the backend API. Each phase is implemented in a dedicated git branch and merged to `main` via PR.

## Phase 0 — Repository Bootstrap ✅

- [x] Initialize git repo, `.gitignore`, `LICENSE` (MIT)
- [x] Create `README.md` with project overview and setup instructions
- [x] Create `ROADMAP.md` (this file)
- [x] Create `AGENTS.md` with AI agent instructions
- [x] Create `.env.example` with all required env vars
- [x] Verify Ollama access and lock model names

---

## Phase 1 — Core RAG Engine ✅

> **Branch:** `phase-1/core-rag-engine` → merged to `main`

### 1.1 — FastAPI Project Scaffold ✅
- [x] `pyproject.toml` with all dependencies
- [x] `app/main.py` with app factory pattern (`create_app()`)
- [x] `app/config.py` with Pydantic `Settings` class
- [x] `app/core/exceptions.py` — custom exception hierarchy
- [x] `app/core/middleware.py` — CORS, request-ID, tenant context
- [x] `app/core/events.py` — startup/shutdown lifecycle
- [x] `app/core/dependencies.py` — FastAPI Depends injection
- [x] Health endpoint at `/health`
- [x] Tests: health endpoint, config validation

### 1.2 — PostgreSQL + ORM Schema ✅
- [x] Async SQLAlchemy engine + session
- [x] ORM models: Tenant, User, Conversation, Message, Document, DocumentChunk, DailyStat
- [x] All ENUMs defined
- [x] Database indexes on tenant_id, created_at, slug, validation_status
- [x] Repository implementations for all domain interfaces
- [ ] Tests: repo CRUD with testcontainers _(deferred — infra repos omitted from coverage, tested via integration tests)_

### 1.3 — Ollama Adapter ✅
- [x] `LLMProvider` ABC with `generate()`, `stream()`, `health_check()`
- [x] `OllamaAdapter` using `openai.AsyncOpenAI` with Cloudflare Access headers
- [x] Streaming via `AsyncGenerator[str, None]`
- [x] Error handling: connection, timeout, model not found → `LLMError`
- [x] Provider factory
- [x] Tests: mock httpx responses, CF header injection

### 1.4 — ChromaDB + Embeddings ✅
- [x] `VectorStore` ABC with `add_documents()`, `search()`, `delete_collection()`
- [x] `ChromaAdapter` namespaced by `tenant_{id}`
- [x] Embedding wrapper for Ollama `/api/embeddings`
- [x] `RecursiveChunker` (chunk_size=512, overlap=50)
- [x] Tests: chunking, chroma add/search/delete

### 1.5 — Bitext Dataset Ingestion ✅
- [x] Download Bitext dataset → `data/bitext/`
- [x] `scripts/seed_demo.py` — CSV → chunk → embed → ChromaDB
- [x] 2 demo tenants: "Acme Store" (all categories), "QuickShip" (SHIPPING + ORDER)
- [ ] Tests: seed idempotency _(deferred — seed script is a one-time setup utility)_

### 1.6 — LangGraph RAG Pipeline ✅
- [x] RAG pipeline with query/context/response/sources/should_escalate state
- [x] Retriever — semantic search, top-k=5
- [x] Grader — relevance assessment
- [x] Generator — cited answer generation
- [x] Escalation — frustration/handoff detection
- [x] Graph wiring: retrieve → grade → generate/escalate
- [x] Tests: node unit tests, full pipeline integration test

### 1.7 — Basic Chat REST Endpoint ✅
- [x] `POST /api/v1/chat` endpoint
- [x] `ChatService` orchestration: conversation → RAG → respond
- [x] Tests: happy path, missing tenant → 422, empty query → 422

### 1.8 — Docker Compose ✅
- [x] Multi-stage Dockerfile
- [x] `docker-compose.yml`: api, postgres, redis, chromadb
- [x] Health checks + depends_on
- [ ] Tests: full suite in Docker _(deferred — will add in Phase 4)_

---

## Phase 2 — Real-time & Admin ✅

> **Branch:** `phase-2/realtime-admin` → merged to `main`

### 2.5 — JWT Authentication ✅
- [x] `app/core/security.py` — bcrypt hashing, JWT access/refresh tokens, TokenPayload
- [x] `app/api/v1/auth.py` — register, login, refresh endpoints
- [x] `get_current_user` dependency with JWT validation
- [x] `require_role()` RBAC factory
- [x] Strong password policy (8-128 chars, mixed case, digit, special char)
- [x] Tests: 30 unit + 11 integration tests (valid/expired/malformed tokens, wrong password, weak password)

### 2.6 — Tenant CRUD + RBAC ✅
- [x] `app/domain/services/tenant_service.py` — slug-unique tenant lifecycle
- [x] `app/api/v1/tenants.py` — admin-only CRUD, authenticated read-by-slug
- [x] Roles enforced: admin (full), viewer (read-only)
- [x] Tests: 12 unit + 7 integration tests (RBAC matrix, cross-tenant isolation)

### 2.7 — Redis Session Cache ✅
- [x] `app/domain/interfaces/cache.py` — CachePort ABC
- [x] `app/infrastructure/cache/redis_adapter.py` — graceful-fallback adapter
- [x] Redis lifespan init/cleanup in `events.py`
- [x] `get_cache` dependency from app.state
- [x] Tests: 11 unit tests (get/set/delete/incr, failure fallback)

### 2.4 — Conversation Persistence ✅
- [x] Relocated `ChatService` → `app/domain/services/chat_service.py`
- [x] `app/api/schemas/conversation.py` — list, detail, message, feedback DTOs
- [x] `app/api/v1/conversations.py` — list, detail, feedback endpoints
- [x] Tenant-scoped isolation for all queries
- [x] Tests: 6 integration tests (list, detail, cross-tenant 404, feedback)

### 2.1 — WebSocket Streaming ✅
- [x] Connection manager (per-tenant tracking)
- [x] `WS /api/v1/ws/chat` route
- [x] Token-by-token streaming from Ollama → WebSocket → client
- [x] JSON frames: `{type: "token"|"source"|"done"|"error", data: ...}`
- [x] Graceful disconnect handling
- [x] **Gap (from Phase 2 review):** Migrate chat endpoint to JWT auth — add `Depends(get_current_user)` and derive `tenant_id` from JWT payload instead of raw `X-Tenant-ID` header
- [x] **Gap (from Phase 2 review):** Wire `ChatService` dependencies via `app.state` / `Depends()` instead of per-request `_build_chat_service()` construction (was planned in master plan Phase 1.1 `dependencies.py`)
- [x] **Gap (from Phase 2 review):** Update `chat_router.py` to import from canonical paths (`app.domain.services.chat_service`, `app.api.schemas.chat`) instead of deprecated shims

### 2.2 — Document Upload API ✅
- [x] Multipart upload: PDF, Markdown, CSV, plain text
- [x] File validation: max 10MB, max 50 files/tenant
- [x] CRUD endpoints: upload, list, status, delete
- [x] `DocumentService` domain service with tenant isolation
- [x] `DocumentResponse`, `DocumentListResponse`, `DocumentUploadResponse` schemas
- [x] Tests: 36 unit + 24 integration (file types, oversized, tenant isolation, RBAC)

### 2.3 — Async Ingestion Worker ✅
- [x] `app/workers/text_extractor.py` — PDF/MD/CSV/TXT extraction with UTF-8 → latin-1 fallback
- [x] `app/domain/services/ingestion_service.py` — pipeline orchestrator: extract → chunk → embed → vector store → DB persist
- [x] `app/workers/ingestion_worker.py` — BackgroundTasks-based async worker with tenant isolation check
- [x] Upload endpoint triggers background ingestion via `BackgroundTasks`
- [x] Status tracking: PENDING → PROCESSING → READY (or FAILED with rollback)
- [x] `document_chunks` table persistence with `chroma_id` reference
- [x] Failure handling: rollback partial chunks, set status=FAILED
- [x] `app/core/events.py` — `embedding_service` and `vector_store` exposed on `app.state` + cleanup on shutdown
- [x] `app/core/dependencies.py` — `get_embedding_service()` and `get_vector_store()` dependency functions
- [x] Tests: 37 unit tests (20 text extractor + 11 ingestion service + 6 ingestion worker) + 2 new lifespan tests

### Known Limitations (Phase 2)

> Documented during code review — to be addressed in future phases.

- **No refresh token revocation (M-2):** Token rotation issues a new refresh token
  on every `/refresh` call, but old tokens remain valid until natural expiry.
  A Redis-backed token blacklist or token-family detection will be added in Phase 4.2.
- **Global admin model (M-5):** ~~Tenant CRUD uses a platform-wide admin role.~~
  **PARTIALLY RESOLVED** in Phase 9 — `SUPERADMIN` role separates platform owner from
  tenant admin. Superadmin passes through `require_role(ADMIN)` guards. Full per-tenant
  admin scoping (admin A cannot manage tenant B) is deferred to Phase 10+ (Tenant Provisioning).
- ~~**`verify_token()` accepts empty `tenant_id`:**~~ **RESOLVED** — `verify_token()` now validates
  non-empty `tenant_id` matching the `user_id` validation pattern.
- ~~**`ChatResponse.created_at` uses deprecated `datetime.utcnow()`:**~~ **RESOLVED** — replaced
  with `datetime.now(timezone.utc)`.
- **Conversation list `total` is page length, not global count:** The `total` field in
  `ConversationListResponse` returns `len(conversations)` (page size), not the actual total
  record count. Needs a separate COUNT query. Fix in Phase 3 (frontend pagination).

---

## Phase 3 — Frontend Integration 🔲

> **Branch:** `phase-3/frontend-integration` (frontend repo)

- See `supportforge-ui` ROADMAP.md

---

## Upcoming Phases (10–22) 🔲

> See `supportforge_plan.md` for detailed task lists and gotchas.

| Phase | Name | Priority | Status |
|---|---|---|---|
| 8 | Feedback Review Queue | High | ✅ |
| 9 | Platform Superadmin Role | High | ✅ |
| 10 | Tenant Provisioning API | High | ✅ |
| 11 | Failed Query Logging & Analytics | High | 🔲 |
| 12 | Tenant Provisioning UI | High | 🔲 |
| 13 | Rate Limiting Middleware | Medium | 🔲 |
| 14 | PII Detection & Masking | Medium | 🔲 |
| 15 | User Approval Workflow | Medium | 🔲 |
| 16 | Role Management API | Medium | 🔲 |
| 17 | User Management UI | Medium | 🔲 |
| 18 | Moderation Dashboard API | Medium | 🔲 |
| 19 | Moderation Dashboard UI | Medium | 🔲 |
| 20 | A/B Testing & Tenant Config | Low | 🔲 |
| 21 | Webhook Integration | Low | 🔲 |
| 22 | Deployment, Docs & E2E | Low | 🔲 |

### Known Gaps (from Phase 2 review)
- [ ] Fix `RedisAdapter.incr()` fail-open bug → Phase 13
- [ ] Add `incr_with_ttl()` to `CachePort` → Phase 13
- [ ] Add Redis-backed refresh token blacklist → Phase 13
- [ ] Generate Alembic migrations → Phase 22
- [ ] Bump `__version__` to `0.2.0` → Phase 22
- [ ] Remove backward-compatibility shims → Phase 22
- [ ] Replace `mypy ignore_errors = true` → Phase 22
- [ ] Fix `.env.example` interpolation → Phase 22
- [ ] Per-tenant admin scoping tests → Phase 22


---

## Phase 5 — Output Validation (Anti-Hallucination Guard) ✅

> **Branch:** `phase-5/output-validation`

### 5.1 — Domain Model Updates ✅
- [x] `ValidationStatus` enum: `passed`, `flagged`, `none`
- [x] `validation_status` field on `Message` domain model (default: `none`)
- [x] `validation_status` column on `MessageModel` ORM model
- [x] Repo layer maps field in both directions

### 5.2 — OutputValidator Domain Service ✅
- [x] `app/domain/services/output_validator.py` — pure domain service (zero framework imports)
- [x] Cross-referenced checks: fabricated phone, email, URL, price, percentage
- [x] Forbidden patterns: LaTeX (`\boxed`, `\text`, `\frac`), third-person refs
- [x] `ValidationResult` + `ValidationViolation` dataclasses
- [x] Disclaimer text appended on flagged responses
- [x] 34 unit tests covering all rules, context pass-through, edge cases

### 5.3 — ChatService Integration ✅
- [x] Validation runs after streaming completes, before done frame
- [x] Disclaimer appended to stored (not streamed) message on flagged responses
- [x] Structured log warnings (`output_validation_failed`) with conversation_id, rule, snippet
- [x] `validation_status` included in done frame and persisted to database
- [x] 6 integration tests (clean passes, fabricated flagged, log verification, context pass-through, LaTeX flagged, escalation bypass)

---

## Phase 6 — Input/Output Content Moderation ✅

> **Branch:** `phase-6/content-moderation`

### 6.1 — ContentModerator Domain Service ✅
- [x] `app/domain/services/content_moderator.py` — pure domain service (zero framework imports)
- [x] `ModerationResult` dataclass with `blocked`, `flagged`, `reason`, `matched_term`, `canned_response`
- [x] 13 compiled regex patterns for jailbreak detection (ignore instructions, pretend, DAN, system prompt, etc.)
- [x] Patterns tightened for precision: `you are now`, `disregard your`, `override your` require jailbreak-specific trailing context to prevent false positives on legitimate customer messages
- [x] Word-boundary matching to prevent false positives (e.g., "reacting" ≠ "act as")
- [x] Tenant-configurable blocklist matching (case-insensitive substring)
- [x] Input moderation (`check_input`): jailbreak + blocklist → blocks with canned response
- [x] Output moderation (`check_output`): blocklist check on LLM-generated text → flags
- [x] 50 unit tests covering clean input, jailbreak patterns, blocklist matching, output moderation, edge cases, and false-positive negative tests

### 6.2 — ChatService Integration ✅
- [x] Input moderation runs BEFORE RAG pipeline (zero LLM cost for blocked inputs)
- [x] Blocked input yields canned response token + done frame with `moderation_blocked=true`
- [x] Output moderation runs AFTER streaming, alongside output validation
- [x] Output flag overrides `validation_status` to `flagged`
- [x] Structured log events: `content_moderation_input_blocked`, `content_moderation_output_flagged`
- [x] `tenant_blocklist` parameter added to both `stream_message()` and `process_message()`
- [x] Both REST (`process_message`) and WebSocket (`stream_message`) paths moderated — no bypass vector
- [x] `matched_term` truncated to 100 chars in all log calls (structured log injection defense)
- [x] 11 integration tests (jailbreak blocked, blocklist blocked, clean passes, output flagged, log verification, default empty blocklist, REST moderation, persist field verification)

### 6.3 — Tenant Configuration ✅
- [x] Blocklist loaded from tenant `config_json["moderation_blocklist"]` in WebSocket handler
- [x] Follows same extraction pattern as per-tenant temperature
- [x] Default: empty list (no tenant-specific blocked terms)

### 6.4 — Moderation Audit Trail ✅
- [x] `moderation_reason` column (VARCHAR 100) on `messages` table — stores `jailbreak_detected` or `blocklist_match`
- [x] `moderation_matched_term` column (VARCHAR 200) on `messages` table — stores the trigger term/pattern
- [x] `ix_messages_validation_status` index for efficient admin queries
- [x] Domain model, ORM model, and repository all map the new fields
- [x] `_persist_exchange()` accepts and stores moderation fields from all call sites
- [x] Blocked exchanges persisted in both REST and WebSocket paths for audit trail
- [x] 5 tests (domain model defaults/set, persist field verification on input block, output flag, REST block)

---

## Phase 7 — Smart Escalation (Sentiment + Repetition + Explicit Request) ✅

> **Branch:** `phase-7/smart-escalation`

### 7.1 — EscalationDetector Domain Service ✅
- [x] `app/domain/services/escalation_detector.py` with pure domain logic
- [x] Sentiment detection: CAPS, punctuation, negative phrases
- [x] Repetition detection: Jaccard similarity on last 3 user messages
- [x] Explicit request detection: compiled regex patterns for handoff
- [x] 42 unit tests covering all detection methods and priority logic

### 7.2 — Domain Model Updates ✅
- [x] `EscalationTrigger` enum
- [x] `escalation_trigger` field on `Conversation` domain model
- [x] `escalation_trigger` column on `ConversationModel` ORM model
- [x] Repo layer mapping and `update_escalation_trigger()` method

### 7.3 — ChatService Integration ✅
- [x] Integrate detectors into `ChatService.stream_message()` and `process_message()`
- [x] Context-aware escalation messages based on trigger type
- [x] Fallback to NO_CONTEXT if RAG retrieval returns no docs
- [x] Persist `escalation_trigger` to database
- [x] Integration tests for sentiment, repetition, explicit, and NO_CONTEXT paths

---

## Phase 8 — Feedback Review Queue ✅

> **Branch:** `phase-8/feedback-review-queue`

### 8.1 — Domain Model Updates ✅
- [x] `reviewed_at` (nullable DateTime) and `reviewed_by` (String) fields on `Message` domain model
- [x] Corresponding ORM columns on `MessageModel`
- [x] `ix_messages_feedback` index for efficient review queries
- [x] `_to_domain` mapping updated for new fields

### 8.2 — Review Queue API Endpoints ✅
- [x] `GET /api/v1/admin/feedback/negative` — paginated negative feedback with user question context
- [x] `GET /api/v1/admin/escalations` — paginated escalated conversations with trigger filter
- [x] `GET /api/v1/admin/flagged` — paginated flagged messages (validation failures)
- [x] `PATCH /api/v1/admin/feedback/{message_id}/review` — mark as reviewed with tenant isolation
- [x] `GET /api/v1/admin/feedback/stats` — aggregate counts for badge display
- [x] All endpoints protected by `require_role(UserRole.ADMIN)`
- [x] Filters: reviewed/unreviewed, date range, escalation trigger type
- [x] 30 new tests (14 schema + 16 integration) — all passing

---

## Phase 9 — Platform Superadmin Role ✅

> **Branch:** `phase-9/platform-superadmin`

### 9.1 — UserRole Enum Expansion ✅
- [x] `SUPERADMIN = "superadmin"` added to `UserRole` enum (platform-wide, not tenant-scoped)
- [x] Enum docstring updated to document superadmin semantics

### 9.2 — User Domain Model ✅
- [x] `is_superadmin` computed property on `User` model (derived from `role == UserRole.SUPERADMIN`)
- [x] No DB migration needed — property is computed, not stored

### 9.3 — JWT Security Layer ✅
- [x] `is_superadmin: bool = False` field on `TokenPayload` (backward-compatible default)
- [x] `create_access_token()` accepts `is_superadmin` parameter, includes in JWT only when `True`
- [x] `verify_token()` reads `is_superadmin` from payload, defaults to `False`
- [x] Existing tokens without `is_superadmin` claim continue to parse correctly

### 9.4 — RBAC Dependencies ✅
- [x] `require_superadmin()` dependency — rejects non-superadmin with 403
- [x] `require_role()` updated: implicitly accepts `SUPERADMIN` when `ADMIN` is in allowed roles
- [x] Existing `require_role(UserRole.ADMIN)` guards unchanged — superadmin passes through

### 9.5 — Auth Router Updates ✅
- [x] Superadmin self-registration blocked at `POST /api/v1/auth/register` (422)
- [x] Login and refresh endpoints include `is_superadmin` claim in JWT

### 9.6 — Bootstrap Script ✅
- [x] `scripts/create_superadmin.py` — CLI script with `--email`, `--password`, `--tenant-id`
- [x] Password strength validation using existing rules
- [x] Idempotent: warns if user already exists
- [x] Validates tenant existence before creation

### 9.7 — Tests ✅
- [x] 24 unit tests: enum, domain property, JWT claims, dependencies, registration blocking
- [x] Backward compatibility: pre-Phase-9 tokens parse without `is_superadmin`
- [x] Existing integration test updated (`test_register_invalid_role` uses truly invalid role)
- [x] 661 total tests passing, zero regressions

---

## Phase 10 — Tenant Provisioning API ✅

> **Branch:** `phase-10/tenant-provisioning-api`
> **Depends on:** Phase 9 (Platform Superadmin)

### 10.1 — TenantStatus Enum + Domain Model ✅
- [x] `TenantStatus` enum: `pending`, `active`, `suspended`, `archived`
- [x] `status` field on `Tenant` domain model (default: `active`)
- [x] `status` field on `TenantCreate` DTO
- [x] ORM `TenantModel.status` column with `server_default="active"`
- [x] Index `ix_tenants_status` on tenants table

### 10.2 — Repository Extensions ✅
- [x] `TenantRepository` ABC: `list_all_with_status()`, `count_all()`, `update_status()`
- [x] `SQLTenantRepository` concrete implementations with status filter + pagination
- [x] `_to_domain()` and `create()` map `status` field

### 10.3 — TenantService Status Transitions ✅
- [x] `VALID_TRANSITIONS` map: pending→active, active→suspended/archived, suspended→active/archived, archived→terminal
- [x] `update_tenant_status()` with transition validation (400 on invalid)
- [x] `list_tenants_with_status()` returning `(list, total)` tuple

### 10.4 — Platform Tenant Endpoints (Superadmin-Only) ✅
- [x] `POST /api/v1/platform/tenants` — create tenant (superadmin only)
- [x] `GET /api/v1/platform/tenants` — paginated list with `?status=` filter
- [x] `PATCH /api/v1/platform/tenants/{id}/status` — status transitions
- [x] All endpoints use `require_superadmin()` dependency
- [x] `PlatformTenantCreateRequest`, `PlatformTenantResponse`, `TenantStatusUpdateRequest` schemas

### 10.5 — Chat Gate Enforcement ✅
- [x] REST chat (`POST /api/v1/chat`): suspended/archived → 403 `TENANT_SUSPENDED`
- [x] WebSocket chat (`/api/v1/ws/chat`): suspended/archived → close code 4003
- [x] `TenantSuspendedError` exception class

### 10.6 — Deprecation + Backward Compatibility ✅
- [x] `POST /api/v1/tenants/` marked `deprecated=True` in OpenAPI
- [x] `TenantResponse` schema includes optional `status` field
- [x] Existing tenant CRUD endpoints pass `status` in responses

### 10.7 — Tests ✅
- [x] 25 unit tests: TenantStatus enum, domain model, service transition validation
- [x] 17 integration tests: platform CRUD, auth enforcement, chat gate, status transitions
- [x] 705 total tests passing, zero regressions
