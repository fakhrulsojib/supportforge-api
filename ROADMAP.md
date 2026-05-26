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
- [x] `VectorStore` ABC with `add_documents()`, `search()`, `delete_collection()`, `get_all_documents()`
- [x] `ChromaAdapter` namespaced by `tenant_{id}`
- [x] Embedding wrapper for Ollama `/api/embeddings`
- [x] `RecursiveChunker` (chunk_size=2500, overlap=300)
- [x] Tests: chunking, chroma add/search/delete

### 1.5 — Bitext Dataset Ingestion ✅
- [x] Download Bitext dataset → `data/bitext/`
- [x] `scripts/seed_demo.py` — CSV → chunk → embed → ChromaDB
- [x] 2 demo tenants: "Acme Store" (all categories), "QuickShip" (SHIPPING + ORDER)
- [ ] Tests: seed idempotency _(deferred — seed script is a one-time setup utility)_

### 1.6 — LangGraph RAG Pipeline ✅
- [x] RAG pipeline with query/context/response/sources/should_escalate state
- [x] Retriever — hybrid search (vector + BM25 + weighted RRF fusion + optional cross-encoder reranker), configurable k, weights, and toggles
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
- [x] `app/domain/services/ingestion_service.py` — pipeline orchestrator: extract → chunk → contextualise → embed → vector store → DB persist
- [x] `app/rag/contextualizer.py` — Anthropic's Contextual Retrieval: LLM-generated context prepended to each chunk before embedding for improved retrieval accuracy
- [x] `app/workers/ingestion_worker.py` — BackgroundTasks-based async worker with tenant isolation check
- [x] Upload endpoint triggers background ingestion via `BackgroundTasks`
- [x] Status tracking: PENDING → PROCESSING → READY (or FAILED with rollback)
- [x] `document_chunks` table persistence with `chroma_id` reference
- [x] Failure handling: rollback partial chunks, set status=FAILED
- [x] `app/core/events.py` — `embedding_service` and `vector_store` exposed on `app.state` + cleanup on shutdown
- [x] `app/core/dependencies.py` — `get_embedding_service()`, `get_vector_store()`, and `get_llm_provider_dep()` dependency functions
- [x] Tests: 37 unit tests (20 text extractor + 15 ingestion service + 6 ingestion worker) + 2 new lifespan tests

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

## Phase 3 — Frontend Integration ✅

> **Branch:** `phase-3/frontend-implementation` (frontend repo)

- See `supportforge-ui` ROADMAP.md

---

## Phase 4 — Response Timeout & Streaming Fallback ✅

> **Branch:** `phase-4/response-timeout`

- [x] First-token timeout (60s) on `OllamaAdapter.stream()`
- [x] Retry with user-facing fallback frame on timeout
- [x] Double-timeout escalation with `escalated=True` in done frame
- [x] `StreamingIndicator.jsx` elapsed-time animation ("Still thinking...")
- [x] Reduced httpx read timeout from 300s to 120s
- [x] Tests: slow stream mock, retry success, double-timeout escalation

---

## Upcoming Phases (10–23) 🔲

> See `supportforge_plan.md` for detailed task lists and gotchas.

| Phase | Name | Priority | Status |
|---|---|---|---|
| 8 | Feedback Review Queue | High | ✅ |
| 9 | Platform Superadmin Role | High | ✅ |
| 10 | Tenant Provisioning API | High | ✅ |
| 11 | Failed Query Logging & Analytics | High | ✅ |
| 12 | Tenant Provisioning UI + Failed Queries UI | High | ✅ |
| 13 | Analytics Backend API | High | ✅ |
| V1 | Voice Pipeline (STT/TTS + Pipecat) | High | 🔧 (`feature/voice-v1`) |
| WGT | Widget SDK + Event Hooks | High | ✅ (`feature/widget-sdk-event-hooks`) |
| 14 | Rate Limiting Middleware | Medium | 🔲 |
| 15 | PII Detection & Masking | Medium | 🔲 |
| 16 | User Approval Workflow | Medium | 🔲 |
| 17 | Role Management API | Medium | 🔲 |
| 18 | User Management UI | Medium | 🔲 |
| 19 | Moderation Dashboard API | Medium | 🔲 |
| 20 | Moderation Dashboard UI | Medium | 🔲 |
| 21 | A/B Testing & Tenant Config | Low | 🔲 |
| 22 | Webhook Integration | Low | ✅ _(covered by `feature/pluggable-tool-system` + `feature/widget-sdk-event-hooks`)_ |
| 23 | Deployment, Docs & E2E | Low | 🔲 |

### Known Gaps (from Phase 2 review)
- [ ] Fix `RedisAdapter.incr()` fail-open bug → Phase 14
- [ ] Add `incr_with_ttl()` to `CachePort` → Phase 14
- [ ] Add Redis-backed refresh token blacklist → Phase 14
- [ ] Generate Alembic migrations → Phase 23
- [ ] Bump `__version__` to `0.2.0` → Phase 23
- [ ] Remove backward-compatibility shims → Phase 23
- [ ] Replace `mypy ignore_errors = true` → Phase 23
- [ ] Fix `.env.example` interpolation → Phase 23
- [ ] Per-tenant admin scoping tests → Phase 23

---

## Phase V1 — Voice Pipeline (STT/TTS + Pipecat) 🔧

> **Branch:** `feature/voice-v1`
> **Status:** Feature branch — pending merge to `main`

### V1.1 — Domain Layer ✅
- [x] `STTProvider` ABC with `transcribe()`, `warm_up()`, `health_check()`
- [x] `TTSProvider` ABC with `synthesize()`, `synthesize_stream()`, `list_voices()`, `warm_up()`, `health_check()`
- [x] `VoiceFrame` frozen dataclass, `VoiceFrameType` constants
- [x] `MessageChannel` enum (`text`, `voice`)
- [x] `STTError`, `TTSError`, `VoiceBusyError` exception classes
- [x] Tests: 23 unit tests (ABC enforcement, enums, value objects, exceptions)

### V1.2 — STT/TTS Infrastructure ✅
- [x] `WhisperAdapter` — faster-whisper with `asyncio.to_thread` offloading, configurable max_audio_bytes
- [x] `PiperAdapter` — piper-tts with sentence-boundary streaming, regex split
- [x] `get_stt_provider()` and `get_tts_provider()` factories with lazy imports
- [x] Tests: 20 unit tests (adapter, factory, error wrapping)

### V1.3 — Tenant Voice Config ✅
- [x] `TenantVoiceConfig` frozen dataclass
- [x] `resolve_tenant_voice_config()` — 3-tier resolver (cloud → local → disabled), never raises
- [x] Voice settings in `app/config.py`: `voice_stt_model`, `voice_tts_voice`, `voice_tts_sample_rate`, `voice_max_audio_bytes`, `voice_max_sessions_per_tenant`
- [x] `MessageChannel` column on `MessageModel` ORM (`channel` with `server_default="text"`)
- [x] `.env.example` updated with voice section
- [x] Tests: 12 unit tests (resolver tiers, edge cases, clamping)

### V1.4 — Pipecat Integration ✅
- [x] `SupportForgeRAGProcessor` — bridges ChatService stream to Pipecat frames
- [x] `PipecatSTTAdapter` and `PipecatTTSAdapter` — domain adapter wrappers
- [x] `VoiceSessionManager` — per-tenant `asyncio.Semaphore` concurrency with atomic check+acquire
- [x] `create_voice_pipeline()` factory
- [x] Tests: 18 unit tests (processor, adapters, session manager, factory)

### V1.5 — REST API Endpoints ✅
- [x] `GET /api/v1/voice/config` — tenant voice availability (reads from DB)
- [x] `GET /api/v1/voice/health` — STT/TTS health status
- [x] `GET /api/v1/voice/sessions` — active session count (admin only)
- [x] Router registered in `main.py`
- [x] Tests: 7 integration tests (auth, config resolution, health, tenant-not-found)

### V1.6 — Frontend UI ✅ _(supportforge-ui)_
- [x] `voiceApi.js` — centralized voice API client
- [x] `useVoice.js` — custom hook (MediaRecorder, state machine)
- [x] `VoiceButton.jsx` — 5-state component (idle, listening, processing, speaking, error)
- [x] `VoiceButton.css` — animations, responsive, dark-mode-safe
- [x] Voice API routes in `constants.js`

### V1.7 — Deployment & Config ✅
- [x] `pyproject.toml` — `voice` and `voice-cloud` optional dependency groups
- [x] `Dockerfile` — `INSTALL_VOICE` build arg, conditional `espeak-ng` installation
- [x] `scripts/migrate_add_channel.py` — idempotent migration for existing databases


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

---

## Phase 11 — Failed Query Logging & Analytics ✅

> **Branch:** `phase-11/failed-query-logging`

### 11.1 — Domain Model & Enum ✅
- [x] `FailureReason` enum: `no_docs`, `low_relevance`, `llm_error`, `timeout`
- [x] `FailedQuery` domain model (pure Pydantic, zero framework imports)
- [x] `FailedQueryRepository` ABC with `create`, `get_by_id`, `list_by_tenant`, `mark_resolved`, `count_unresolved`, `get_stats`

### 11.2 — ORM Model & SQL Repository ✅
- [x] `FailedQueryModel` ORM class: `failed_queries` table with proper indexes
- [x] `SQLFailedQueryRepository` concrete implementation
- [x] SQL aggregation for stats: reason breakdown, top 10 repeated queries, daily trend

### 11.3 — ChatService Integration ✅
- [x] `_persist_failed_query()` best-effort method (mirrors `_persist_exchange()` pattern)
- [x] Wired into `process_message()` RAG escalation path (`should_escalate=True`)
- [x] Wired into `stream_message()` RAG escalation path (`should_escalate=True`)
- [x] Extracts `retrieved_doc_count` and `max_relevance_score` from RAG state

### 11.4 — Admin API Endpoints ✅
- [x] `GET /api/v1/admin/failed-queries` — paginated list with filters (failure_reason, resolved, date range)
- [x] `PATCH /api/v1/admin/failed-queries/{id}/resolve` — mark as resolved with tenant ownership check
- [x] `GET /api/v1/admin/failed-queries/stats` — aggregated stats (reason breakdown, top queries, daily trend)
- [x] All endpoints enforce `require_role(UserRole.ADMIN)`
- [x] `ReviewStatsResponse` extended with `unresolved_failed_queries` count

### 11.5 — API Schemas ✅
- [x] `FailedQueryResponse`, `FailedQueryListResponse`, `FailedQueryResolveResponse`, `FailedQueryStatsResponse`
- [x] Router registered in `main.py`

### 11.6 — Tests ✅
- [x] 11 unit tests: FailureReason enum, FailedQuery domain model
- [x] 7 unit tests: Failed query API schemas
- [x] 14 integration tests: RBAC, list/filter, resolve, cross-tenant isolation, stats
- [x] Updated existing review stats test for new `unresolved_failed_queries` field
- [x] 737 total tests passing, zero regressions

---

## Phase 13 — Analytics Backend API ✅

> Real-time analytics endpoints for the admin dashboard. No Alembic migration — tables already exist.

### 13.1 — Domain Models & Interface ✅
- [x] `DailyStatEntry`, `IntentEntry`, `SatisfactionSummary` Pydantic models (zero framework imports)
- [x] `AnalyticsRepository` ABC with `get_daily_stats`, `get_top_intents`, `get_satisfaction_summary`

### 13.2 — SQL Repository ✅
- [x] `SQLAnalyticsRepository` with real-time SQL aggregation
- [x] Daily stats: GROUP BY DATE on conversations + messages (tenant-scoped via JOIN)
- [x] Top intents: `sources_json` filename extraction with Python Counter
- [x] Satisfaction: positive/negative feedback counts with rate computation

### 13.3 — Domain Service ✅
- [x] `AnalyticsService` thin orchestrator with parameter clamping (days 1–365, limit 1–100)
- [x] Pure domain layer (zero framework imports)

### 13.4 — API Schemas & Router ✅
- [x] `DailyStatsResponse`, `TopIntentsResponse`, `SatisfactionResponse` DTOs
- [x] `GET /api/v1/analytics/daily-stats?days=30` — admin-only
- [x] `GET /api/v1/analytics/top-intents?limit=10` — admin-only
- [x] `GET /api/v1/analytics/satisfaction` — admin-only
- [x] All scoped to JWT user's `tenant_id`
- [x] Router registered in `main.py`

### 13.5 — Frontend Comment Cleanup ✅
- [x] Removed stale Phase 13 references from `analyticsApi.js` and `AnalyticsPage.jsx`

### 13.6 — Tests ✅
- [x] 15 unit tests: DailyStatEntry, IntentEntry, SatisfactionSummary domain models
- [x] 9 unit tests: analytics API schemas serialization and edge cases
- [x] 12 integration tests: RBAC, happy path, empty state, parameter forwarding, boundaries
- [x] 773 total tests passing (737 + 36 new), zero regressions

---

## Phase 14a — Hybrid Retrieval Pipeline ✅

> **Branch:** `main` (direct commit)

### 14a.1 — BM25 Keyword Retriever ✅
- [x] `app/rag/bm25_retriever.py` — stateless BM25 search with min-max score normalization
- [x] Whitespace tokenization (case-insensitive)
- [x] Returns same dict shape as vector search (`content`, `metadata`, `score`, `id`)

### 14a.2 — Weighted RRF Fusion ✅
- [x] `app/rag/fusion.py` — Reciprocal Rank Fusion with per-list weights
- [x] Preserves original `score` for downstream grading; adds `rrf_score` for observability
- [x] Supports N-way fusion (2+ ranked lists)

### 14a.3 — Reranker Interface + Adapters ✅
- [x] `app/domain/interfaces/reranker.py` — `Reranker` ABC
- [x] `app/infrastructure/reranker/noop_reranker.py` — pass-through (default)
- [x] `app/infrastructure/reranker/cross_encoder_reranker.py` — lazy-loaded cross-encoder (optional `[reranker]` dependency)
- [x] `app/infrastructure/reranker/factory.py` — config-driven factory with import-guarded fallback

### 14a.4 — Pipeline Integration ✅
- [x] `retrieve_node` upgraded to 4-stage hybrid pipeline (zero signature change)
- [x] BM25 path wrapped in graceful degradation (falls back to vector-only on error)
- [x] `get_all_documents()` added to `VectorStore` ABC and `ChromaAdapter`

### 14a.5 — Config + Validation ✅
- [x] 13 new config parameters in `app/config.py` (toggles, k values, weights, reranker model)
- [x] Pydantic validators: k values ≥ 1, RRF k ≥ 1 (prevents division by zero), weights ≥ 0.0
- [x] `rank_bm25` core dependency; `sentence-transformers` + `torch` optional `[reranker]` group

### 14a.6 — Tests ✅
- [x] 11 unit tests: BM25 retriever (empty corpus, normalization, ranking, edge cases)
- [x] 12 unit tests: RRF fusion (weighting, deduplication, k parameter, multi-way)
- [x] 5 unit tests: Reranker (NoOp, factory, interface, fallback)
- [x] 2 pipeline tests: hybrid integration, BM25 degradation
- [x] 775 total tests passing, zero regressions

---

## Pluggable Tool System ✅

> **Branch:** `feature/pluggable-tool-system`
> **Status:** Feature branch — pending merge to `main`

### PTS.1 — Unified RAG Pipeline (LangGraph Graph) ✅
- [x] `app/rag/graph.py` — LangGraph graph with retrieve→grade→decide flow
- [x] Extracted duplicated RAG logic from `process_message`/`stream_message` into reusable graph
- [x] `RAGState` TypedDict extended with tool fields in `app/rag/pipeline.py`
- [x] `app/rag/prompt_builder.py` — shared `build_rag_messages()` for unified prompt construction
- [x] Tests: 7 unit tests (`tests/unit/rag/test_graph.py`), 32 unit tests (`tests/unit/rag/test_prompt_builder.py`)

### PTS.2 — Tenant Agent Personality ✅
- [x] Per-tenant `agent_config` (custom_prompt, domain_rules, custom_instructions) in tenant config
- [x] Prompt sandwich defense for injection-resistant personality enforcement
- [x] Unified prompt construction via `prompt_builder.py`

### PTS.3 — Tool Definitions & Executor ✅
- [x] `app/rag/tools/__init__.py` — tool system package
- [x] `app/rag/tools/base.py` — `ToolDefinition`, `ToolResult`, `ESCALATE_TOOL_DEFINITION`
- [x] `app/rag/tools/executor.py` — `ToolExecutor` with `asyncio.wait_for` timeout, 50KB response size limits, error isolation
- [x] `app/rag/tools/resolver.py` — `resolve_tenant_tools`, `BuiltinEscalateTool`, per-tool auth headers and base URL support
- [x] Tests: 24 unit tests (`tests/unit/rag/test_tools.py`)

### PTS.4 — Tool Loop & WebhookTool ✅
- [x] `app/rag/tools/tool_loop.py` — `run_tool_loop` for multi-turn LLM↔tool interaction (configurable `max_rounds`)
- [x] `app/rag/tools/webhook.py` — `WebhookTool` for tenant-configured external API endpoints (GET/POST/PUT/PATCH/DELETE)
- [x] SSRF protection: async DNS resolution, IPv4-mapped IPv6 blocking, private/loopback/reserved IP blocking
- [x] Circuit breaker for repetitive failed tool calls (auto-escalates)
- [x] Shared `httpx.AsyncClient` with `asyncio.Lock` for connection pooling
- [x] `tool_answer` passthrough (skip LLM regeneration when tools provide the answer)
- [x] Tests: 13 unit tests (`tests/unit/rag/test_tool_loop.py`)

### PTS.5 — Config Validators ✅
- [x] `app/core/config_validators.py` — Pydantic validators with schema enforcement for tool/tenant config
- [x] Tests: 26 unit tests (`tests/unit/test_config_validators.py`)

### PTS.6 — Tenant Secrets API ✅
- [x] `app/api/v1/tenant_secrets.py` — CRUD API for encrypted tenant secrets (Admin)
- [x] `app/infrastructure/database/repositories/tenant_secret_repo.py` — Fernet-encrypted secret storage
- [x] Race condition fix (`begin_nested` SAVEPOINT) in secret upsert
- [x] Endpoints: `POST /api/v1/tenants/{id}/secrets`, `GET /api/v1/tenants/{id}/secrets`, `DELETE /api/v1/tenants/{id}/secrets/{key}`

### PTS.7 — LLM Adapter Updates ✅
- [x] `app/domain/interfaces/llm_provider.py` — added `generate_with_tools`, `ToolCall`, `ToolAwareResponse`
- [x] `app/infrastructure/llm/gemini_adapter.py` — `generate_with_tools` implementation with graceful `JSONDecodeError` handling for malformed tool args
- [x] `app/infrastructure/llm/ollama_adapter.py` — `generate_with_tools` implementation
- [x] 100KB payload truncation for LLM-generated arguments

### PTS.8 — ChatService Integration ✅
- [x] `app/domain/services/chat_service.py` — tool loop integration in both `process_message` and `stream_message`

---

## Widget SDK Backend + Event Hooks ✅

> **Branch:** `feature/widget-sdk-event-hooks`
> **Status:** Feature branch — pending merge to `main`

### WGT.1 — Widget Session & Auth ✅
- [x] `app/core/widget_token.py` — `create_widget_token()`, `verify_widget_token()` with `ws_` prefix, `WidgetTokenPayload` dataclass
- [x] `app/api/schemas/widget.py` — `WidgetSessionRequest`, `WidgetSessionResponse`, `WidgetUIConfigResponse` DTOs
- [x] `app/api/v1/widget.py` — Widget endpoints:
  - `POST /api/v1/widget/session` — embed_key validation, origin domain matching, IP rate limiting
  - `GET /api/v1/widget/ui-config/{slug}` — public UI config (theme, branding) without exposing secrets
- [x] `app/config.py` — `widget_cors_origins`, `widget_session_expire_minutes`, `widget_rate_limit_per_ip`
- [x] `app/core/middleware.py` — merged `widget_cors_origin_list` into CORS origins
- [x] `app/main.py` — widget router registration
- [x] `.env.example` — widget SDK section
- [x] Tests: 19 unit tests (`tests/unit/core/test_widget_token.py`), 27 unit tests (`tests/unit/test_widget.py`)

### WGT.2 — WebSocket Dual Auth ✅
- [x] `app/api/v1/chat_ws.py` — dual authentication: JWT (admin/agent) + `ws_` widget token (anonymous visitors)
- [x] `_ResolvedConnection` dataclass — shared auth result for both paths
- [x] `_extract_tenant_config()` — DRY tenant config extraction (eliminates prior duplication)
- [x] Anonymous widget visitors: `user_id=""` → NULL `user_id` in conversations table

### WGT.3 — Outbound Event Hooks ✅
- [x] `app/core/event_hooks.py` — `dispatch_event()` fire-and-forget async dispatcher
- [x] `EventType` enum: `ON_ESCALATION`, `ON_NEW_CONVERSATION`, `ON_TOOL_FAILURE`, `ON_NEGATIVE_FEEDBACK`
- [x] `HookPayload` dataclass with auto-generated ISO timestamp
- [x] `_send_hook()` — SSRF-protected HTTP POST with Host header injection (never crashes the request)
- [x] `app/domain/services/chat_service.py` — event dispatch:
  - `ON_ESCALATION` — 5 points (process_message + stream_message smart/RAG/tool/post-gen)
  - `ON_NEW_CONVERSATION` — 2 points (process_message + stream_message, fires when `is_new_conversation=True`)
  - `ON_TOOL_FAILURE` — 2 points (process_message + stream_message, fires for each `success=False` tool result)
- [x] `app/api/v1/conversations.py` — `ON_NEGATIVE_FEEDBACK` dispatch on `FeedbackType.NEGATIVE`
- [x] Tests: 16 unit tests (`tests/unit/core/test_event_hooks.py`)

