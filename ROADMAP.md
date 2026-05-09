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
- **Global admin model (M-5):** Tenant CRUD uses a platform-wide admin role — an admin
  authenticated under tenant A can manage any tenant. Per-tenant admin scoping
  (cross-tenant isolation for admins) will be evaluated in Phase 4.
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

## Phase 4 — Production Polish 🔲

> **Branch:** `phase-4/production-polish`

### 4.1 — A/B Testing
- [ ] Tenant config: model, prompt variant, temperature
- [ ] Admin config endpoint
- [ ] Per-message variant logging

### 4.2 — Rate Limiting
- [ ] Redis-backed rate limiting middleware
- [ ] Per-tenant + per-user limits (configurable)
- [ ] 429 + Retry-After header
- [ ] **Gap (from Phase 2 review):** Fix `RedisAdapter.incr()` fallback — currently returns `0` on failure which bypasses rate limiting. Should return `limit + 1` (deny by default) or raise
- [ ] **Gap (from Phase 2 review):** Add `expire(key, ttl)` or `incr_with_ttl()` to `CachePort` for atomic rate limit counter + TTL
- [ ] **Gap (from Phase 2 review):** Add Redis-backed refresh token blacklist (resolves Known Limitation M-2)

### 4.3 — Embeddable Chat Widget
- [ ] Standalone JS bundle in `widget/`
- [ ] Shadow DOM isolation
- [ ] Tenant-scoped WebSocket

### 4.4 — Webhook Integration
- [ ] Tenant webhook_url config
- [ ] Events: new conversation, escalation, negative feedback
- [ ] Slack + Discord format support
- [ ] Retry with exponential backoff

### 4.5 — Email Digest
- [ ] Daily scheduled aggregation
- [ ] HTML email with unresolved conversations, feedback, top queries
- [ ] SMTP integration

### 4.6 — Deployment & Documentation
- [ ] `docker-compose.prod.yml` with Nginx, SSL, resource limits
- [ ] Deployment guides: Docker Compose, Railway, Render
- [ ] README update with badges, screenshots, architecture diagram
- [ ] **Gap (from Phase 2 review):** Generate Alembic migration for Phase 2 model changes (new fields, new tables)
- [ ] **Gap (from Phase 2 review):** Bump `__version__` to `0.2.0` (Phase 2 deprecation warning already references it)
- [ ] **Gap (from Phase 2 review):** Remove backward-compatibility shims (`app/api/v1/schemas.py`, `app/api/v1/chat_service.py`)
- [ ] **Gap (from Phase 2 review):** Replace `mypy ignore_errors = true` with targeted `# type: ignore[specific-error]` annotations in infrastructure layer
- [ ] **Gap (from Phase 2 review):** Fix `.env.example` — remove `${VAR}` shell interpolation syntax (pydantic-settings doesn't expand these)

### 4.7 — End-to-End Test Suite
- [ ] Full user journey tests
- [ ] Multi-tenant isolation tests
- [ ] Auth flow tests
- [ ] Rate limiting tests
- [ ] Coverage ≥ 95%, mutation kill rate > 80%
- [ ] **Gap (from Phase 2 review):** Per-tenant admin scoping tests (resolves Known Limitation M-5)

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
