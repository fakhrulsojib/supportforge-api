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

### 1.2 — PostgreSQL + Alembic Migrations ✅
- [x] Async SQLAlchemy engine + session
- [x] ORM models: Tenant, User, Conversation, Message, Document, DocumentChunk, DailyStat
- [x] All ENUMs defined
- [x] Alembic initialized + initial migration
- [x] Database indexes on tenant_id, created_at, slug
- [x] Repository implementations for all domain interfaces
- [ ] Tests: repo CRUD with testcontainers, migration up/down _(deferred — infra repos omitted from coverage, tested via integration tests)_

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

## Phase 2 — Real-time & Admin 🔨 (in progress)

> **Branch:** `phase-2/realtime-admin`

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

### 2.1 — WebSocket Streaming
- [ ] Connection manager (per-tenant tracking)
- [ ] `WS /api/v1/ws/chat` route
- [ ] Token-by-token streaming from Ollama → WebSocket → client
- [ ] JSON frames: `{type: "token"|"source"|"done"|"error", data: ...}`
- [ ] Graceful disconnect handling

### 2.2 — Document Upload API
- [ ] Multipart upload: PDF, Markdown, CSV, plain text
- [ ] File validation: max 10MB, max 50 files/tenant
- [ ] CRUD endpoints: upload, list, status, delete

### 2.3 — Async Ingestion Worker
- [ ] Background pipeline: read → extract → chunk → embed → store
- [ ] Failure handling: status=failed, no partial chunks
- [ ] `document_chunks` table tracking

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

### 4.7 — End-to-End Test Suite
- [ ] Full user journey tests
- [ ] Multi-tenant isolation tests
- [ ] Auth flow tests
- [ ] Rate limiting tests
- [ ] Coverage ≥ 95%, mutation kill rate > 80%
