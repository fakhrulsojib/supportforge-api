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

## Phase 1 — Core RAG Engine 🔲

> **Branch:** `phase-1/core-rag-engine`

### 1.1 — FastAPI Project Scaffold
- [ ] `pyproject.toml` with all dependencies
- [ ] `app/main.py` with app factory pattern (`create_app()`)
- [ ] `app/config.py` with Pydantic `Settings` class
- [ ] `app/core/exceptions.py` — custom exception hierarchy
- [ ] `app/core/middleware.py` — CORS, request-ID, tenant context
- [ ] `app/core/events.py` — startup/shutdown lifecycle
- [ ] `app/core/dependencies.py` — FastAPI Depends injection
- [ ] Health endpoint at `/health`
- [ ] Tests: health endpoint, config validation

### 1.2 — PostgreSQL + Alembic Migrations
- [ ] Async SQLAlchemy engine + session
- [ ] ORM models: Tenant, User, Conversation, Message, Document, DocumentChunk, DailyStat
- [ ] All ENUMs defined
- [ ] Alembic initialized + initial migration
- [ ] Database indexes on tenant_id, created_at, slug
- [ ] Repository implementations for all domain interfaces
- [ ] Tests: repo CRUD with testcontainers, migration up/down

### 1.3 — Ollama Adapter
- [ ] `LLMProvider` ABC with `generate()`, `stream()`, `health_check()`
- [ ] `OllamaAdapter` using `openai.AsyncOpenAI` with Cloudflare Access headers
- [ ] Streaming via `AsyncGenerator[str, None]`
- [ ] Error handling: connection, timeout, model not found → `LLMError`
- [ ] Provider factory
- [ ] Tests: mock httpx responses, CF header injection

### 1.4 — ChromaDB + Embeddings
- [ ] `VectorStore` ABC with `add_documents()`, `search()`, `delete_collection()`
- [ ] `ChromaAdapter` namespaced by `tenant_{id}`
- [ ] Embedding wrapper for Ollama `/api/embeddings`
- [ ] `RecursiveChunker` (chunk_size=512, overlap=50)
- [ ] Tests: chunking, chroma add/search/delete

### 1.5 — Bitext Dataset Ingestion
- [ ] Download Bitext dataset → `data/bitext/`
- [ ] `scripts/seed_demo.py` — CSV → chunk → embed → ChromaDB
- [ ] 2 demo tenants: "Acme Store" (all categories), "QuickShip" (SHIPPING + ORDER)
- [ ] Tests: seed idempotency

### 1.6 — LangGraph RAG Pipeline
- [ ] LangGraph `StateGraph` with query/context/response/sources/is_relevant/should_escalate
- [ ] Retriever node — semantic search, top-k=5
- [ ] Grader node — relevance assessment
- [ ] Generator node — cited answer generation
- [ ] Escalation node — frustration/handoff detection
- [ ] Graph wiring: retrieve → grade → generate/escalate
- [ ] Tests: node unit tests, full graph integration test

### 1.7 — Basic Chat REST Endpoint
- [ ] `POST /api/v1/chat` endpoint
- [ ] `ChatService` orchestration: conversation → RAG → persist → respond
- [ ] Tests: happy path, missing tenant → 400, empty query → 422

### 1.8 — Docker Compose
- [ ] Multi-stage Dockerfile
- [ ] `docker-compose.yml`: api, postgres, redis, chromadb
- [ ] Health checks + depends_on
- [ ] Tests: full suite in Docker

---

## Phase 2 — Real-time & Admin 🔲

> **Branch:** `phase-2/realtime-admin`

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

### 2.4 — Conversation Persistence
- [ ] Full message history in PostgreSQL
- [ ] Multi-turn context loading
- [ ] Conversation CRUD endpoints
- [ ] Feedback endpoint: PATCH messages/{id}/feedback

### 2.5 — JWT Authentication
- [ ] Access tokens (15min) + refresh tokens (7d in Redis)
- [ ] Register, login, refresh endpoints
- [ ] `get_current_user` dependency

### 2.6 — Tenant CRUD + RBAC
- [ ] Tenant CRUD endpoints (admin only)
- [ ] Roles: admin, agent, viewer
- [ ] `X-Tenant-ID` header middleware
- [ ] Cross-tenant isolation enforcement

### 2.7 — Redis Session Cache
- [ ] Conversation context cache (last 10 messages, TTL 1h)
- [ ] Rate limit counters
- [ ] Refresh token storage
- [ ] DB fallback on Redis failure

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
