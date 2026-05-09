# SupportForge API

> Production-grade AI customer support agent — Backend API

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7+-DC382D?style=flat-square&logo=redis&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-FF6F00?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-RAG-1C3C3C?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

## Overview

SupportForge is a multi-tenant AI customer support agent powered by a self-hosted Ollama LLM, RAG (Retrieval-Augmented Generation) via LangGraph, and real-time WebSocket streaming. It provides intelligent, context-aware responses grounded in your organization's knowledge base.

### Key Features

- **RAG Pipeline** — LangGraph state machine with semantic retrieval, relevance grading, and source-cited answers
- **Multi-Tenant** — Full data isolation per tenant with RBAC (admin, agent, viewer)
- **Real-Time Streaming** — Token-by-token WebSocket responses for instant chat UX
- **Self-Hosted LLM** — Zero-cost inference via Ollama behind Cloudflare Access
- **Document Ingestion** — Upload PDF, Markdown, CSV, and plain text to build your knowledge base
- **Conversation Memory** — Full audit trail in PostgreSQL with feedback tracking
- **Analytics** — Daily stats, intent classification, satisfaction metrics
- **Output Validation** — Anti-hallucination guard detects fabricated contact info, prices, and forbidden patterns with context cross-referencing
- **Content Moderation** — Input filtering (jailbreak detection, tenant blocklist) and output flagging with full DB audit trail
- **Smart Escalation** — Context-aware human handoff triggered by frustrated sentiment, repeated questions, or explicit user requests

## Architecture

```
Hexagonal Architecture (Ports & Adapters)
┌──────────────────────────────────────────────┐
│                 DOMAIN CORE                   │
│   (Pure Python — no FastAPI, no SQLAlchemy)   │
│   models/ ← services/ → interfaces/ (ports)  │
└──────────────┬───────────────────┬────────────┘
               │                   │
    ┌──────────▼──────┐ ┌─────────▼───────────┐
    │  API Layer       │ │  Infrastructure      │
    │  (FastAPI routes │ │  (adapters)          │
    │   + schemas)     │ │  DB, LLM, Vector,   │
    │                  │ │  Redis, WebSocket    │
    └──────────────────┘ └─────────────────────┘
```

## Tech Stack

| Component | Technology |
|---|---|
| Framework | FastAPI (async) |
| LLM | Ollama (self-hosted, OpenAI-compatible) |
| RAG | LangGraph + ChromaDB |
| Database | PostgreSQL (SQLAlchemy async) |
| Cache | Redis |
| Auth | JWT (access + refresh tokens) |
| Streaming | WebSocket |
| Validation | Pydantic v2 |
| Logging | structlog (JSON) |
| Testing | pytest + testcontainers + hypothesis |

## Quick Start

> **Prerequisites:** Docker & Docker Compose

```bash
# 1. Clone the repo
git clone https://github.com/fakhrulsojib/supportforge-api.git
cd supportforge-api

# 2. Copy environment config
cp .env.example .env
# Edit .env with your Ollama credentials and model names

# 3. Start all services
docker compose up -d

# 4. Create tenant and seed demo data
docker compose exec api python scripts/create_tenant.py
docker compose exec api python scripts/seed_demo.py

# 5. Verify
curl http://localhost:8000/health
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest --cov --cov-branch --cov-fail-under=95

# Type checking
mypy app/ --strict

# Linting
ruff check app/
```

## Project Structure

```
supportforge-api/
├── app/
│   ├── main.py                    # FastAPI app factory
│   ├── config.py                  # Pydantic Settings
│   ├── core/                      # Security, middleware, dependencies
│   ├── domain/                    # Pure business logic (models, services, interfaces)
│   ├── infrastructure/            # Adapters (DB, LLM, vector, cache, WebSocket)
│   ├── rag/                       # LangGraph RAG pipeline
│   ├── api/                       # HTTP + WebSocket endpoints
│   └── workers/                   # Background tasks
├── tests/                         # Unit, integration, E2E tests
├── data/                          # Bitext dataset
├── scripts/                       # Seed & utility scripts
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | — | Health check |
| `POST` | `/api/v1/auth/register` | — | Register user |
| `POST` | `/api/v1/auth/login` | — | Login |
| `POST` | `/api/v1/auth/refresh` | — | Refresh token |
| `POST` | `/api/v1/chat` | JWT | Send chat message |
| `GET` | `/api/v1/conversations` | JWT | List conversations |
| `GET` | `/api/v1/conversations/{id}` | JWT | Get conversation detail |
| `PATCH` | `/api/v1/conversations/messages/{id}/feedback` | JWT | Update message feedback |
| `POST` | `/api/v1/tenants` | Admin | Create tenant |
| `GET` | `/api/v1/tenants/{slug}` | JWT | Get tenant by slug |
| `PATCH` | `/api/v1/tenants/{id}` | Admin | Update tenant |
| `DELETE` | `/api/v1/tenants/{id}` | Admin | Delete tenant |

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full implementation plan and progress tracking.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
