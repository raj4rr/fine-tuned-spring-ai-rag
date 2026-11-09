# Spring AI + Ollama + RAG + React + Fine-Tuning Studio

Production-oriented starter implementing:

React UI → Spring Boot 4 → Spring AI 2.0.2 → Ollama → RAG (PGVector) + GUI Fine-Tuning

---

## Architecture Overview

```text
React Web UI (Port 5173)
  │
  ├── [Chat Prompt / Test Bench]
  │     │
  │     ▼
  │   Spring Boot 4 API (Port 8080)
  │     ├── ChatController & RAG Orchestration
  │     │     ├── PGVector VectorStore (Port 5433) ──> Hybrid Similarity Search
  │     │     └── Ollama Local Inference (Port 11434) ──> qwen2.5:1.5b / Custom Models
  │     │
  │     └── FineTuningController & FineTuningService
  │           ├── Dataset Storage ──> fine-tune/dataset/*.jsonl
  │           ├── Modelfile Generation ──> fine-tuning/output/<model>/Modelfile
  │           └── Ollama API Engine ──> Automatic Model Compilation & Registration
  │
  └── [Dataset Ingestion & Management]
```

- **Fine-Tuning = Behavior, Terminology & Domain Personas**
- **RAG = Dynamic, Grounded & Live Knowledge Retrieval**
- **Spring AI = Enterprise Orchestration & Guardrails**
- **Ollama = Local GPU/CPU Model Inference**
- **PostgreSQL / PGVector = Semantic Dense Vector Search**

---

## Requirements

- **JDK 21+** (e.g. OpenJDK 21/27)
- **Maven 3.9+**
- **Docker & Docker Compose**
- **Node.js 20+** & **npm**

---

## Quickstart with `run.sh`

A unified runner script [`run.sh`](run.sh) is provided to manage infrastructure, models, backend, and frontend automatically.

```bash
# Start full application (Postgres + Ollama + Model Checks + Spring Boot + React)
./run.sh
```

### CLI Subcommands

```bash
# Start only Docker infrastructure and pull required Ollama models
./run.sh infra

# Start only the Spring Boot backend
./run.sh backend

# Start only the React frontend
./run.sh frontend

# Pull or verify required Ollama models
./run.sh models

# Check status of containers, Ollama models, and health endpoints
./run.sh status

# Stop all Docker containers
./run.sh stop

# Restart infrastructure and services
./run.sh restart

# Show help & environment variables
./run.sh help
```

---

## Manual Step-by-Step Setup

If you prefer starting each service manually:

### 1. Start Infrastructure (Docker)

```bash
docker compose up -d
```

Pull the default chat and embedding models in Ollama:
```bash
docker compose exec -T ollama ollama pull qwen2.5:1.5b
docker compose exec -T ollama ollama pull nomic-embed-text
```

### 2. Start Backend (Spring Boot 4)

```bash
cd backend
mvn spring-boot:run
```

- **Backend API**: `http://localhost:8080`
- **Actuator Health**: `http://localhost:8080/actuator/health`

### 3. Start Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

- **Web UI**: `http://localhost:5173`

---

## Features & Usage

### 1. Fine-Tuning via GUI

1. Open **`http://localhost:5173`**.
2. Under **Fine-Tuning Operations**:
   - **Upload Dataset**: Choose a training dataset file (`.jsonl`, `.json`, `.csv`, `.txt`) and click **Submit File**. The file is stored in `fine-tune/dataset/`.
   - **Configure Model**: Select your dataset file, target model name (e.g. `spring-ai-expert`), and base model (`qwen2.5:1.5b`).
   - **Process**: Click **⚡ Process & Build Fine-Tuned Model**.
3. View the **Success** or **Failed** banner with expandable build logs. Once compiled, the new model is automatically registered in Ollama and selected in the test bench.

### 2. Interactive Test Bench & Clickable Test Prompts

On the same page (**Interactive Test Bench**):
- **Model Switcher**: Select between default (`qwen2.5:1.5b`), fine-tuned models (`spring-ai-expert`), or other installed models.
- **RAG Context Toggle**: Toggle RAG retrieval ON or OFF to test pure fine-tuned weights vs. RAG-augmented knowledge.
- **Clickable Test Prompts**: Click one of the quick test chips:
  - 💬 `What is r4r?`
  - 💬 `What is Spring AI?`
  - 💬 `When should I use RAG instead of fine-tuning?`
  - 💬 `What is the role of Ollama in a Spring AI application?`
  - 💬 `Explain dependency injection in Spring Boot.`
  - 💬 `What is the recommended architecture for an enterprise Spring AI RAG application?`
  - 💬 `Where is company data stored and what is the SLA?`

### 3. Knowledge Base Ingestion (PGVector)

Upload markdown (`.md`), text (`.txt`), JSON (`.json`), or CSV (`.csv`) files from the **Knowledge Base Ingestion** card. Spring AI embeds the content using `nomic-embed-text` and stores it into PostgreSQL PGVector on port `5433`.

Pre-built sample datasets are available in [`data/`](data/):
- [`data/spring-ai-architecture.md`](data/spring-ai-architecture.md)
- [`data/company-faq.json`](data/company-faq.json)
- [`data/services-catalog.csv`](data/services-catalog.csv)
- [`data/company-policies.txt`](data/company-policies.txt)

---

## API Reference

### Chat Endpoint
**`POST /api/chat`**
```json
{
  "message": "What is r4r?",
  "model": "spring-ai-expert",
  "enableRag": true
}
```
Response:
```json
{
  "answer": "Educational articles covering web fundamentals like HTML, CSS, JavaScript, and backend logic with PHP.",
  "modelUsed": "spring-ai-expert",
  "ragEnabled": true
}
```

### Fine-Tuning Endpoints
- **`GET /api/finetune/files`**: Lists dataset files in `fine-tune/dataset/`.
- **`POST /api/finetune/upload`**: Multipart file upload (`file`) to `fine-tune/dataset/`.
- **`POST /api/finetune/process`**: Compiles model into Ollama.
  ```json
  {
    "datasetFile": "spring-ai-1.jsonl",
    "modelName": "spring-ai-expert",
    "baseModel": "qwen2.5:1.5b",
    "temperature": 0.1
  }
  ```
- **`GET /api/finetune/models`**: Lists installed Ollama models available for inference.

### RAG Document Ingestion
- **`POST /api/rag/file`**: Multipart form upload (`file`) to embed and store in PGVector.
- **`POST /api/rag/text`**: Ingest raw text with metadata.
  ```json
  {
    "source": "manual-entry",
    "text": "Enterprise guidelines and instructions..."
  }
  ```

---

## Configuration & Ports

Configured via [`.env`](.env) (or environment variables):

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5433` | Host port mapped to PostgreSQL container |
| `DB_NAME` | `ragdb` | PostgreSQL database name |
| `DB_USER` | `postgres` | PostgreSQL username |
| `DB_PASSWORD` | `postgres` | PostgreSQL password |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama engine API URL |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:1.5b` | Default chat model |
| `OLLAMA_EMBED_MODEL`| `nomic-embed-text` | Dense embedding model |
| `VECTOR_DIMENSIONS` | `768` | Embedding vector dimensions |
