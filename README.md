# Spring AI + Ollama + RAG + React

Production-oriented starter implementing:

React UI → Spring Boot 4 → Spring AI 2.0.2 → Ollama → RAG → PostgreSQL/PGVector

## Requirements
- JDK 21+
- Maven 3.9+
- Docker + Docker Compose
- Node.js 20+

## 1. Start infrastructure

```bash
docker compose up -d

docker exec -it spring-ai-rag-demo-ollama-1 ollama pull qwen2.5:7b
docker exec -it spring-ai-rag-demo-ollama-1 ollama pull nomic-embed-text
```

Container names can differ; use `docker compose exec ollama ollama pull ...` if preferred.

## 2. Start backend

```bash
cd backend
mvn spring-boot:run
```

Backend: http://localhost:8080
Health: http://localhost:8080/actuator/health

## 3. Start React

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally http://localhost:5173.

## 4. Use RAG

Upload a `.txt`, `.md`, `.json` or `.csv` file from the UI. Spring AI creates embeddings through Ollama and stores them in PGVector.

Then ask questions. `QuestionAnswerAdvisor` performs vector similarity retrieval and injects the retrieved context into the LLM prompt.

## API

### Chat
POST `/api/chat`
```json
{"message":"What does our knowledge base say about Spring AI?"}
```

### Index text
POST `/api/rag/text`
```json
{"source":"spring-ai-notes","text":"Spring AI integrates LLMs, vector stores and tools into Spring applications."}
```

### Upload
POST `/api/rag/file` multipart field `file`.

## Configuration

The backend defaults to:
- Chat model: `qwen2.5:7b`
- Embedding model: `nomic-embed-text`
- Ollama: `http://localhost:11434`
- PostgreSQL: `localhost:5432/ragdb`
- PGVector dimensions: `768`

If your embedding model produces a different vector size, change `VECTOR_DIMENSIONS` to match it before initializing the schema.

## Fine-tuned model

After Unsloth produces a GGUF model, load it into Ollama and change:

```yaml
spring:
  ai:
    ollama:
      chat:
        model: my-finetuned-model
```

The RAG pipeline remains unchanged. This is the intended separation:

**Fine-tuning = behavior**

**RAG = current/domain knowledge**

**Spring AI = orchestration**

**Ollama = local inference**

## 5. Add Unsloth fine-tuning

The project now includes `fine-tuning/` with a QLoRA training script and a small
Spring AI-oriented instruction dataset.

```bash
cd fine-tuning
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install the PyTorch build matching your NVIDIA CUDA environment first.
pip install -r requirements.txt
python train.py
```

The training script produces a LoRA adapter and a GGUF quantized model. Copy the
GGUF to `ollama/` and create the local Ollama model:

```bash
cd ../ollama
ollama create spring-ai-expert -f Modelfile
ollama run spring-ai-expert
```

Then set:

```bash
export OLLAMA_CHAT_MODEL=spring-ai-expert
```

On Windows PowerShell:

```powershell
$env:OLLAMA_CHAT_MODEL="spring-ai-expert"
```

Restart Spring Boot. React continues to call `/api/chat`; the backend now uses
the fine-tuned Ollama model while RAG continues to retrieve context from PGVector.

### Final architecture

```text
React
  |
  v
Spring Boot 4 + Spring AI
  |              |
  |              +---- PGVector RAG ----> domain documents
  |
  +---- Ollama ----> fine-tuned QLoRA/GGUF model
```

**Fine-tuning = behavior**  
**RAG = knowledge**  
**Ollama = local inference**  
**Spring AI = orchestration**  
**Spring Boot = security + business logic**
