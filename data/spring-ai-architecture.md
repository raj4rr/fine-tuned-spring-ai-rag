# Spring AI & Enterprise Architecture Knowledge Base

## Overview
Spring AI brings portable and declarative AI capabilities to the Spring Boot ecosystem. It allows Java developers to integrate Large Language Models (LLMs), embedding models, vector databases, and Retrieval-Augmented Generation (RAG) pipelines using idiomatic Spring abstractions like `ChatClient`, `VectorStore`, and `Advisor`.

## Key Architectural Principles

1. **Separation of Concerns: Fine-tuning vs. RAG**
   - **Fine-tuning (Behavior & Tone)**: Adapts the model's style, reasoning pattern, formatting output (JSON, specialized syntax), or domain-specific terminology.
   - **RAG (Dynamic & Grounded Knowledge)**: Injects up-to-date, proprietary company documents into the prompt at query time to prevent hallucinations and provide citations.

2. **Spring AI ChatClient & Advisors**
   - `ChatClient.Builder`: Fluent API to build chat prompts with system instructions, user prompts, parameters, and advisors.
   - `QuestionAnswerAdvisor`: Intercepts the user query, retrieves relevant document embeddings from PGVector via cosine or cosine distance similarity search, and injects context into the prompt before sending it to the model.

3. **Ollama Integration**
   - Ollama provides low-latency local LLM inference without sending sensitive data to external APIs.
   - Default configuration uses `qwen2.5:1.5b` for fast, lightweight responses and `nomic-embed-text` (768-dimensional embeddings) for dense semantic retrieval.

4. **Vector Store: PGVector on PostgreSQL 17**
   - Document chunks are stored with 768-dimensional vectors.
   - Indexing uses HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor lookups.
   - Cosine distance (`COSINE_DISTANCE`) calculates semantic similarity between the user prompt and stored knowledge.

5. **Security and Enterprise Governance**
   - Keep business logic, access control, audit logging, and authorization within Spring Boot services.
   - Never delegate authorization or critical data mutation directly to unvetted LLM tool calls.

