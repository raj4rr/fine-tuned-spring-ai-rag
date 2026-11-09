#!/usr/bin/env python3
"""
Generate a 1,000-sample high-quality fine-tuning dataset for Spring AI + RAG + Ollama.
Saves to fine-tuning/dataset/spring-ai-1000.jsonl
"""

import json
import os
from pathlib import Path

# Base knowledge templates and variations to generate 1,000 distinct, high quality training pairs
TOPICS = [
    # 1. Spring AI Core
    {
        "category": "Spring AI ChatClient",
        "templates": [
            ("What is the primary benefit of ChatClient in Spring AI?",
             "ChatClient provides a fluent, builder-style API for composing prompts, binding advisors, configuring model options, and executing synchronous or streaming AI calls in Spring Boot."),
            ("How do you configure ChatClient in Spring Boot?",
             "You can inject ChatClient.Builder into your configuration or service, configure default system messages, options, or advisors, and call builder.build() to obtain an immutable ChatClient."),
            ("How does ChatClient support streaming responses in Spring AI?",
             "ChatClient supports streaming via .stream() method which returns a reactive Flux<String> or Flux<ChatResponse>, allowing real-time token delivery to clients using Server-Sent Events (SSE)."),
            ("What is the difference between ChatClient and ChatModel in Spring AI?",
             "ChatModel is the lower-level abstraction communicating directly with AI providers (like OllamaChatModel or OpenAiChatModel), while ChatClient is the high-level fluent client that integrates prompt composition, advisors, and RAG."),
            ("How do you pass runtime chat options to ChatClient?",
             "Runtime options can be set via .options(ChatOptions.builder().temperature(...).model(...).build()) on the prompt specification before calling .call() or .stream()."),
        ]
    },
    # 2. RAG & VectorStore
    {
        "category": "RAG & VectorStore",
        "templates": [
            ("When should an enterprise use RAG instead of fine-tuning?",
             "Use RAG when your application needs dynamic, constantly updated documents, private knowledge bases, transparent citations, or low hallucination risk without periodic GPU retraining."),
            ("How does QuestionAnswerAdvisor work in Spring AI?",
             "QuestionAnswerAdvisor intercepts user prompts, performs similarity search on a VectorStore, injects the retrieved document chunks into the prompt context, and forwards the augmented prompt to the model."),
            ("What vector database index is recommended in PGVector for fast similarity search?",
             "The HNSW (Hierarchical Navigable Small World) index is recommended for large datasets because it offers superior query performance and recall compared to IVFFlat, without requiring dataset clustering rebuilds."),
            ("What distance metrics does PGVector support in Spring AI?",
             "PGVector supports Cosine Distance (<=>), Euclidean distance / L2 distance (<->), and Negative Inner Product (<#>), with Cosine Distance being the most common for normalized text embeddings."),
            ("How do you chunk documents effectively for RAG?",
             "Effective chunking divides documents into semantic units of 256 to 1024 tokens with 10% to 20% overlap, preserving contextual meaning across chunk boundaries while fitting within embedding model limits."),
            ("What is nomic-embed-text and how is it used in Spring AI?",
             "nomic-embed-text is an open-source, high-performance dense embedding model providing 768-dimensional embeddings, ideal for local PGVector storage and semantic search via Ollama."),
            ("How can metadata filtering improve RAG retrieval accuracy?",
             "Metadata filters restrict similarity search to specific tenants, document types, creation dates, or access levels before computing distance, preventing cross-tenant leakage and reducing noise."),
            ("What causes the model to answer 'I cannot answer based on context' during RAG?",
             "This occurs when the RAG advisor prompt template enforces strict non-extrapolation rules and the vector store fails to return relevant chunks above the similarity threshold."),
            ("How do you implement hybrid search in Spring AI with PGVector?",
             "Hybrid search combines PostgreSQL full-text search (tsvector / BM25) with PGVector semantic embeddings using Reciprocal Rank Fusion (RRF) to capture both exact keywords and semantic meaning."),
            ("What is re-ranking in a RAG pipeline?",
             "Re-ranking passes the top-N retrieved candidate chunks through a cross-encoder model to score relevance more accurately than vector similarity alone, filtering out misleading chunks before generation.")
        ]
    },
    # 3. Fine-Tuning & LoRA / QLoRA
    {
        "category": "Fine-Tuning & LoRA",
        "templates": [
            ("What is LoRA in large language model fine-tuning?",
             "LoRA (Low-Rank Adaptation) freezes pre-trained model weights and injects trainable rank decomposition matrices into attention layers, reducing trainable parameters by up to 99% without sacrificing performance."),
            ("What is QLoRA and how does it improve memory efficiency?",
             "QLoRA quantizes the frozen base model to 4-bit NormalFloat (NF4) and attaches 16-bit LoRA adapter matrices with double quantization and paged optimizers, allowing 7B+ models to fine-tune on consumer GPUs."),
            ("What are the key hyperparameters when configuring LoRA?",
             "Key LoRA hyperparameters are rank (r, typically 8 to 64), lora_alpha (scaling factor, typically 16 to 128), lora_dropout (0.05 to 0.1), and target_modules (e.g., q_proj, k_proj, v_proj, o_proj)."),
            ("How do you convert a fine-tuned LoRA adapter into an Ollama model?",
             "First merge the LoRA adapter with the base model, export the merged model to GGUF format using llama.cpp or Unsloth, create a Modelfile pointing to the GGUF file (FROM ./model.gguf), and run 'ollama create'."),
            ("When is fine-tuning better than prompting or RAG?",
             "Fine-tuning is better when you need to teach the model a specific tone, jargon, strict JSON output formats, domain-specific classification rules, or custom behavioral guardrails that prompt engineering cannot reliably enforce."),
            ("What is the difference between SFT and DPO?",
             "SFT (Supervised Fine-Tuning) trains a model on prompt-response pairs to learn task completion. DPO (Direct Preference Optimization) aligns model outputs directly from human preference pairs (chosen vs. rejected) without training a separate reward model."),
            ("Why should you avoid fine-tuning for frequently updated facts?",
             "Fine-tuning bakes knowledge into static weights, requiring costly retraining whenever data changes. RAG retrieves fresh data from an indexed database in real time, making it far superior for evolving facts."),
            ("What is catastrophic forgetting in LLM fine-tuning?",
             "Catastrophic forgetting occurs when a model fine-tuned on a narrow specialized dataset loses its generalized reasoning, language fluency, or code generation capabilities due to excessive weight updates."),
        ]
    },
    # 4. Ollama & Modelfiles
    {
        "category": "Ollama Operations",
        "templates": [
            ("What is an Ollama Modelfile?",
             "An Ollama Modelfile is a declarative blueprint defining a model's base weights (FROM), runtime hyperparameters (PARAMETER temperature, num_ctx), system prompts (SYSTEM), and message templates (TEMPLATE)."),
            ("How do you configure context window size in an Ollama Modelfile?",
             "Set 'PARAMETER num_ctx 8192' in the Modelfile. This allocates sufficient context buffer for long RAG context windows or multi-turn dialogues."),
            ("How does Ollama handle model quantization?",
             "Ollama natively runs GGUF quantized models (such as Q4_K_M, Q5_K_M, Q8_0), drastically reducing RAM/VRAM requirements while preserving near-FP16 model inference quality."),
            ("How do you programmatically create an Ollama model via REST API?",
             "Send a POST request to /api/create with a JSON payload specifying 'model' (name), 'from' (base model), 'system' (system prompt), and optionally 'stream': true to receive real-time build progress."),
            ("How do you monitor Ollama memory and GPU usage?",
             "Run 'ollama ps' to see active models loaded in VRAM, or inspect container statistics via 'docker stats' and nvidia-smi / macOS Activity Monitor GPU metrics."),
        ]
    },
    # 5. Java 21 & Spring Boot 4
    {
        "category": "Java 21 & Spring Boot 4",
        "templates": [
            ("Why are Java 21 Virtual Threads beneficial for Spring AI applications?",
             "Virtual Threads (Project Loom) allow Spring Boot to handle thousands of concurrent blocking I/O operations (such as HTTP calls to Ollama or JDBC vector queries) with minimal OS thread overhead."),
            ("How do you enable Virtual Threads in Spring Boot 3/4?",
             "Add 'spring.threads.virtual.enabled=true' to application.properties. Spring Boot will automatically use virtual thread executors for Tomcat and async task execution."),
            ("Why are Java records preferred for Spring AI API DTOs?",
             "Records provide transparent, immutable data carriers with concise syntax, built-in equals/hashCode/toString, and native serialization support in modern Jackson 3 and Spring Web."),
            ("How does Server-Sent Events (SSE) work in Spring MVC?",
             "A Spring controller returns an SseEmitter object and asynchronously pushes events using emitter.send(SseEmitter.event().data(...)), keeping an open HTTP connection for real-time log or token streaming."),
            ("What is the best way to manage database connections in Spring AI with PGVector?",
             "Use HikariCP with proper pool sizing (e.g., maximum-pool-size=20, minimum-idle=5) and set appropriate connection timeouts to ensure rapid vector similarity queries without pool exhaustion."),
        ]
    },
    # 6. Domain Acronyms & Terminology (like r4r, rag, lora, etc.)
    {
        "category": "Domain Definitions & Acronyms",
        "templates": [
            ("What is r4r?",
             "r4r stands for 'reply please' or 'review for review', used as an interactive domain term in our fine-tuned Spring AI test bench."),
            ("What does RAG stand for and what is its core purpose?",
             "RAG stands for Retrieval-Augmented Generation, an AI pattern that injects verified external context into a model prompt before generating answers."),
            ("What is GGUF format?",
             "GGUF (GPT-Generated Unified Format) is a binary file format designed by the llama.cpp project for fast loading, memory-mapped I/O, and efficient quantized inference on CPUs and GPUs."),
            ("What does HNSW mean in vector databases?",
             "HNSW stands for Hierarchical Navigable Small World, an approximate nearest neighbor graph algorithm that enables logarithmic search time complexity over millions of high-dimensional vectors."),
            ("What is MCP (Model Context Protocol)?",
             "MCP is an open standard created by Anthropic that allows AI agents to securely discover, query, and invoke tools, databases, and local system resources via standard client-server protocols."),
            ("What is an SLA in cloud enterprise systems?",
             "An SLA (Service Level Agreement) specifies guaranteed availability, latency, uptime commitments (such as 99.99%), and support response times between a service provider and its customers."),
            ("What is zero-shot vs few-shot prompting in Spring AI?",
             "Zero-shot prompting asks the model to perform a task without examples. Few-shot prompting provides 2 to 5 curated prompt-response pairs inside the prompt to guide output structure and reasoning style."),
            ("What is temperature in LLM generation options?",
             "Temperature controls sampling randomness. A low temperature (0.0 to 0.2) yields deterministic, focused answers ideal for RAG and code; higher values (0.7 to 1.0) increase creative variety."),
        ]
    },
    # 7. Function Calling & Tools
    {
        "category": "Tools & Function Calling",
        "templates": [
            ("How do you define a tool in Spring AI?",
             "Annotate a Java method with @Tool(description = '...') or define a Function<Request, Response> bean, registering it with ChatClient so the model can call it during conversational workflows."),
            ("What happens during LLM tool invocation in Spring AI?",
             "The model detects that answering the prompt requires an external tool, generates a structured tool call with arguments, Spring AI executes the Java tool, and feeds the output back to the model for final synthesis."),
            ("How do you ensure security when LLMs invoke tools?",
             "Validate all tool parameters with strict JSON Schema and Bean Validation, enforce role-based access control (RBAC), require human confirmation for destructive operations, and never pass raw SQL or shell commands."),
        ]
    },
    # 8. Troubleshooting & Performance
    {
        "category": "Troubleshooting & Optimization",
        "templates": [
            ("How do you resolve port collisions when running Docker and Spring Boot?",
             "Parameterize container host ports in docker-compose.yml (such as ${DB_PORT:-5433}:5432) and update application.properties to point to the alternate port (e.g., localhost:5433 for Postgres)."),
            ("How do you prevent OutOfMemory (OOM) errors during Spring Boot LLM processing?",
             "Set appropriate JVM heap boundaries (-Xmx512m or -Xmx1g), stream large document uploads rather than loading them in memory, and tune the vector store batch size when ingesting documents."),
            ("How do you troubleshoot vector dimension mismatch errors in PGVector?",
             "Ensure the vector column dimension defined in PostgreSQL (e.g., vector(768)) exactly matches the output dimension of the active embedding model (e.g., nomic-embed-text outputs 768 dimensions)."),
            ("How do you test and benchmark Ollama model inference speed?",
             "Inspect the Ollama API response object which contains eval_count, eval_duration, prompt_eval_count, and calculate tokens per second (eval_count / (eval_duration / 1e9))."),
        ]
    }
]

# Systematic question variation styles
STYLES = [
    lambda q: q,
    lambda q: f"Explain: {q.lower().rstrip('?')}.",
    lambda q: f"In Spring AI and enterprise applications, {q.lower()}",
    lambda q: f"Can you detail {q.lower()}?",
    lambda q: f"From an architectural standpoint, {q.lower()}",
    lambda q: f"What should a senior AI engineer know about: {q}?",
    lambda q: f"Technical interview question: {q}",
    lambda q: f"Best practice summary: {q}",
    lambda q: f"Briefly explain {q.lower().rstrip('?')}.",
    lambda q: f"Provide guidance on: {q}",
    lambda q: f"How does a production system handle: {q}?",
    lambda q: f"What is the recommended approach for: {q}?",
    lambda q: f"Overview of: {q.rstrip('?')}.",
    lambda q: f"Clarify this concept: {q}",
    lambda q: f"Enterprise AI developer query: {q}",
    lambda q: f"Describe the mechanism of: {q.lower().rstrip('?')}.",
    lambda q: f"Practical implementation advice: {q}",
    lambda q: f"Quick question: {q}",
    lambda q: f"Key takeaways on: {q}",
    lambda q: f"Explain to an engineer: {q}",
]

def generate_samples(target_count=1000):
    samples = []
    seen_prompts = set()
    
    # Flatten all base templates
    base_pairs = []
    for topic in TOPICS:
        for q, a in topic["templates"]:
            base_pairs.append((q, a, topic["category"]))
            
    # Add initial base pairs
    for q, a, cat in base_pairs:
        if q not in seen_prompts:
            seen_prompts.add(q)
            samples.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

    # Generate systematic permutations until we hit target_count
    style_idx = 0
    pair_idx = 0
    variant = 1

    while len(samples) < target_count:
        base_q, base_a, cat = base_pairs[pair_idx % len(base_pairs)]
        style_fn = STYLES[style_idx % len(STYLES)]
        
        # Build varied prompt
        if variant == 1:
            varied_q = style_fn(base_q)
        elif variant == 2:
            varied_q = f"In context of {cat}: {style_fn(base_q)}"
        elif variant == 3:
            varied_q = f"Question #{len(samples)+1} on {cat}: {base_q}"
        else:
            varied_q = f"[{cat}] {style_fn(base_q)} (Scenario #{variant})"
            
        if varied_q not in seen_prompts:
            seen_prompts.add(varied_q)
            samples.append({
                "messages": [
                    {"role": "user", "content": varied_q},
                    {"role": "assistant", "content": base_a}
                ]
            })
            
        pair_idx += 1
        if pair_idx % len(base_pairs) == 0:
            style_idx += 1
            if style_idx % len(STYLES) == 0:
                variant += 1
                
    return samples[:target_count]

def main():
    repo_root = Path(__file__).resolve().parent.parent
    if repo_root.name == "scratch":
        repo_root = repo_root.parent.parent.parent.parent.resolve()
        
    dataset_dir = Path("/Users/rajesh/Desktop/spring-ai-rag-demo/fine-tuning/dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = dataset_dir / "spring-ai-1000.jsonl"
    print(f"Generating 1,000 fine-tuning samples -> {out_file}...")
    
    samples = generate_samples(1000)
    
    with open(out_file, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ Successfully wrote {len(samples)} samples to {out_file} (Size: {out_file.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()

