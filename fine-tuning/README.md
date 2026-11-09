# Unsloth QLoRA fine-tuning

This folder adds a small, reproducible fine-tuning stage to the Spring AI RAG demo.

## Pipeline

```text
spring-ai.jsonl
      |
      v
Unsloth + QLoRA
      |
      +---- LoRA adapter
      |
      +---- GGUF q4_k_m
               |
               v
             Ollama
               |
               v
        Spring AI + RAG
```

## Requirements

- Python 3.10+
- The project uses a single shared root virtual environment at the project root (`./.venv`)

Activate the root environment:

```bash
# From project root:
source .venv/bin/activate

# Or from inside fine-tuning/:
source ../.venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets

The dataset folder [`fine-tuning/dataset/`](dataset/) contains ready-to-use instruction datasets:
- **`spring-ai-1000.jsonl`**: Comprehensive 1,000-sample enterprise dataset covering Spring AI, ChatClient, RAG, PGVector, Ollama, LoRA/QLoRA, and Java 21.
- **`spring-ai.jsonl`**: Compact 6-sample baseline dataset.
- **`generate_dataset_1000.py`**: Python script to regenerate or customize the 1,000-sample dataset:
  ```bash
  python generate_dataset_1000.py
  ```

## Train

From this directory:

```bash
python train.py
```

The script starts deliberately small. After evaluating the result, increase the
number and quality of examples, then tune epochs, learning rate, batch size,
and LoRA rank.

## Use the GGUF with Ollama

Copy the generated GGUF into `../ollama/` and edit the model filename in
`../ollama/Modelfile` if necessary.

```bash
cd ../ollama
ollama create spring-ai-expert -f Modelfile
ollama run spring-ai-expert
```

Then configure the Spring Boot application:

```yaml
spring:
  ai:
    ollama:
      chat:
        model: spring-ai-expert
```

The RAG pipeline does not need to change. Fine-tuning controls behavior; RAG
continues to provide retrieved domain knowledge.
