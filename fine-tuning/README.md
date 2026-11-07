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

- NVIDIA GPU with a compatible CUDA/PyTorch installation
- Python 3.10+
- A CUDA-compatible PyTorch build
- Unsloth, TRL, Transformers, Datasets

Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install the PyTorch build appropriate for your CUDA version first.
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
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
