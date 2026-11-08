This error means your Java application cannot find the python executable in its default system environment path. When running microservices or deep learning tasks, Java's ProcessBuilder does not automatically load your shell profiles (like .bashrc, .zshrc, or Conda environments).
To fix this, you need to tell Java exactly where your Python executable lives (especially if it is inside a virtual environment or an Anaconda path where Unsloth is installed).
Here is how to fix your Java code and locate your path:

1. Find your Exact Python Executable Path

Open your terminal where you successfully run your Unsloth scripts and execute one of these commands depending on your operating system:
• Linux / macOS: Run which python or which python3
• Windows (PowerShell): Run where.exe python
This will give you a direct path, for example:
• Standard Linux: /usr/bin/python3
• Conda/Venv: /home/user/miniconda3/envs/unsloth_env/bin/python
• Windows: C:\Users\Username\miniconda3\envs\unsloth\python.exe

2. Update the Java Code with the Full Path

Modify your ProcessBuilder in FineTuningService.java to use that explicit path instead of the generic "python" string.
java
// Replace "/home/user/miniconda3/envs/unsloth_env/bin/python" with the path you found in Step 1
String pythonExecutable = "/home/user/miniconda3/envs/unsloth_env/bin/python";

ProcessBuilder pb = new ProcessBuilder(
pythonExecutable, "train.py",
"--dataset", datasetPath.toAbsolutePath().toString(),
"--output", outputPath.toAbsolutePath().toString(),
"--gguf", ggufPath.toAbsolutePath().toString()
);
Use code with caution.

3. Alternative: Pass the Path via application.properties (Cleanest approach)

Instead of hardcoding the environment location inside your Java class, abstract it into your Spring configuration files so it adapts to different machines easily.
Add this to your src/main/resources/application.properties:
properties
# Path to the specific python environment where unsloth is installed
python.executable.path=/home/user/miniconda3/envs/unsloth_env/bin/python
Use code with caution.
Then update your FineTuningService.java:
java
@Value("${python.executable.path:python}")
private String pythonExecutable;

public FineTuneResult processFineTuning(FineTuneRequest request) {
// ... setup paths ...

    ProcessBuilder pb = new ProcessBuilder(
        this.pythonExecutable, "train.py",
        "--dataset", datasetPath.toAbsolutePath().toString(),
        "--output", outputPath.toAbsolutePath().toString(),
        "--gguf", ggufPath.toAbsolutePath().toString()
    );
    
    // Optional: If your script relies on specific environment variables, 
    // you can pass them to the process environment directly here:
    // pb.environment().put("CUDA_VISIBLE_DEVICES", "0");

    // ... rest of execution code ...
}
Use code with caution.
Once you explicitly map the executable location, ProcessBuilder will instantly initialize the Python runtime, hook into your CUDA cards, and kick off the real training process.
Is your python environment running on the same local machine as the Java backend, or are you running Java inside a Docker container? (If Java is containerised, it won't see your host machine's Python environment without volume mapping or API bridging).


You are running your terminal inside a virtual environment (.venv), but your Java program is executing a global python environment that cannot see it.
Additionally, there is a fundamental hardware problem hidden in your log path: You are running on a Mac (/Users/rajesh/...), but your script explicitly attempts to load unsloth, which requires an NVIDIA CUDA GPU and does not work on macOS.
Here is the exact step-by-step plan to fix both the path issue and the hardware compatibility block:

Step 1: Fix the Python Path in Java

Because you created the virtual environment inside the spring-ai-rag-demo folder, your exact Python executable path is inside that folder.
Update your src/main/resources/application.properties file to point directly to the binary inside that folder:
properties
# Point Java directly to the virtual environment binary instead of the system default
python.executable.path=${user.dir}/.venv/bin/python
Use code with caution.
(If ${user.dir} doesn't resolve properly in your setup, swap it with the absolute path: /Users/rajesh/Desktop/spring-ai-rag-demo/.venv/bin/python).

Step 2: Fix the macOS Architecture Block (Crucial)

Unsloth uses hard-coded NVIDIA CUDA operations that fail on Apple Silicon or Intel Macs. If you try to run your current train.py script on your Mac, it will throw a crash error immediately after finding the module.
To get the exact same behavior (True QLoRA training saving to GGUF for Ollama) natively on a Mac, you must swap Unsloth out for standard Hugging Face PEFT, which natively supports Apple Metal (MPS) acceleration.

1. Update your requirements.txt

Change your dependencies to use the standard, platform-agnostic packages:
text
transformers
peft
trl
datasets
accelerate
mlx-lm
Use code with caution.
Run pip install -r requirements.txt inside your activated terminal loop to update.

2. Update your train.py Script

Replace the Unsloth code with this platform-agnostic fallback script. It automatically detects if you are on a Mac (mps) or a CUDA machine and performs the exact same QLoRA tuning:
python
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--gguf", type=str, required=True)
args = parser.parse_args()

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 1. Device Selection (Auto-detect Mac Metal framework or CUDA)
if torch.backends.mps.is_available():
device = "mps"
# Mac MPS doesn't support 4-bit bitsandbytes quantization natively well yet;
# We load in bfloat16/float16 for matrix operations on Apple Silicon
compute_dtype = torch.bfloat16
quantization_config = None
else:
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_dtype = torch.float16
quantization_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=compute_dtype
)

print(f"Targeting compute device: {device.upper()}")

# 2. Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
MODEL_NAME,
quantization_config=quantization_config,
torch_dtype=compute_dtype,
device_map=device if device != "mps" else None
)

if device == "mps":
model = model.to(device)

# 3. Setup standard LoRA Adapters
if quantization_config is not None:
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
r=16,
lora_alpha=16,
target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
lora_dropout=0.05,
bias="none",
task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 4. Load Dataset & Format Chat Templates
dataset = load_dataset("json", data_files=args.dataset, split="train")

def format_example(example):
return tokenizer.apply_chat_template(
example["messages"], tokenize=False, add_generation_prompt=False
)

# 5. Execute Training
trainer = SFTTrainer(
model=model,
tokenizer=tokenizer,
train_dataset=dataset,
formatting_func=format_example,
max_seq_length=2048,
args=SFTConfig(
output_dir=args.output,
num_train_epochs=2,
per_device_train_batch_size=1,
gradient_accumulation_steps=4,
learning_rate=2e-4,
logging_steps=1,
report_to="none"
),
)
trainer.train()

# 6. Save Adapter Output
model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)

print(f"LoRA training completed successfully on {device.upper()}.")
Use code with caution.

Step 3: Exporting to GGUF on Mac

Because Unsloth's direct native .save_pretrained_gguf() function is tightly bound to CUDA hardware platforms, the easiest way to generate a GGUF format on a Mac once the Python script saves the adapter files is to use llama.cpp directly in your terminal loop or trigger it from Java.
Alternatively, you can skip local translation completely and push your generated adapters straight to Ollama via your existing Java structure using standard weights.
Are you developing on this Mac laptop but intending to deploy the backend to a Linux server with an NVIDIA GPU later? If so, I can show you how to structure a simple profile switch (dev for Mac, prod for Linux/Unsloth) so your code runs correctly in both locations.


The installation failed because your Mac has completely run out of disk space ([Errno 28] No space left on device).
Deep learning packages like PyTorch (torch), Transformers, and the model weights themselves take up a massive amount of storage space (often tens of gigabytes). The system crashed right as it was compiling and installing mlx-metal.
Here is a quick, structured checklist to free up space on your Mac so you can complete the installation:

1. Clear Your Package Manager Caches (Fastest Win)

When pip downloads packages, it caches the .whl files on your drive. You can instantly free up space by wiping this cache:
bash
pip cache purge
Use code with caution.
If you use Homebrew, clean up old versions and download caches:
bash
brew cleanup -s
Use code with caution.

2. Delete Cached Hugging Face Models

If you have previously downloaded large models (like Llama, Qwen, or Mistral) through Python scripts or UI interfaces, they are taking up space in a hidden cache directory. Inspect and clear this folder:
bash
rm -rf ~/.cache/huggingface/hub/*
Use code with caution.
(Note: This only deletes downloaded models; your source code remains safe).

3. Clear Ollama's Cached Models

Since your Java service integrates with Ollama, you might have duplicate base models taking up 5–20 GB each. Check what you have installed:
bash
ollama list
Use code with caution.
Remove any models you aren't currently using for this specific project:
bash
ollama rm <model_name>
Use code with caution.

4. Delete Temporary Virtual Environments

If you have older, abandoned virtual environments inside your code workspaces, delete them to free up space. A single machine learning virtual environment can easily exceed 5 GB:
bash
# Double-check you are in the correct directory before running rm -rf!
rm -rf /path/to/old/project/.venv
Use code with caution.

Verification

Once you have cleared out some room, check your remaining storage from the terminal:
bash
df -h
Use code with caution.
Look at the Avail column next to your main hard drive partition (/dev/disk3s1 or similar). You will need at least 15–20 GB of free space to comfortably install PyTorch, downscale the dependencies, and allow room for your training run.
Once space is clear, activate your environment and retry:
bash
source .venv/bin/activate
pip install -r requirements.txt
Use code with caution.
Let me know if you are still running short on space after clearing these folders! I can show you how to target your training script to use smaller baseline models (like Qwen2.5-0.5B) to keep the disk footprint as light as possible.


Your terminal output shows that your main storage partition (/System/Volumes/Data) has only 5.4 GiB of available space left (98% full). A standard PyTorch machine learning environment requires at least 8–10 GB just to install, and downloading the Qwen2.5-7B model weights requires another 15 GB. You do not have enough physical disk space to install or run the full Python training pipeline on this machine right now.
Since you cannot run the Python QLoRA pipeline locally under these storage limits, you have two realistic paths forward:

Path 1: Revert to the Java Ollama Workaround (Zero Extra Disk Space)

If you cannot free up 25+ GB on your Mac, you should go back to the original FineTuningService.java prompt-hacking solution.
• Why it works now: Ollama uses your already existing local model files and doesn't require installing massive Python dependencies like torch or transformers.
• Fixing the Truncated Code: Since your original Java file snippet was cut off at the bottom, here is the complete, working /api/create implementation to finish your FineTuningService.java:
java
// ... (Keep the rest of your original FineTuningService code up to building the systemPrompt)

    systemPrompt.append("### INSTRUCTION GUIDELINES:\n");
    systemPrompt.append("- Always prioritize and use the learned domain knowledge and definitions above when answering user questions.\n");
    systemPrompt.append("- Provide concise, accurate, production-grade answers.\n");

    // Save generated Modelfile locally
    try {
        Path modelOutputDir = getOutputDirectory().resolve(targetModel);
        Files.createDirectories(modelOutputDir);
        Path modelfile = modelOutputDir.resolve("Modelfile");

        String modelfileContent = String.format(
            "FROM %s\nPARAMETER temperature %.2f\nPARAMETER num_ctx 8192\nSYSTEM \"\"\"\n%s\n\"\"\"\n",
            baseModel, temperature, systemPrompt.toString().replace("\"\"\"", "\\\"\\\"\\\"")
        );
        Files.writeString(modelfile, modelfileContent, StandardCharsets.UTF_8);
        logs.append(String.format("Generated Modelfile saved to %s\n", modelfile.toAbsolutePath()));
    } catch (Exception e) {
        logs.append(String.format("Warning: could not save local Modelfile: %s\n", e.getMessage()));
    }

    // Call Ollama API to compile and register the model
    logs.append(String.format("Invoking Ollama API at %s/api/create...\n", ollamaBaseUrl));
    try {
        Map<String, Object> payload = new HashMap<>();
        payload.put("model", targetModel);
        payload.put("from", baseModel);
        payload.put("system", systemPrompt.toString());
        payload.put("parameters", Map.of("temperature", temperature));
        payload.put("stream", false);

        String jsonPayload = objectMapper.writeValueAsString(payload);

        HttpRequest httpReq = HttpRequest.newBuilder()
            .uri(URI.create(ollamaBaseUrl + "/api/create"))
            .header("Content-Type", "application/json")
            .timeout(Duration.ofMinutes(5))
            .POST(HttpRequest.BodyPublishers.ofString(jsonPayload, StandardCharsets.UTF_8))
            .build();

        HttpResponse<String> response = httpClient.send(httpReq, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() >= 200 && response.statusCode() < 300) {
            logs.append(String.format("Model '%s' registered in Ollama successfully.\n", targetModel));
            return new FineTuneResult(true, targetModel, "System Prompt customization complete!", logs.toString(), sampleCount);
        } else {
            return new FineTuneResult(false, targetModel, "Ollama API error: " + response.statusCode(), logs.toString(), sampleCount);
        }
    } catch (Exception e) {
        return new FineTuneResult(false, targetModel, "Failed to connect to Ollama: " + e.getMessage(), logs.toString(), sampleCount);
    }
}
Use code with caution.

Path 2: Free Up Disk Space for the Python Pipeline

If you are determined to perform true QLoRA weight training, you must clear at least 25 GB of space. Based on your storage profile, look into these locations to free up space immediately:
• Delete hidden Docker caches: Run docker system prune -a --volumes (can free up 10–40 GB instantly if you use Docker).
• Clear Xcode Developer storage: Delete the contents of ~/Library/Developer/Xcode/DerivedData and old iOS simulators.
• Empty Trash via Terminal: Sometimes the UI fails to clear it completely: rm -rf ~/.Trash/*
Which path makes the most sense for your project right now? If you want to clear space for the Python setup, I can give you precise commands to hunt down the largest files on your Mac; otherwise, we can focus on getting the Java + Ollama system prompt workaround fully operational.



I have read the complete `fine-tune-lora.md` (329 lines). It contains a sequence of troubleshooting steps and implementation approaches for integrating Java Spring Boot + Python + LoRA/QLoRA + Ollama.

I will reorganize it into a practical, step-by-step implementation guide, explain what each component does, how the components communicate, what went wrong in your Mac environment, and how to run the project.

There is one important distinction we need to understand first:

Creating a customized Ollama model is not the same as actually fine-tuning its neural network weights.

Your document describes both approaches. We should understand them separately before implementing them.

# 1. Understand the complete architecture

Your intended application is:

- Backend: Java 21 + Spring Boot
- AI model: Qwen 2.5 7B Instruct
- Model runtime: Ollama
- Training framework: Python + Hugging Face + PEFT/LoRA
- Training orchestration: Java `ProcessBuilder`
- Output: trained adapter or GGUF model
- Frontend: your existing React/Spring Boot application

[Augmenting LLMs: Fine-Tuning or RAG?](https://images.openai.com/static-rsc-4/dX_Vvk2EiWtoi9c5DTR-ThJflzZc_6YtzoOkb14IFqy4BwpbtQ53Wh4edhjwmApRKyRlde3bpMeljj3Cq_QB6-TyP7ZHvRqHi5XL5AoawrR_OKmGb8JIWwy6CGTgrA-nMXF8mXs4qhIF6eq-wZj3axjv1OoQfe0G07IHtmaQCQg?purpose=inline)

[dailydoseofds.com](https://www.dailydoseofds.com/augmenting-llms-fine-tuning-or-rag/?utm_source=chatgpt.com)

The overall workflow:

```
              React Frontend
                    |
                    v
             Spring Boot API
                    |
           FineTuningService.java
                    |
             ProcessBuilder
                    |
                    v
              Python train.py
                    |
             Hugging Face
                    |
              Base Qwen Model
                    |
                LoRA
                    |
              Training Dataset
                    |
                    v
             Trained LoRA Adapter
                    |
              Model Export
                    |
                    v
                  Ollama
                    |
                    v
             Customized AI Model
```

There are two possible ways to customize the model:

| Approach                    | What happens                                                          |
| --------------------------- | --------------------------------------------------------------------- |
| System prompt customization | Changes model instructions and behavior without training weights      |
| LoRA fine-tuning            | Trains additional model parameters (LoRA adapters) using your dataset |

Your document initially attempts actual LoRA training and later suggests a system-prompt workaround because of disk and hardware limitations.&#x20;

fine-tune-lora.md



# 2. Understand LoRA and QLoRA first

Before implementing the code, understand what happens inside the model.

[How Transformer LLMs Generate Text: One Token at a Time | by alex buzunov | CodeX | Medium](https://images.openai.com/static-rsc-4/lJj5gV5bdveaeulut26-trcZIAevb3fupjSD2cNqyvrf9FMx0iGscsMCbupaqE7Clyl5aO_Cp3olJQT1hreBBv_9JqCpCtLmm_2j53PQFkCy43iIId-v_BTX7sBVN0mAioRt4SY5fgfiNfqlsJlYX_YVvvfc-KDbmooS8ikogio?purpose=inline)

Base model

Qwen 2.5 7B is a pretrained language model containing billions of parameters.

[Deeply Learning 4: LoRA (Low Rank Adaptation) - Math, Intuition and Implementation from Scratch | Deeply Learning](https://images.openai.com/static-rsc-4/ri-RN8ouizlMOLwPcP3-8SURVawCa7lrso6qeuYhozt7lwIsYMWLCoq-h1dgIlRZ6BpZKqT9Tg7OvwSUf34fWBg1g-X-9vW3WJ12ZPu2hdrKeDR99hPpZnYx-knZsXcUHVMaJ6uSgOTNoFFFgRB_2eRx9BcSVvZ3wVH0-6G1WKs?purpose=inline)

LoRA

Instead of updating all the original model weights, we train smaller additional matrices called adapters.

[Understanding QLoRA: Quantized Fine-Tuning | AI Tutorial | Next Electronics](https://images.openai.com/static-rsc-4/yJ6gxdQGsW2GgMigQkpkloJ5IyCoa8w0weaN8yj0jjW-w5uTGDNL6uGbTPd-afjzlwrOmN7yFf2n30LNBbKZ_GGv7XZIiVYiccY7RK_GaOs6Yneo7FYg1fC11RIG8g0mCKpMRVXd_JeUX3xsoo0UoMKui-bYLeQPbC2N0M1OTYw?purpose=inline)

QLoRA

Keeps the base model quantized, typically to 4 bits, while training LoRA adapters.

## How LoRA works

Suppose the original neural network has a weight matrix:

\\[ W \\]

Traditional fine-tuning updates the original matrix:

\\[ W' = W + \Delta W \\]

LoRA represents the weight update using two smaller matrices:

\\[ \Delta W = B \times A \\]

Therefore:

\\[ W' = W + BA \\]

Where:

- W = original pretrained weights.
- A = low-rank matrix.
- B = low-rank matrix.
- BA = learned weight update.

The original model remains mostly unchanged. We train the adapter matrices.

Why is this useful?

Instead of training billions of parameters, you train a much smaller number of additional parameters.

For example, LoRA configuration:

```
LoraConfig(    r=16,    lora_alpha=16,    target_modules=[        "q_proj",        "v_proj",        "k_proj",        "o_proj"    ],    lora_dropout=0.05,    bias="none",    task_type="CAUSAL_LM")
```

| Parameter         | Meaning                               |
| ----------------- | ------------------------------------- |
| r=16              | Rank of the adapter matrices          |
| lora_alpha=16     | Scaling factor                        |
| target_modules    | Transformer layers receiving adapters |
| lora_dropout=0.05 | Regularization                        |
| bias=none         | Do not train bias parameters          |
| CAUSAL_LM         | Autoregressive language modeling      |

# 3. Step-by-step implementation

Let's recreate the implementation in the same sequence as your document.

## Step 1: Prepare the project structure

Your project should have a structure similar to this:

```
spring-ai-rag-demo/
│
├── src/
│   └── main/
│       ├── java/
│       │   └── com/example/
│       │       ├── controller/
│       │       │   └── FineTuningController.java
│       │       │
│       │       ├── service/
│       │       │   └── FineTuningService.java
│       │       │
│       │       └── dto/
│       │           ├── FineTuneRequest.java
│       │           └── FineTuneResult.java
│       │
│       └── resources/
│           └── application.properties
│
├── python/
│   ├── train.py
│   └── requirements.txt
│
├── datasets/
│   └── training.json
│
├── models/
│
└── pom.xml
```

Responsibilities:

| Component            | Responsibility                   |
| -------------------- | -------------------------------- |
| FineTuningController | Accepts frontend requests        |
| FineTuningService    | Coordinates training             |
| ProcessBuilder       | Starts Python                    |
| train.py             | Performs LoRA training           |
| training.json        | Contains training examples       |
| models               | Stores generated adapters/models |
| Ollama               | Loads and serves the model       |

## Step 2: Understand your Python environment

Your document identifies a major problem:

Java cannot find the Python environment containing the required ML libraries.

For example, your terminal might use:

```
spring-ai-rag-demo/
    .venv/
        bin/
            python
```

But Java executes:

```
new ProcessBuilder("python", "train.py");
```

The problem is that Java's process does not necessarily inherit the activated shell environment.

First, activate Python manually:

```
cd spring-ai-rag-demo

python3 -m venv .venv

source .venv/bin/activate
```

Check the executable:

```
which python
```

Expected output:

```
/Users/rajesh/Desktop/spring-ai-rag-demo/.venv/bin/python
```

Now test:

```
python --version
python -m pip --version
```

The important concept: always use the Python executable from the environment in which you installed the ML dependencies.

## Step 3: Configure Python in Spring Boot

Instead of hardcoding the executable in Java, configure it in `application.properties`.

```
python.executable.path=${user.dir}/.venv/bin/python

fine-tuning.script.path=${user.dir}/python/train.py

fine-tuning.dataset.path=${user.dir}/datasets/training.json

fine-tuning.output.path=${user.dir}/models
```

In Java:

```
@Value("${python.executable.path}")
private String pythonExecutable;

@Value("${fine-tuning.script.path}")
private String trainingScript;
```

Why?

Because different environments have different Python installations.

For example:

```
# Mac development
python.executable.path=${user.dir}/.venv/bin/python
```

```
# Linux production
python.executable.path=/opt/venvs/finetuning/bin/python
```

This makes your application configurable.

## Step 4: Understand ProcessBuilder

This is the bridge between Java and Python.

```
ProcessBuilder pb = new ProcessBuilder(
    pythonExecutable,
    trainingScript,
    "--dataset", datasetPath,
    "--output", outputPath,
    "--gguf", ggufPath
);

pb.redirectErrorStream(true);

Process process = pb.start();
```

What happens?

1. Java receives the training request.
2. Java constructs the Python command.
3. `ProcessBuilder` launches a new operating-system process.
4. Python starts executing `train.py`.
5. Python loads the dataset.
6. Python loads the model.
7. Python performs training.
8. Python saves the trained adapter.
9. Java reads the process output and determines success or failure.

Think of it this way:

```
Java Application
      |
      | ProcessBuilder.start()
      |
      v
Operating System Process
      |
      v
Python Interpreter
      |
      v
train.py
```

Important: your Java application and Python interpreter must be able to access the same filesystem paths. If Java runs inside Docker, your host's `.venv` is not automatically accessible.

## Step 5: Understand the Mac hardware problem

This is one of the most important findings in your document.

Your environment:

```
Operating System: macOS
Project: spring-ai-rag-demo
Python: .venv
Model: Qwen2.5-7B-Instruct
Training framework: Unsloth
```

The original implementation attempted to use Unsloth.

However:

- Unsloth's conventional CUDA training setup requires an NVIDIA GPU.
- Your Mac does not have an NVIDIA CUDA GPU.
- Apple Silicon uses Apple's Metal framework (MPS).
- CUDA-based 4-bit BitsAndBytes training is not a drop-in solution for MPS.

Therefore, the original Unsloth configuration is not appropriate for your Mac.

fine-tune-lora.md

[Meilleur Mac pour IA en 2026 : LLM Locaux et OpenClaw](https://images.openai.com/static-rsc-4/yOWk_EbsNnYJJbi_On-YW2PzWpd0sW-RnakTVq8C5-lISpfuUmwIb34s4yqPFax_i-XaeBSiVrx8PdSxfBzZ1reu6dfY2mFnJMxwBcaqHHC3YfdgK_5EQwEBsSA13GG6luySTCciCXlelRxBGCo60WZjd8yt7iblJqg106n9Bn0?purpose=inline)

[refurb.me](https://www.refurb.me/fr/blog/meilleur-mac-pour-ia?utm_source=chatgpt.com)

There are three hardware possibilities:

| Environment             | Training approach                        |
| ----------------------- | ---------------------------------------- |
| NVIDIA GPU Linux server | Unsloth + QLoRA                          |
| Apple Silicon Mac       | MLX-LM or compatible PyTorch + PEFT LoRA |
| CPU-only machine        | Small-model LoRA, generally slow         |

My recommendation: For your Mac, investigate MLX-LM first. It is designed for Apple Silicon. Do not assume that the supplied PyTorch script provides QLoRA on MPS.

## Step 6: Install Python dependencies

Your document proposes these dependencies:

`requirements.txt`

```
transformers
peft
trl
datasets
accelerate
mlx-lm
```

Install:

```
source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r python/requirements.txt
```

However, there is an important implementation detail:

- `transformers`, `peft`, `trl`, `datasets` and `accelerate` form the Hugging Face training stack.
- `mlx-lm` is a separate Apple Silicon-oriented training/inference ecosystem.
- Installing both does not automatically make a PyTorch training script use MLX.

You need to select the actual training backend before writing the final training script.

## Step 7: Understand the Python training script

Your document proposes replacing Unsloth with Hugging Face PEFT.

Let's understand the major stages.

7.1 Import libraries

```
import argparseimport torchfrom transformers import (    AutoModelForCausalLM,    AutoTokenizer,    BitsAndBytesConfig)from datasets import load_datasetfrom peft import (    LoraConfig,    get_peft_model,    prepare_model_for_kbit_training)from trl import SFTTrainer, SFTConfig
```

What each library does:

| Library      | Responsibility                         |
| ------------ | -------------------------------------- |
| PyTorch      | Tensor computation                     |
| Transformers | Loading pretrained models              |
| Datasets     | Loading training data                  |
| PEFT         | LoRA adapter implementation            |
| TRL          | Supervised fine-tuning                 |
| BitsAndBytes | Quantization, primarily CUDA workflows |

7.2 Receive parameters from Java

```
parser = argparse.ArgumentParser()parser.add_argument("--dataset", required=True)parser.add_argument("--output", required=True)parser.add_argument("--gguf", required=True)args = parser.parse_args()
```

Java passes these parameters:

```
python train.py \
  --dataset datasets/training.json \
  --output models/qwen-custom \
  --gguf models/qwen-custom.gguf
```

The Python script receives them through `args`.

7.3 Load the base model

```
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)model = AutoModelForCausalLM.from_pretrained(    MODEL_NAME)
```

What happens?

1. The tokenizer loads the vocabulary and tokenization rules.
2. The model loader downloads or reads the pretrained model.
3. The model weights are loaded into memory.
4. The model becomes available for training.

The 7B designation means approximately seven billion parameters.

7.4 Configure LoRA

```
peft_config = LoraConfig(    r=16,    lora_alpha=16,    target_modules=[        "q_proj",        "v_proj",        "k_proj",        "o_proj"    ],    lora_dropout=0.05,    bias="none",    task_type="CAUSAL_LM")model = get_peft_model(    model,    peft_config)
```

This is where the actual LoRA adapter configuration is applied.

`get_peft_model()` wraps the pretrained model with trainable adapter layers.

7.5 Load the training dataset

The original document uses JSON:

```
dataset = load_dataset(    "json",    data_files=args.dataset,    split="train")
```

It expects a `messages` field.

For illustration, a compatible chat dataset could look like:

`training.json`

```
[
  {
    "messages": [
      {
        "role": "system",
        "content": "You are a banking domain assistant."
      },
      {
        "role": "user",
        "content": "What is a payment reconciliation?"
      },
      {
        "role": "assistant",
        "content": "Payment reconciliation is the process of comparing payment records across systems to identify discrepancies."
      }
    ]
  },
  {
    "messages": [
      {
        "role": "user",
        "content": "What is an authorization decision?"
      },
      {
        "role": "assistant",
        "content": "An authorization decision determines whether a financial transaction is approved or declined."
      }
    ]
  }
]
```

The dataset teaches the model the expected responses.

7.6 Apply the chat template

```
def format_example(example):    return tokenizer.apply_chat_template(        example["messages"],        tokenize=False,        add_generation_prompt=False    )
```

Why?

The model does not directly understand JSON objects as conversations.

The tokenizer converts the conversation into the model's expected chat format, including special tokens and role boundaries.

7.7 Start training

The original document configures:

```
SFTConfig(    output_dir=args.output,    num_train_epochs=2,    per_device_train_batch_size=1,    gradient_accumulation_steps=4,    learning_rate=2e-4,    logging_steps=1,    report_to="none")
```

Explanation:

| Configuration                 | Meaning                              |
| ----------------------------- | ------------------------------------ |
| num_train_epochs=2            | Process the dataset twice            |
| batch_size=1                  | One example per device batch         |
| gradient_accumulation_steps=4 | Accumulate gradients over four steps |
| learning_rate=2e-4            | Learning rate                        |
| logging_steps=1               | Log each training step               |

Conceptually:

```
trainer.train()
```

This executes the training process.

7.8 Save the trained adapter

```
model.save_pretrained(args.output)tokenizer.save_pretrained(args.output)
```

After successful training, you should have a directory containing the adapter configuration and adapter weights.

For example:

```
models/
    qwen-custom/
        adapter_config.json
        adapter_model.safetensors
        tokenizer_config.json
```

The exact generated files depend on the library and model configuration.



Important

&#x20;This is the actual LoRA training stage. The system-prompt approach described later does not perform this operation.



fine-tune-lora.md

## Step 8: Export the trained model to GGUF

Why do we need GGUF?

Your training framework may save the adapter in Hugging Face/PEFT format, whereas Ollama commonly uses GGUF models for local model distribution.

The conversion process is:

```
Qwen Base Model
       +
Trained LoRA Adapter
       |
       v
Merge Adapter (if required)
       |
       v
Export / Convert
       |
       v
GGUF Model
       |
       v
Ollama
```

Your document suggests using `llama.cpp` for GGUF conversion on Mac.

However, it does not provide the actual conversion commands or a complete adapter-merging implementation.

Also, there are two distinct export possibilities:

1. Export a merged full model into GGUF.
2. Export a compatible LoRA adapter and reference it from an Ollama Modelfile.

Do not simply rename an adapter file to `.gguf`. It must be converted using a compatible conversion workflow.

## Step 9: Register the model with Ollama

Once you have a compatible model or adapter, Ollama can serve it.

For example, a Modelfile for a merged GGUF model:

```
FROM ./qwen-custom.gguf

PARAMETER temperature 0.2
PARAMETER num_ctx 8192

SYSTEM """
You are a banking domain assistant.
Answer accurately and concisely.
"""
```

Create the model:

```
ollama create banking-assistant -f Modelfile
```

Run it:

```
ollama run banking-assistant
```

You can also use Ollama's HTTP API from Java.

For example:

```
POST http://localhost:11434/api/chat
Content-Type: application/json
```

```
{
  "model": "banking-assistant",
  "messages": [
    {
      "role": "user",
      "content": "Explain payment reconciliation."
    }
  ],
  "stream": false
}
```

# 4. Alternative: Your Java + Ollama system-prompt workaround

This is the second approach in your document.

It is particularly important because you encountered insufficient disk space on your Mac.

Instead of training model weights, Java creates a customized Ollama model using a system prompt.

[Running Google Gemma 4 Locally: Truth, Specs, Hardware, and Use Cases| Sabbirz | Blog](https://images.openai.com/static-rsc-4/Vv81XXOEF0ZzsLGCjvLqbE53LDabuXY2wPVHR0wXRJQdOC2LolxCtRWVowgH1k_iU8XPP2LxObY-91foh4UioinPYfoTlyfUW1tq4u2EXKJARFz6v66gxsM-l6hXCVk5pPnAs1Twg9fWSwNQCkq66q_MUftqFW0djj_2oKxOZIM?purpose=inline)

[sabbirz.com](https://www.sabbirz.com/blog/running-google-gemma-4-locally-truth-specs-hardware-and-use-cases?utm_source=chatgpt.com)

The architecture becomes:

```
React UI
   |
   v
Spring Boot REST API
   |
   v
FineTuningService.java
   |
   +---- Load domain knowledge
   |
   +---- Build system prompt
   |
   +---- Generate Modelfile
   |
   +---- Call Ollama /api/create
   |
   v
Customized Ollama Model
```

## Step 1: Build the system prompt

Your Java code can construct a prompt:

```
StringBuilder systemPrompt = new StringBuilder();

systemPrompt.append(
    "You are a banking domain expert.\n"
);

systemPrompt.append(
    "Always use the supplied domain knowledge.\n"
);

systemPrompt.append(
    "Provide accurate answers.\n"
);

systemPrompt.append(
    "Do not invent banking information.\n"
);
```

You can append domain-specific information:

```
systemPrompt.append(
    "Payment reconciliation compares financial records.\n"
);

systemPrompt.append(
    "Authorization decisioning determines whether transactions are approved or declined.\n"
);
```

This gives the model additional instructions and context.

## Step 2: Generate the Modelfile

Your document provides this approach:

```
String modelfileContent = String.format(
    """
    FROM %s
    PARAMETER temperature %.2f
    PARAMETER num_ctx 8192
    SYSTEM \"\"\"
    %s
    \"\"\"
    """,
    baseModel,
    temperature,
    systemPrompt
);
```

Save it:

```
Path modelfile = modelOutputDir.resolve("Modelfile");

Files.writeString(
    modelfile,
    modelfileContent,
    StandardCharsets.UTF_8
);
```

Now your Java application has generated an Ollama model configuration.

## Step 3: Call Ollama API

Your document uses:

```
POST http://localhost:11434/api/create
```

Example payload:

```
{
  "model": "banking-assistant",
  "from": "qwen2.5:7b",
  "system": "You are a banking domain expert.",
  "parameters": {
    "temperature": 0.2
  },
  "stream": false
}
```

Java can create this payload with Jackson:

```
Map<String, Object> payload = new HashMap<>();

payload.put("model", targetModel);
payload.put("from", baseModel);
payload.put("system", systemPrompt.toString());

payload.put(
    "parameters",
    Map.of("temperature", temperature)
);

payload.put("stream", false);
```

Send the request:

```
HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create(ollamaBaseUrl + "/api/create"))
    .header("Content-Type", "application/json")
    .timeout(Duration.ofMinutes(5))
    .POST(
        HttpRequest.BodyPublishers.ofString(
            objectMapper.writeValueAsString(payload)
        )
    )
    .build();

HttpResponse<String> response =
    httpClient.send(
        request,
        HttpResponse.BodyHandlers.ofString()
    );
```

If successful, Ollama registers the customized model.

Important distinction: This approach does not update the Qwen neural network weights. It creates a model configuration with different instructions and parameters. It is useful for experimentation but is not LoRA fine-tuning.

fine-tune-lora.md

# 5. Troubleshooting: Disk space problem

Your document reports:

Mac storage at the time of the error

# 5.4 GB

Available space

98% used

2% available

These are the historical figures in your uploaded document, not a live measurement.

The installation failed with:

```
OSError: [Errno 28] No space left on device
```

The document identifies the following storage consumers:

1. PyTorch and its dependencies.
2. Hugging Face model downloads.
3. Ollama model files.
4. Python virtual environments.
5. Docker images and caches.
6. Xcode developer files.

fine-tune-lora.md

## Step 1: Check available space

```
df -h
```

Find large directories:

```
du -sh ~/.cache/huggingface
du -sh ~/.ollama
du -sh ~/Library/Developer/Xcode
du -sh ~/Library/Caches
```

For Docker:

```
docker system df
```

## Step 2: Clear unnecessary caches

Python cache:

```
pip cache purge
```

Homebrew cache:

```
brew cleanup -s
```

Hugging Face cache:

```
du -sh ~/.cache/huggingface/hub
```

Only if you are comfortable downloading the models again, remove their cached files:

```
rm -rf ~/.cache/huggingface/hub/*
```

Check Ollama models:

```
ollama list
```

Remove only models you no longer need:

```
ollama rm MODEL_NAME
```

Be careful with Docker cleanup. In particular, `docker system prune -a --volumes` can delete persistent volumes containing databases and other important application data. Do not run it blindly.

## Step 3: Verify storage again

```
df -h
```

Your document recommends approximately 15–20 GB for installing dependencies and additional space for model weights and training. Its later recommendation is to free at least 25 GB.

For Qwen 2.5 7B, actual requirements depend on precision, training backend, memory and model storage.

# 6. Important implementation issues I found in your document

These are worth correcting before you start coding.

| Issue                | Explanation                                                                          |
| -------------------- | ------------------------------------------------------------------------------------ |
| Unsloth on Mac       | Conventional CUDA-based Unsloth setup is not suitable for your Mac                   |
| QLoRA on MPS         | The supplied code does not implement 4-bit QLoRA on Apple Metal                      |
| CPU fallback         | The supplied code still configures BitsAndBytes when CUDA is unavailable             |
| GGUF export          | The supplied Python script never actually uses `args.gguf` to export GGUF            |
| TRL compatibility    | `SFTTrainer` constructor arguments depend on the installed TRL version               |
| Ollama customization | `/api/create` with a system prompt does not train LoRA weights                       |
| Python path          | Java must use the correct Python executable                                          |
| Disk capacity        | Your historical 5.4 GB free space is insufficient for the proposed 7B training setup |

The biggest problem is that the document describes the Python implementation as a platform-agnostic fallback, but the code is not a complete cross-platform training implementation.

I would not run it unchanged.

# 7. Which implementation should you choose?

Here is my recommendation for your current Mac development environment.

Recommended immediately

Option A — Java + Ollama system-prompt customization

No separate PyTorch training environment required.

- Reuse your existing Qwen model.
- Build domain-specific system prompts.
- Generate Modelfiles.
- Register customized models through Ollama.
- Test from your Spring Boot application.

For actual fine-tuning

Option B — MLX-LM + LoRA on Apple Silicon

Use an Apple Silicon-compatible training workflow.

- Prepare the training dataset.
- Train LoRA adapters using MLX-LM.
- Evaluate the adapter.
- Export or convert the resulting model for Ollama.
- Integrate training orchestration with Spring Boot.

Option C — Unsloth + QLoRA on NVIDIA Linux

For a dedicated GPU training server.

- NVIDIA GPU with supported CUDA.
- Unsloth.
- 4-bit quantized base model.
- QLoRA training.
- Export and deploy through Ollama.

# 8. How to test whether the customization works

Regardless of which approach you choose, test the original model and customized model using the same questions.

Example:

| Test                                  | Base Qwen      | Customized model                |
| ------------------------------------- | -------------- | ------------------------------- |
| Explain payment reconciliation        | General answer | Expected domain-specific answer |
| Explain authorization decisioning     | General answer | Expected domain-specific answer |
| Explain banking transaction lifecycle | General answer | Expected domain-specific answer |

For actual LoRA training, also measure:

- Training loss.
- Validation loss.
- Response accuracy.
- Hallucination rate.
- Inference latency.
- Memory consumption.

Remember that fine-tuning teaches patterns and behavior; it is not automatically a reliable replacement for RAG when retrieving changing or factual enterprise data.

# 9. Final understanding: What have we built?

## Java + Python + LoRA + Ollama

1. Spring Boot

   Receives requests from React and manages the workflow.
2. FineTuningService

   Prepares datasets, configuration and training commands.
3. ProcessBuilder

   Starts the Python process from Java.
4. Python training engine

   Loads Qwen and applies LoRA to train adapters.
5. Model export

   Converts or merges the trained model into an Ollama-compatible format.
6. Ollama

   Loads and serves the customized model.
7. React UI

   Allows users to submit questions and interact with the model.

# 10. Let's recreate the project practically

Before generating all the source code, I want to establish the correct hardware and training approach. Your uploaded document does not establish whether your Mac has Apple Silicon or Intel, and that changes the implementation.

1\. Which Mac processor do you have?

Apple Silicon (M1 / M2 / M3 / M4 / M5)

Intel Mac

Not sure

2\. What do you want to implement?

Actual LoRA fine-tuning (train model weights)

Actual QLoRA (4-bit quantized training)

Both LoRA and system-prompt customization

System-prompt customization only

3\. Which base model?

Qwen 2.5 7B Instruct

Qwen 3 8B

Smaller model (0.5B–3B)

4\. How much free disk space do you currently have?

Less than 10 GB

10–25 GB

25–50 GB

More than 50 GB

Not checked yet

Build my complete project

My goal is to give you a reproducible project, not just disconnected code snippets: Java APIs, Python training, dataset preparation, adapter generation, Ollama integration, and complete execution instructions.