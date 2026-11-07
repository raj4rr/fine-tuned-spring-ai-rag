"""Fine-tune an instruction model with Unsloth + QLoRA.

Run on a CUDA-capable machine. The script intentionally starts with a small
training configuration so the experiment can be evaluated before scaling up.
"""
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "output/spring-ai-qwen2.5-7b"
GGUF_DIR = "output/gguf"
MAX_SEQ_LENGTH = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
)

dataset = load_dataset("json", data_files="dataset/spring-ai.jsonl", split="train")

# Apply the model's own chat template so training and inference use the same
# conversation format.
def format_example(example):
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=format_example,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=42,
    ),
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# GGUF is convenient for local Ollama inference. q4_k_m is a practical
# starting quantization; choose another quantization if your deployment needs it.
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q4_k_m",
)

print(f"LoRA/adapter output: {OUTPUT_DIR}")
print(f"GGUF output: {GGUF_DIR}")
