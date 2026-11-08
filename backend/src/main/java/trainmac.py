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
