import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--gguf", type=str, required=True)
args = parser.parse_args()

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 4048

# ... (Keep your existing Unsloth model loading, lora targets, and format functions)

dataset = load_dataset("json", data_files=args.dataset, split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=format_example,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True,
    args=SFTConfig(
        output_dir=args.output,
        num_train_epochs=3, # Scale up safely now
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        report_to="none",
    ),
)

trainer.train()

# Export seamlessly back to the Java output directories
model.save_pretrained_gguf(
    args.gguf,
    tokenizer,
    quantization_method="q4_k_m",
)
print("SUCCESS: GGUF generation complete.")
