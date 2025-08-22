from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

import torch
import os

torch_device = "cpu"

# Load a prompt-response dataset
print("Loading prompt-response dataset...")
dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:2000]")  # Small subset for demo
print("First Prompt:", dataset[0]["instruction"])
print("First Response:", dataset[0]["response"])

# Load tokenizer and model
model_name = os.getenv("ORIGINAL_MODEL", "google/gemma-3-270m")
print(f"Loading model and tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, attn_implementation='eager')
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 doesn't have a pad token by default
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Tokenize the dataset: concatenate instruction and response
def preprocess(examples):
    texts = [
        f"### Instruction:\n\n{inst}\n\n### Response:\n\n{resp}"
        for inst, resp in zip(examples["instruction"], examples["response"])
    ]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    # Add labels for causal LM (labels = input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_steps=50,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune
print("Starting training...")
trainer.train()

# Save model and tokenizer
print("Saving model and tokenizer...")
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_tokenizer")