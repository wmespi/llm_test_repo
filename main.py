from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load a prompt-response dataset
print("Loading prompt-response dataset...")
dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:2000]")  # Small subset for demo
print(dataset)

# Load tokenizer and model
model_name = "gpt2"
print(f"Loading model and tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 doesn't have a pad token by default
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Tokenize the dataset: concatenate instruction and response
def preprocess(examples):
    texts = [
        f"### Instruction:\n{inst}\n### Response:\n{resp}"
        for inst, resp in zip(examples["instruction"], examples["response"])
    ]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

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