from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import os

# Determine device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {torch_device}")

# Model and prompt
print("Setting up model and prompt...")
prompt = "### Instruction:\nWrite a short poem about the ocean.\n### Response:\n"
model_name = "meta-llama/Llama-3.1-8B"

# Load model and tokenizer
print(f"Loading model and tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(torch_device)

# Generate text
print("Generating text...")
inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)
print(f"Input IDs: {inputs['input_ids']}")
outputs = model.generate(**inputs, max_length=20, pad_token_id=tokenizer.eos_token_id)
print(f"Output IDs: {outputs}")
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated output:\n{text}")