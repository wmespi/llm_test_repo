from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate text from a Hugging Face model.")
parser.add_argument('--ft', type=bool, default=False,
                    help='Boolean to indicate if the model is fine-tuned.')
parser.add_argument('--prompt', type=str, default="Write a short poem about the ocean.",
                    help='The prompt to send to the model.')
args = parser.parse_args()

# Determine device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {torch_device}")

# Model and prompt
ft = args.ft
prompt = "#INSTRUCTIONS:\n\n" + args.prompt + "\n\n#RESPONSE:\n\n"

# Determine model name based on fine-tuning status
model_name = os.getenv("ORIGINAL_MODEL", "google/gemma-3-270m")
model_name = model_name if not ft else "./finetuned_model"
tokenizer_name = model_name if not ft else "./finetuned_tokenizer"
print(f"Using model: {model_name}")
print(f"Using tokenizer: {tokenizer_name}")

# Display input prompt
print(f"Input prompt: {prompt}")

# Load model and tokenizer
print(f"Loading model and tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(torch_device)

# Generate text
print("Generating inputs...")
inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)
print(f"Input IDs: {inputs['input_ids']}")
print(f"Generating outputs...")
outputs = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
print(f"Output IDs: {outputs}")
print("Decoding outputs...")
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated output:\n{text}")