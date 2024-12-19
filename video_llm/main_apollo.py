#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model name
model_name = "GoodiesHere/Apollo-LMMs-Apollo-7B-t32"

# Update/install required packages if not done yet:
# pip install --upgrade transformers accelerate

# Load the tokenizer with remote code execution allowed
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    trust_remote_code=True
)

# Load the model with remote code execution allowed
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# Now you can use the model for inference
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated text:")
print(generated_text)
