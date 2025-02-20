#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download GSM8K, MATH500, and AIME datasets and run unsloth inference on a sample from each.

The prompt format is as follows:
    [system] You are a helpful math solver.
    [user] <question>
    <thinking> Provide step-by-step reasoning: </thinking>
    <answer> 

This script loads unsloth’s Meta-Llama-3.1-8B model, samples up to 100 questions from each dataset,
generates answers ensuring termination by passing the end-of-sequence token, and saves the results
in separate JSON files for each dataset.
"""

import torch
import json
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import AutoTokenizer

# --- Load the unsloth model and tokenizer ---
max_seq_length = 8192  # adjust as needed
dtype = None
load_in_4bit = True
#model_path = "unsloth/Meta-Llama-3.1-8B"
#model_name = "unsloth_Meta-Llama-3.1-8B"
model_path ="S1_Models/Qwen2.5-14B-S1"
model_name = "unsloth_Qwen2.5-14B-S1"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable inference optimizations.
FastLanguageModel.for_inference(model)

# --- Define a helper function to format a prompt ---
def format_prompt(question):
    system_msg = "You are a helpful math solver. You need to think about the solution step by step. Provide a concise final answer at the end."
    prompt = (
        f"[system] {system_msg}\n"
        f"[user] {question}\n"
        f"<thinking> Let's think about this step-by-step."
    )
    return prompt

# --- Load Datasets ---
# GSM8K uses the "question" field.
gsm8k = load_dataset("gsm8k", "main", split="test")
# MATH-500 uses the "problem" field.
math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
# AIME uses the "problem" field.
aime = load_dataset("AI-MO/aimo-validation-aime", split="train")

datasets_info = [
    {"name": "GSM8K", "dataset": gsm8k, "field": "question"},
    {"name": "MATH500", "dataset": math500, "field": "problem"},
    {"name": "AIME", "dataset": aime, "field": "problem"},
]

# Determine device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Inference and Saving Results ---
for info in datasets_info:
    dataset_name = info["name"]
    ds = info["dataset"]
    field = info["field"]
    num_samples = min(100, len(ds))
    results = []
    
    print(f"Processing {dataset_name} with {num_samples} samples...")
    for i in range(num_samples):
        question = ds[i][field]
        prompt = format_prompt(question)
        
        # Tokenize input prompt and move to device.
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response.
        # Pass eos_token_id to ensure the generation stops appropriately.
        outputs = model.generate(
            **inputs,
            max_new_tokens=8000,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Optionally extract the answer after the "<answer>" tag.
        #answer_parts = generated_text.split("<answer>")
        #answer = answer_parts[1].strip() if len(answer_parts) > 1 else generated_text.strip()
        
        results.append({
            "question": question,
            "prompt": prompt,
            "generated_text": generated_text,
            "answer": "",
        })
        print(f"  Processed sample {i+1}/{num_samples} for {dataset_name}.")
    
    # Save results to a JSON file (including model name in the filename).
    output_filename = f"{dataset_name}_{model_name}_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results for {dataset_name} to {output_filename}.")
