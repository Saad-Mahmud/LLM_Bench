#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script loads a JSON file created by the previous unsloth inference script,
processes each sample’s generated text by:
  1. Keeping only the text after the "<answer>" tag (if it exists).
  2. Removing any repeating cycle at the end of the text.
It then saves the processed results into a new JSON file.
Usage:
    python process_results.py input_file.json output_file.json
"""

import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

# --- Load Datasets ---
# GSM8K uses the "question" field.
gsm8k = load_dataset("gsm8k", "main", split="test")
# MATH-500 uses the "problem" field.
math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
# AIME uses the "problem" field.
aime = load_dataset("AI-MO/aimo-validation-aime", split="train")

datasets_info = {"GSM8K": gsm8k, "MATH500": math500, "AIME": aime,}


def post_process_generated_text(generated_text):
    """
    Post-process the generated text by:
      1. Extracting text after the <answer> tag (if present).
      2. Removing any cyclic repetition at the end.
    """
    # Step 1: If an <answer> tag exists, keep only the text after it.
    if "<answer>" in generated_text:
        generated_text = generated_text.split("<answer>", 1)[1].strip()
    generated_text = generated_text.replace("</answer>", "").strip()
    # Step 2: Remove any cyclic repetition at the end.
    return generated_text

def process_json_file(input_file, output_file, data_name):
    # Load the JSON file generated by the previous script.
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each entry.
    for idx, entry in enumerate(data):
        # Process the 'generated_text' and update the 'answer' field.
        processed_answer = post_process_generated_text(entry.get("generated_text", ""))
        entry["answer"] = processed_answer
        entry["correct_answer"] = datasets_info[data_name][idx]["answer"]

    # Save the processed results to a new JSON file.
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Processed JSON saved to {output_file}")

def main(input_file, output_file, data_name):
    process_json_file(input_file, output_file, data_name)
    
if __name__ == "__main__":
    main("results/AIME_unsloth_Meta-Llama-3.1-8B_results.json","results/AIME_unsloth_Meta-Llama-3.1-8B_results_p.json", "AIME")
    main("results/GSM8K_unsloth_Meta-Llama-3.1-8B_results.json","results/GSM8K_unsloth_Meta-Llama-3.1-8B_results_p.json", "GSM8K")
    main("results/MATH500_unsloth_Meta-Llama-3.1-8B_results.json","results/MATH500_unsloth_Meta-Llama-3.1-8B_results_p.json","MATH500")
    
    main("results/AIME_unsloth_Meta-Llama-3.1-8B-S1_results.json","results/AIME_unsloth_Meta-Llama-3.1-8B-S1_results_p.json", "AIME")
    main("results/GSM8K_unsloth_Meta-Llama-3.1-8B-S1_results.json","results/GSM8K_unsloth_Meta-Llama-3.1-8B-S1_results_p.json", "GSM8K")
    main("results/MATH500_unsloth_Meta-Llama-3.1-8B-S1_results.json","results/MATH500_unsloth_Meta-Llama-3.1-8B-S1_results_p.json","MATH500")
    