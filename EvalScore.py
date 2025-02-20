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
import requests

def cal_score(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each entry.
    total = 0
    cor = 0
    for idx, entry in tqdm(enumerate(data)):
        # Process the 'generated_text' and update the 'answer' field.
        question = entry["question"]
        generated_ans = entry["answer"]
        correct_ans = entry["correct_answer"]
        prompt = "You are a helpful math grader. You are going to evaluate some AI generted answer with actual correct answer and say if it is correct or not.\n"
        prompt = prompt + f"Question: {question}\n"
        prompt = prompt + f"AI generated solution: {generated_ans}\n"
        prompt = prompt + f"Correct solution to the math question: {correct_ans}\n"
        prompt = prompt + f"Print Yes or No if the AI generted answer was correct or not.\n Answer: "
        query = {
                "prompt": prompt,
                "type": "mcq_logits",
                "class_names": ["Yes", "No", "Yes.", "No."],
                "cot": False,
                "eof_tag": "<END>"
            }
        response = requests.post("http://localhost:8000/generate", json=query)
        if response.status_code == 200:
            data = response.json()
            #print("Output:", data["output"])
            #print("Class_names:", data["class_names"])
            cor += (data["output"] == "Yes" or data["output"] == "Yes.") 
            total +=1 
            #print("Time taken:", data["time_taken"])
        else:
            print("Error:", response.text)

    print(f"File Name: {input_file}, Accuracy: {cor/total}")

if __name__ == "__main__":
    
    cal_score("results/AIME_unsloth_Meta-Llama-3.1-8B_results_p.json")
    cal_score("results/GSM8K_unsloth_Meta-Llama-3.1-8B_results_p.json")
    cal_score("results/MATH500_unsloth_Meta-Llama-3.1-8B_results_p.json")
    
    cal_score("results/AIME_unsloth_Meta-Llama-3.1-8B-S1_results_p.json")
    cal_score("results/GSM8K_unsloth_Meta-Llama-3.1-8B-S1_results_p.json")
    cal_score("results/MATH500_unsloth_Meta-Llama-3.1-8B-S1_results_p.json")
    

    cal_score("results/AIME_unsloth_Mistral-Small-24B-Instruct-2501_results_p.json")
    cal_score("results/GSM8K_unsloth_Mistral-Small-24B-Instruct-2501_results_p.json")
    cal_score("results/MATH500_unsloth_Mistral-Small-24B-Instruct-2501_results_p.json")
    
    cal_score("results/AIME_unsloth_Mistral-Small-24B-Instruct-2501-S1_results_p.json")
    cal_score("results/GSM8K_unsloth_Mistral-Small-24B-Instruct-2501-S1_results_p.json")
    cal_score("results/MATH500_unsloth_Mistral-Small-24B-Instruct-2501-S1_results_p.json")
    

    cal_score("results/AIME_unsloth_Qwen2.5-14B-Instruct_results_p.json")
    cal_score("results/GSM8K_unsloth_Qwen2.5-14B-Instruct_results_p.json")
    cal_score("results/MATH500_unsloth_Qwen2.5-14B-Instruct_results_p.json")
    
    cal_score("results/AIME_unsloth_Qwen2.5-14B-S1_results_p.json")
    cal_score("results/GSM8K_unsloth_Qwen2.5-14B-S1_results_p.json")
    cal_score("results/MATH500_unsloth_Qwen2.5-14B-S1_results_p.json")
    
    #File Name: results/AIME_unsloth_Meta-Llama-3.1-8B_results_p.json, Accuracy: 0.1
    #File Name: results/GSM8K_unsloth_Meta-Llama-3.1-8B_results_p.json, Accuracy: 0.42
    #File Name: results/MATH500_unsloth_Meta-Llama-3.1-8B_results_p.json, Accuracy: 0.22
    #File Name: results/AIME_unsloth_Meta-Llama-3.1-8B-S1_results_p.json, Accuracy: 0.2777777777777778
    #File Name: results/GSM8K_unsloth_Meta-Llama-3.1-8B-S1_results_p.json, Accuracy: 0.7
    #File Name: results/MATH500_unsloth_Meta-Llama-3.1-8B-S1_results_p.json, Accuracy: 0.4639175257731959
    #File Name: results/AIME_unsloth_Mistral-Small-24B-Instruct-2501_results_p.json, Accuracy: 0.4444444444444444
    #File Name: results/GSM8K_unsloth_Mistral-Small-24B-Instruct-2501_results_p.json, Accuracy: 0.96
    #File Name: results/MATH500_unsloth_Mistral-Small-24B-Instruct-2501_results_p.json, Accuracy: 0.8
    #File Name: results/AIME_unsloth_Mistral-Small-24B-Instruct-2501-S1_results_p.json, Accuracy: 0.37777777777777777
    #File Name: results/GSM8K_unsloth_Mistral-Small-24B-Instruct-2501-S1_results_p.json, Accuracy: 0.98
    #File Name: results/MATH500_unsloth_Mistral-Small-24B-Instruct-2501-S1_results_p.json, Accuracy: 0.83
    #File Name: results/AIME_unsloth_Qwen2.5-14B-Instruct_results_p.json, Accuracy: 0.43333333333333335
    #File Name: results/GSM8K_unsloth_Qwen2.5-14B-Instruct_results_p.json, Accuracy: 0.96
    #File Name: results/MATH500_unsloth_Qwen2.5-14B-Instruct_results_p.json, Accuracy: 0.84
    #File Name: results/AIME_unsloth_Qwen2.5-14B-S1_results_p.json, Accuracy: 0.32222222222222224
    #File Name: results/GSM8K_unsloth_Qwen2.5-14B-S1_results_p.json, Accuracy: 0.96
    #File Name: results/MATH500_unsloth_Qwen2.5-14B-S1_results_p.json, Accuracy: 0.88
