#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finetune LLama 3.1 8B with Unsloth using the simplescaling/s1K-1.1 dataset.
Each dataset example is expanded into two training samples:
    - One using the Gemini reasoning trace and attempt.
    - One using the DeepSeek reasoning trace and attempt.

Each prompt is structured as:
    [system] <system instruction>
    [user] <question>
    [thinking] <reasoning trace truncated as needed>
    [answer] <attempt>

The complete prompt is truncated such that its token length does not exceed 8K tokens.
After training, the model is saved to disk and then reloaded for inference.
The script also prints the maximum token length of the samples (after truncation).
"""

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import numpy as np
# Settings
max_seq_length = 8192  # You can adjust this if needed.
dtype = None           # Auto-detect datatype; adjust if needed.
load_in_4bit = True    # Use 4bit quantization to reduce memory usage.
MAX_TOTAL_TOKENS = 8000  # Maximum allowed tokens per sample.

# Load the simplescaling/s1K-1.1 dataset (there is only one split).
raw_dataset = load_dataset("simplescaling/s1K-1.1", split="train")

# Load the model and tokenizer.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Mistral-Small-24B-Instruct-2501",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Get the EOS token (ensures generation stops).
EOS_TOKEN = tokenizer.eos_token

def build_prompt(system_msg, question, thinking, attempt):
    """
    Build the full prompt from components.
    If the complete prompt (tokenized) is longer than MAX_TOTAL_TOKENS,
    the "thinking" section is truncated as needed.
    The prompt now uses explicit opening and closing XML-style tags.
    """
    base_prompt = f"[system] {system_msg}\n[user] {question}\n<thinking> "
    answer_part = f" </thinking>\n<answer> {attempt}</answer>{EOS_TOKEN}"
    
    # Token counts for base parts.
    base_tokens = tokenizer(base_prompt, add_special_tokens=False)["input_ids"]
    answer_tokens = tokenizer(answer_part, add_special_tokens=False)["input_ids"]
    base_count = len(base_tokens)
    answer_count = len(answer_tokens)
    
    # Available tokens for the thinking section.
    available = MAX_TOTAL_TOKENS - (base_count + answer_count)
    if available <= 0:
        truncated_thinking = ""
    else:
        thinking_tokens = tokenizer(thinking, add_special_tokens=False)["input_ids"]
        if len(thinking_tokens) > available:
            truncated_tokens = thinking_tokens[:available]
            truncated_thinking = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        else:
            truncated_thinking = thinking
    return base_prompt + truncated_thinking + answer_part

def process_example(example):
    """
    Given an example from the dataset, create training samples:
      - One using the Gemini reasoning fields.
      - One using the DeepSeek reasoning fields.
    Each sample is built with the helper function to ensure the tokenized
    prompt does not exceed MAX_TOTAL_TOKENS.
    """
    system_msg = "You are a reasoning model trained to reason about math, code, and other topics."
    samples = []
    question = example.get("question", "").strip()

    # Process Gemini fields if available.
    gemini_thinking = example.get("gemini_thinking_trajectory", "").strip()
    gemini_attempt = example.get("gemini_attempt", "").strip()
    if gemini_thinking and gemini_attempt:
        text = build_prompt(system_msg, question, gemini_thinking, gemini_attempt)
        samples.append({"text": text})
    
    # Process DeepSeek fields if available.
    deepseek_thinking = example.get("deepseek_thinking_trajectory", "").strip()
    deepseek_attempt = example.get("deepseek_attempt", "").strip()
    if deepseek_thinking and deepseek_attempt:
        text = build_prompt(system_msg, question, deepseek_thinking, deepseek_attempt)
        samples.append({"text": text})
    
    return samples

# Manually expand the dataset: each original example may yield up to two training samples.
processed_samples = []
for example in raw_dataset:
    processed_samples.extend(process_example(example))

# Create a new Dataset from the processed samples.
dataset = Dataset.from_list(processed_samples)

# Add code to compute and print the maximum token length in the dataset.
def compute_token_length(example):
    tokens = tokenizer(example["text"], truncation=False)["input_ids"]
    return {"token_length": len(tokens)}

dataset = dataset.map(compute_token_length)

max_length = max(dataset["token_length"])
mean_length = np.mean(dataset["token_length"])
std_length = np.std(dataset["token_length"])
print(f"Token length stat in the dataset: {max_length}, {mean_length}, {std_length}")


# Create a train/test split (90% train, 10% test).
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]
print("Train: ",len(train_dataset))
print("Test: ",len(test_dataset))

# Integrate LoRA adapters so that only a fraction of parameters are updated.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Set up training arguments.
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=3,  # Run for 3 full epochs
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
)

# Set up the SFTTrainer using the train dataset.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Enable packing for shorter sequences if desired.
    args=training_args,
)

def main():
    # Train the model.
    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds.")

    # Sample generation after training.
    print("Generating sample output from trained model...")
    FastLanguageModel.for_inference(model)  # Enable 2x faster inference.
    sample_prompt = (
        "[system] You are a reasoning model trained to reason about math, code, and other topics.\n"
        "[user] Solve the following problem: What is the derivative of sin(x)?\n"
        "<thinking> "
    )
    inputs = tokenizer([sample_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=7000, use_cache=True)
    generated_text = tokenizer.batch_decode(outputs)
    print("Generated response:")
    print(generated_text[0])

    # --- Saving the model and tokenizer ---
    save_path = "lora_model"
    print(f"Saving model and tokenizer to '{save_path}'...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model and tokenizer saved.")

    # --- Loading the saved model and tokenizer ---
    print("Loading the saved model and tokenizer for inference...")
    loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
        model_name=save_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(loaded_model)
    # Generate sample output using the loaded model.
    inputs = loaded_tokenizer([sample_prompt], return_tensors="pt").to("cuda")
    outputs = loaded_model.generate(**inputs, max_new_tokens=7000, use_cache=True)
    loaded_generated_text = loaded_tokenizer.batch_decode(outputs)
    print("Generated response from loaded model:")
    print(loaded_generated_text[0])

if __name__ == "__main__":
    main()
