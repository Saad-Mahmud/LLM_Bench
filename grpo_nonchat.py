import re
from typing import Optional, List, Union
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# -----------------------------------------------------------------------------
# Define system instructions and XML formatting (for generation/formatting purposes)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Respond in the following format:\n"
    "<reasoning>\n"
    "...\n"
    "</reasoning>\n"
    "<answer>\n"
    "...\n"
    "</answer>"
)

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# -----------------------------------------------------------------------------
# Helper functions for extracting answers
# -----------------------------------------------------------------------------
def extract_xml_answer(text: str) -> str:
    """Extract the answer contained in <answer>...</answer> tags."""
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except Exception:
        return ""

def extract_hash_answer(text: str) -> Optional[str]:
    """Extract the answer if it is separated by '####'."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# -----------------------------------------------------------------------------
# Prepare the GSM8K dataset
# -----------------------------------------------------------------------------
def get_gsm8k_questions(split: str = "train") -> Dataset:
    """
    Loads the gsm8k dataset and converts each example into a plain-text prompt.
    The prompt now is a string combining the system instructions and the question.
    """
    data = load_dataset("openai/gsm8k", "main")[split]
    # Convert the example to a simple prompt string rather than a chat-message list.
    data = data.map(lambda x: {
        "prompt": f"{SYSTEM_PROMPT}\n{x['question']}",
        "answer": extract_hash_answer(x["answer"])
    })
    return data

# -----------------------------------------------------------------------------
# Reward functions
# -----------------------------------------------------------------------------
def correctness_reward_func(prompts: List[str], completions: Union[List[str], str], 
                          answer: List[str], **kwargs) -> List[float]:
    """
    Compare the extracted answer from the model output to the target answer.
    """
    responses = completions if isinstance(completions, list) else [completions]
    q = prompts[0]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"\nQuestion:\n{q}",
        f"\nTarget Answer:\n{answer[0]}",
        f"\nModel Response:\n{responses[0]}",
        f"\nExtracted Answer:\n{extracted_responses[0]}"
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions: Union[List[str], str], **kwargs) -> List[float]:
    responses = completions if isinstance(completions, list) else [completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions: Union[List[str], str], **kwargs) -> List[float]:
    """Check if the output exactly matches the expected XML format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = completions if isinstance(completions, list) else [completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions: Union[List[str], str], **kwargs) -> List[float]:
    """Check if the output roughly matches the XML format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = completions if isinstance(completions, list) else [completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions: Union[List[str], str], **kwargs) -> List[float]:
    contents = completions if isinstance(completions, list) else [completions]
    return [count_xml(c) for c in contents]

# -----------------------------------------------------------------------------
# Main training script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = get_gsm8k_questions()
    model_name = "Qwen/Qwen2.5-1.5B"
    output_dir = "outputs/Qwen2.5-1.5B-GRPO"
    run_name = "Qwen/Qwen2.5-1.5B-GRPO-gsm8k"
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=256,
        num_train_epochs=1,
        save_steps=1500,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.35,
        vllm_device="cuda:0",
        report_to="none"  # disable logging to Wandb, etc.
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------------
    # Override the chat template function
    # -----------------------------------------------------------------------------
    def simple_apply_chat_template(example: dict, tools: Optional[dict] = None) -> dict:
        # If the prompt is a list (from an older chat format), join the message texts.
        prompt = example["prompt"]
        if isinstance(prompt, list):
            prompt = "\n".join([m["content"] for m in prompt])
        return {"prompt": prompt}

    tokenizer.apply_chat_template = simple_apply_chat_template

    # -----------------------------------------------------------------------------
    # Instantiate and run the GRPOTrainer
    # -----------------------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,  # now with our simple apply_chat_template override
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()