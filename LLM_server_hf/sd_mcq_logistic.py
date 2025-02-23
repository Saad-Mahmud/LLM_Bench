import sys
import os
import contextlib
import numpy as np
import torch  # Used for computing log probabilities
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
# Import your constrained logit bias module. Ensure that ConstrainedLogitBias.apply_bias is callable as a logits processor.
from constrained_logits import ConstrainedLogitBias

# A simple context manager to suppress unwanted stdout (if desired)
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

########################################################################
# A simple Hugging Face wrapper class that provides generation, tokenization,
# and (optionally) logits processor support.
########################################################################
class HFModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def tokenize(self, text, add_bos=True, special=True):
        # Use the Hugging Face tokenizer.
        return self.tokenizer.encode(text, add_special_tokens=add_bos)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate(self, prompt: str, max_tokens: int, temperature: float = 1.0,
                 n_samples: int = 1, logits_processor: list = None, stop: list = None):
        # Tokenize the prompt.
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # Setup the logits processors if provided.
        if logits_processor is not None:
            logits_processor = LogitsProcessorList(logits_processor)
        # Call generate with the Hugging Face API.
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            num_return_sequences=n_samples,
            logits_processor=logits_processor
        )
        # For a single sample, decode the first sequence.
        if n_samples == 1:
            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return {'choices': [{'text': text}]}
        else:
            texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            return {'choices': [{'text': t} for t in texts]}

########################################################################
# Logit selection function that uses the HF wrapper.
########################################################################
def logit_select(model_wrapper: HFModelWrapper, prompt, class_names):
    logits = []
    for target_sequence in class_names:
        # Tokenize the target sequence.
        target_tokens = model_wrapper.tokenizer.encode(target_sequence, add_special_tokens=False)
        # Instantiate and reset the constrained logit bias.
        CLB = ConstrainedLogitBias("")
        CLB.reset(target_tokens)
        # Create a logits processor list with our custom bias function.
        logits_processors = [CLB.apply_bias]
        # Generate exactly as many tokens as in the target sequence.
        output = model_wrapper.generate(
            prompt, 
            max_tokens=len(target_tokens),
            temperature=1.0,
            logits_processor=logits_processors
        )
        full_text = output['choices'][0]['text']
        generated_text = full_text[len(prompt):]
        # Ensure the generated text exactly matches the target sequence.
        assert generated_text == target_sequence, (
            f"Generated text '{generated_text}' does not match target '{target_sequence}'"
        )
        logits.append(CLB.nll)
    best_index = np.argmin(logits)
    return class_names[best_index], logits

########################################################################
# MCQ logit decoding function that uses Hugging Face for both chain-of-thought
# generation and constrained logit selection.
########################################################################
def mcq_logit_decoding(model, tokenizer, prompt, class_names, temp=0.8, cot=False, cot_tokens=2000, eof_tags=['</s>']):
    model_wrapper = HFModelWrapper(model, tokenizer)
    with suppress_stdout():
        best_option, logits = logit_select(model_wrapper, prompt, class_names)
        return {'ans': best_option, 'logits': logits, 'cot': ""}