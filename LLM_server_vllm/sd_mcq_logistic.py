import sys
import os
import contextlib
import numpy as np
import torch  # Used for computing log probabilities
from constrained_logits import ConstrainedLogitBias
# Import vLLM classes – adjust these imports per your vLLM installation
from vllm import LLMEngine
from vllm.engine import EngineArgs, Request as VLLMRequest, SamplingParams

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
# A simple vLLM wrapper class that provides generation, tokenization, 
# and (optionally) logits processor support. Replace the stub tokenizer 
# and decode with your actual model’s methods.
########################################################################
class VLLMModelWrapper:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        # If your engine comes with a tokenizer, assign it here.
        # For example: self.tokenizer = engine.tokenizer

    def tokenize(self, text, add_bos=True, special=True):
        """
        Stub tokenizer: convert the input text into a list of integer tokens.
        Replace this with your model’s actual tokenization.
        """
        # Here we simply convert each character to its ordinal value.
        return [ord(c) for c in text]

    def decode(self, tokens):
        """
        Stub decoder: converts a list of integer tokens back to text.
        Replace this with your model’s actual decoding logic.
        """
        return ''.join(chr(t) for t in tokens)

    def generate(self, prompt: str, max_tokens: int, temperature: float = 1.0,
                 n_samples: int = 1, logits_processor: list = None, stop: list = None):
        """
        Generate text from the prompt using vLLM.
        
        Parameters:
          prompt: The input text.
          max_tokens: Maximum number of tokens to generate.
          temperature: Sampling temperature.
          n_samples: Number of samples to generate (currently only 1 is handled).
          logits_processor: A list of callback functions to adjust logits.
          stop: A list of stop tokens to end generation.
        
        Returns:
          A dict with a 'choices' key whose value is a list of dicts containing 'text'.
          
        Note: This implementation is a simplified stub. In a full implementation, you might need
        to loop token-by-token applying logits_processor callbacks and checking for stop tokens.
        """
        sampling_params = SamplingParams(temperature=temperature)
        # In this stub, we ignore n_samples, logits_processor, and stop tokens.
        req = VLLMRequest(prompt=prompt, sampling_params=sampling_params, max_tokens=max_tokens)
        result = self.engine.submit(req)
        # Assume the result has an attribute .text with the generated text.
        return {'choices': [{'text': result.text}]}
    
########################################################################
# Logit selection function that uses the vLLM wrapper.
########################################################################
def logit_select(model, prompt, class_names):
    CLB = ConstrainedLogitBias("")
    logits = []
    for target_sequence in class_names:
        # Tokenize the target sequence.
        target_tokens = model.tokenize(target_sequence, add_bos=False, special=False)
        CLB.reset(target_tokens)
        # Generate exactly as many tokens as the length of the target.
        output = model.generate(
            prompt=prompt,
            max_tokens=len(target_tokens),
            temperature=1.0,
            logits_processor=[CLB.apply_bias]
        )
        generated_text = output['choices'][0]['text']
        # Ensure the generated text exactly matches the target sequence.
        assert generated_text == target_sequence, (
            f"Generated text '{generated_text}' does not match target '{target_sequence}'"
        )
        logits.append(CLB.nll)
    best_index = np.argmin(logits)
    return class_names[best_index], logits

########################################################################
# MCQ logit decoding function that uses vLLM for both chain-of-thought 
# generation and constrained logit selection.
########################################################################
def mcq_logit_decoding(model, vllm_model, prompt, class_names, temp=0.8,
                       cot=False, cot_tokens=2000, eof_tags=['</s>']):
    with suppress_stdout():
        cotans = ""
        if cot:
            # Generate chain-of-thought text using vLLM.
            cot_prompt = prompt + "\nAnswer: Let's think step-by-step. "
            ans = model.generate(
                prompt=cot_prompt,
                max_tokens=cot_tokens,
                temperature=temp,
                stop=eof_tags  # Assuming your generate method can handle stop tokens.
            )
            cotans = ans['choices'][0]['text']
            prompt = prompt + "\nAnswer: " + cotans + "\n In conclusion, the correct option to the given math problem is "
        else:
            prompt = prompt
        best_option, logits = logit_select(vllm_model, prompt, class_names)
        return {'ans': best_option, 'logits': logits, 'cot': cotans}

