import sys
import os
import contextlib
import numpy as np
import torch  # Used for computing log probabilities
from constrained_logits import ConstrainedLogitBias
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs

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
# and (optionally) logits processor support.
#
# Note: In recent versions of vLLM the API no longer exposes a separate
# Request class. Instead, you can directly call engine.generate().
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
          
        Note: In this simplified stub we ignore n_samples, logits_processor, and stop.
        """
        sampling_params = SamplingParams(temperature=temperature,
                                         logits_processors=logits_processor)
        # Call the engine's new generate() method directly.
        result = self.engine.generate(prompt, sampling_params, max_tokens)
        return {'choices': [{'text': result.text}]}


########################################################################
# Logit selection function that uses the vLLM wrapper.
########################################################################
def logit_select(model, tokenizer, prompt, class_names):
    CLB = ConstrainedLogitBias("")
    logits = []
    for target_sequence in class_names:
        # Tokenize the target sequence.
        target_tokens = tokenizer.encode(target_sequence)
        CLB.reset(target_tokens)
        # Generate exactly as many tokens as the length of the target.
        sp = SamplingParams(
                        max_tokens=len(target_tokens),
                        temperature=1.0,
                        logits_processors=[CLB.apply_bias])

        output = model.generate(prompt,sp)
        generated_text = output[0].outputs[0].text
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
def mcq_logit_decoding(model, tokenizer, prompt, class_names, temp=0.8,
                       cot=False, cot_tokens=2000, eof_tags=['</s>']):
    with suppress_stdout():
        cotans = ""
        if cot:
            # Generate chain-of-thought text using vLLM.
            cot_prompt = prompt + "\nAnswer: Let's think step-by-step. "
            ans = model.generate(cot_prompt,
                                SamplingParams(max_tokens=cot_tokens,
                                temperature=temp))
            cotans = ans[0].outputs[0].text
            prompt = prompt + "\nAnswer: " + cotans + "\n In conclusion, the correct option to the given math problem is "
        best_option, logits = logit_select(model, tokenizer, prompt, class_names)
        return {'ans': best_option, 'logits': logits, 'cot': cotans}
