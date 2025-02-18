from guidance import gen,select,with_temperature
import sys,os
import contextlib
from constrained_logits import ConstrainedLogitBias
import numpy as np


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout



def logit_select(model, prompt, class_names):
    CLB = ConstrainedLogitBias("")
    logits = []
    for target_sequence in class_names:
        target_tokens = model.tokenize(target_sequence.encode("utf-8"),add_bos=False, special=False)
        CLB.reset(target_tokens)
        #print(target_tokens)
        output = model(
                    prompt,
                    logits_processor=[CLB.apply_bias],
                    max_tokens = len(target_tokens))
        #print(output['choices'][0]['text'])
        #print(target_sequence)
        #print(model.tokenize(output['choices'][0]['text'].encode("utf-8"),add_bos=False, special=False))
        assert output['choices'][0]['text'] == target_sequence
        logits.append(CLB.nll)
    return class_names[np.argmin(logits)], logits

def mcq_logit_decoding(model, llama_model, prompt, class_names, temp = 0.8, cot = False, cot_tokens = 2000, eof_tags = ['</s>']):
    with suppress_stdout():
        cotans = ""
        if cot:
            ans = model+prompt+ "\nAnswer: Let's think step-by-step. "+ gen(max_tokens=cot_tokens,
                                                                            temperature = temp, name = 'text', stop = eof_tags)
            cotans = ans["text"]     
            prompt = prompt + "\nAnswer: "+cotans+"\n In conclusion, the correct option to the given math problem is "
        else:
            #prompt = prompt + "\nAnswer: The correct option to the given math problem is "
            prompt = prompt 
            
        ans = logit_select(llama_model, prompt, class_names)
        return {'ans': ans[0], 'logits': ans[1], 'cot': cotans}