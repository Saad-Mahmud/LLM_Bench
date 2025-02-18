from guidance import gen,select,with_temperature
import sys,os
import contextlib

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def mcq_decoding(model, prompt, class_names, temp = 0.8, cot = False, cot_tokens = 2000, eof_tags = None):
    with suppress_stdout():
        cotans = ""
        if cot:
            if eof_tags is not None:
                ans = model+prompt+ "\nAnswer: Let's think step-by-step. "+ gen(max_tokens=cot_tokens,
                                                                                 temperature = temp, name = 'text',stop=eof_tags)
            else:
                ans = model+prompt+ "\nAnswer: Let's think step-by-step. "+ gen(max_tokens=cot_tokens,
                                                                                 temperature = temp, name = 'text')
            cotans = ans["text"]

        if cot:      
            prompt = prompt + "\nAnswer: "+cotans+"\n In conclusion, the correct option to the given math problem is "
        else:
            prompt = prompt + "\nAnswer: The correct option to the given math problem is "
        
        ans = model+prompt+ select(class_names, name='ans')
        return ans['ans'], cotans