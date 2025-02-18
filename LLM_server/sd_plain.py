from guidance import gen
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

def plain_decoding(model, prompt, temp = 0.8):
    with suppress_stdout():
        ans = model+prompt+ gen(max_tokens=2000, temperature = temp, name = 'text')
        return ans['text']