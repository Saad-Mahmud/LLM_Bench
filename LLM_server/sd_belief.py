from guidance import gen,select,with_temperature
import sys,os
import contextlib
import random
import copy
import numpy as np
import guidance
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def _belief_decoding(model, _prompt, class_names, temp = 0.8, cot = False):
    with suppress_stdout():
        b = {}
        txtans = ""
        prompt = copy.deepcopy(_prompt)
        if cot:
            for cl in class_names:
                prompt += f"\n{cl}: "
            ans = model+prompt+ "\nAnswer: Let's think step-by-step. "+ gen(max_tokens=1000, temperature = temp, name = 'text')
            prompt = prompt + "\nAnswer: "+ans["text"]+"\n So the final answer is: "
            txtans = ans["text"]
            print(prompt)
        pm = 1.0+len(class_names)/100.0
        for cl in class_names:
            num = []
            for i in range(9):
                for j in range(9):
                    if float(f"0.{i}{j}")<=pm:
                        num.append(f"0.{i}{j}")
            prompt+=f"\n{cl}: "
            if len(num)==0:
                num = ["0.00"]
            ans = model+prompt+with_temperature(select(num, name=cl),temperature=temp)    
            b[cl] = float(ans[cl])
            pm-=b[cl]
            prompt+=ans[cl]
    return b,txtans


def belief_decoding(model, prompt, class_names, n_samples = 5, temp = 0.8, tb = False):
        B = {cl: 0.0 for cl in class_names}
        TA = []
        for sample in range(n_samples):
            random.shuffle(class_names)

            b,txtans = _belief_decoding(model,prompt,class_names,temp,tb)
            for c in b:
                B[c]+=b[c]/n_samples
            TA.append(txtans)
        tot = np.sum([v+1e-6 for k,v in B.items()])
        B = {k:(v+1e-6)/tot for k,v in B.items()}
        return B,TA

