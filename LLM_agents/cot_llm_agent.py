import requests
import warnings
from LLM_agents.utils import find_lowest_entropy_object, find_majority_object

class COTLLMAgent(object):

    def __init__(self, name ="generic", api_endpoint = "http://localhost:8000/generate", n_sample = 1, selection = "entropy"):
        self.api_endpoint = api_endpoint
        self.name = name
        self.n_sample = n_sample
        self.selecton = selection

    def answer(self, prompt, class_names, max_cot_tokens = 512, eof_tags = ['</s>', '<|endoftext|>', '<|end|>']):
        query = {
                "prompt": prompt,
                "type": "mcq_logits",
                "class_names": class_names,
                "eof_tags": eof_tags,
                "cot": True,
                "cot_tokens": max_cot_tokens
            }
        attempt = []
        server_request = 0
        response = None
        while len(attempt)<self.n_sample:
            response = requests.post("http://localhost:8000/generate", json=query)
            if response.status_code == 200:
                attempt.append(response)
                server_request+=1
            if server_request>self.n_sample*3:
                break
        if len(attempt) == 0:
            return response
        if len(attempt) == 1:
            return attempt[0]
        else:
            if self.selecton == "entropy":
                return find_lowest_entropy_object(attempt)
            elif self.selecton == "majority":
                return find_majority_object(attempt)
            else:
                warnings.warn(f"Selection criteri defaulted to return first", UserWarning)
                return attempt[0]



        