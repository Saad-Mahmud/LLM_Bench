from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from guidance import models
from llama_cpp import Llama
import uvicorn
import time
from structered_decoding import structered_generation
from contextlib import asynccontextmanager


class Query():
    prompt: str
    type: str
    class_names: list = []
    T: float = 1.0
    n_samples: int = 1
    cot: bool = False
    cot_tokens: int = 500
    eof_tags: list[str] = None

class AppState:

    def __init__(self):
        self.model = None
        self.llama_model = None

def generate(query: Query, state):
    start_time = time.time()
    if query.type in ["logistic","belief","plain", "mcq", "mcq_logits"]:
        output_text = structered_generation(state,  query)
    else:
        return JSONResponse(content={"error": "This feature has not been implemented yet."})
    tm = time.time() - start_time
    response = {
        "output": output_text,
        "class_names": query.class_names,
        "time_taken": tm
    }
    return response


if __name__ == "__main__":
    state = AppState()
    model_path = "/home/smahmud/Documents/saadprojects/GGUFs/Qwen2.5-32B-Instruct-Q4_K_M.gguf"  # Update this path
    n_layer = -1
    llama_model = Llama(
        model_path=model_path,
        verbose=False,
        n_gpu_layers=n_layer,  # Uncomment to use GPU acceleration
        seed=1337,             # Uncomment to set a specific seed
        n_ctx=4096,            # Uncomment to increase the context window
    )
    model = models.LlamaCpp(llama_model)
    # Store the model in app.state so it can be accessed in endpoints
    state.model = model
    state.llama_model = llama_model

    query = Query()
    query.prompt = "Once upon a time "
    query.class_names = ["there was a boy.", "there was a girl.", "there was a king.", "there was a queen."]
    query.type = "mcq_logits"
    query.cot = False
    query.eof_tags = ['</s>', '<|endoftext|>', '<|end|>']

    response = generate(query, state)
    print(response["output"], response["time_taken"])

    del state.model
    del state.llama_model

