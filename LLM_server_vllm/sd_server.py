from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time
import asyncio
import torch
# Updated vLLM imports for GPU inference.
from vllm import SamplingParams, LLM
from structered_decoding import structered_generation

# Define the request data model.
class Query(BaseModel):
    prompt: str
    type: str
    class_names: list = []
    T: float = 1.0
    n_samples: int = 1
    cot: bool = False
    cot_tokens: int = 500
    eof_tags: list[str] = None


app = FastAPI()

@app.on_event("startup")
def startup_event():
    model_id = "unsloth/tinyllama-bnb-4bit"
    llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True, \
                quantization="bitsandbytes", load_format="bitsandbytes")
    app.state.tokenizer = app.state.model.get_tokenizer()

# Shutdown event to clean up if necessary.
@app.on_event("shutdown")
def shutdown_event():
    del app.state.model

# API endpoint for generating responses.
@app.post("/generate")
def generate(query: Query):
    start_time = time.time()
    if query.type in ["mcq_logits"]:
        # structured_generation is expected to use app.state.model.generate(...)
        output_text = structered_generation(app.state, query)
    else:
        return JSONResponse(content={"error": "This feature has not been implemented yet."})
    tm = time.time() - start_time
    response = {
        "output": output_text['ans'],
        "logits": output_text['logits'],
        "class_names": query.class_names,
        "time_taken": tm
    }
    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run("sd_server:app", host="0.0.0.0", port=8000)
