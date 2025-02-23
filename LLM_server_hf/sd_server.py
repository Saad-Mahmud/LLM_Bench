from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time
import torch
from unsloth import FastLanguageModel
from structured_decoding import structured_generation

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
    model_id = "unsloth/Qwen2.5-72B-Instruct-bnb-4bit"
    max_seq_length = 16384  # Adjust as needed for your use case.
    # Load the model and tokenizer using Unsloth.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    # Enable Unsloth's native 2× faster inference mode.
    FastLanguageModel.for_inference(model)
    app.state.model = model
    app.state.tokenizer = tokenizer

@app.on_event("shutdown")
def shutdown_event():
    del app.state.model

@app.post("/generate")
def generate(query: Query):
    start_time = time.time()
    if query.type in ["mcq_logits"]:
        output_text = structured_generation(app.state, query)
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
