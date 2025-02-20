from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from guidance import models
from llama_cpp import Llama
import uvicorn
import time
import threading
from structered_decoding import structered_generation
from contextlib import asynccontextmanager

app = FastAPI()

# Create a threading lock
lock = threading.Lock()

# Define the request data model using Pydantic
class Query(BaseModel):
    prompt: str
    type: str
    class_names: list = []
    T: float = 1.0
    n_samples: int = 1
    cot: bool = False
    cot_tokens: int = 500
    eof_tags: list[str] = None

# Use the lifespan parameter to load and unload the model
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup
    model_path = "/home/smahmud/Documents/saadprojects/GGUFs/Llama-3.3-70B-Instruct-Q4_K_M.gguf"  # Update this path
    n_layer = -1
    llama_model = Llama(
        model_path=model_path,
        verbose=False,
        n_gpu_layers=n_layer,  # Uncomment to use GPU acceleration
        seed=1337,             # Uncomment to set a specific seed
        n_ctx=16384,            # Uncomment to increase the context window
    )
    model = models.LlamaCpp(llama_model)
    # Store the model in app.state so it can be accessed in endpoints
    app.state.model = model
    app.state.llama_model = llama_model
    yield
    # Cleanup code (if any) goes here
    # For example, you can delete the model to free up resources
    del app.state.model

# Pass the lifespan function to the FastAPI app
app = FastAPI(lifespan=lifespan)


# Define the API endpoint
@app.post("/generate")
async def generate(query: Query):
    start_time = time.time()
    with lock:
        if query.type in ["logistic", "belief", "plain", "mcq", "mcq_logits"]:
            output_text = structered_generation(app.state,  query)
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
