from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time
import threading
from structered_decoding import structered_generation
from contextlib import asynccontextmanager

# Import vLLM classes
from vllm import LLMEngine
from vllm.engine import EngineArgs, Request as VLLMRequest, SamplingParams

# Create the FastAPI app and a threading lock
app = FastAPI()
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

# Define a simple wrapper to mimic the interface expected by your structured_generation
class VLLMModelWrapper:
    def __init__(self, engine: LLMEngine):
        self.engine = engine

    def generate(self, prompt: str, temperature: float, n_samples: int, max_tokens: int = 500, **kwargs):
        sampling_params = SamplingParams(temperature=temperature)
        # Note: vLLM may not support batch generation natively.
        # Here, we loop over n_samples to generate each sample individually.
        responses = []
        for _ in range(n_samples):
            req = VLLMRequest(prompt=prompt, sampling_params=sampling_params)
            result = self.engine.submit(req)
            responses.append(result)  # You might extract result.text or similar
        return responses

# Use the lifespan parameter to load and unload the model
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Update this path to point to your vLLM-compatible model
    model_path = "/work/pi_shlomo_umass_edu/saadprojects/GGUFs/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
    n_gpu_layers = -1  # Adjust GPU layer setting as needed

    # Create engine arguments; note that max_seq_len here is equivalent to n_ctx
    engine_args = EngineArgs(
        model=model_path,
        n_gpu_layers=n_gpu_layers,
        max_seq_len=16384,
        seed=1337,
    )
    engine = LLMEngine(engine_args)
    # Wrap the engine in our simple model interface
    model = VLLMModelWrapper(engine)
    # Store the model in app.state so it can be accessed in endpoints
    app.state.model = model
    app.state.engine = engine
    yield
    # Cleanup code (if any) goes here; for example, delete the model to free resources
    del app.state.model

# Pass the lifespan function to the FastAPI app
app = FastAPI(lifespan=lifespan)

# Define the API endpoint
@app.post("/generate")
async def generate(query: Query):
    start_time = time.time()
    with lock:
        if query.type in ["mcq_logits"]:
            # Your structured_generation function should now call app.state.model.generate(...)
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
