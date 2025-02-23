from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time
import asyncio
import torch
# Updated vLLM imports for GPU inference.
from vllm import SamplingParams, LLM
from LLM_server_hf.structured_decoding import structered_generation

model_id = "unsloth/tinyllama-bnb-4bit"
llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True, \
                quantization="bitsandbytes", load_format="bitsandbytes")
tokenizer = llm.get_tokenizer()


if __name__ == "__main__":
    text = llm.generate("Tell me a long story.", SamplingParams(temperature=1.0, max_tokens=128))
    print(text[0].outputs[0].text)
    print(tokenizer.encode("Tell me a story."))