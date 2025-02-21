from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time
import asyncio

# Updated vLLM imports for GPU inference.
from vllm import SamplingParams, LLM
from structered_decoding import structered_generation

model_path = "/home/smahmud/Documents/saadprojects/GGUFs/Qwen2.5-3B-Instruct-Q8_0.gguf"  # update as needed
model = LLM(model_path, 
            max_seq_len_to_capture=12288,
            enable_prefix_caching=True)
tokenizer = model.get_tokenizer()


if __name__ == "__main__":
    text = model.generate("Tell me a long story.", SamplingParams(temperature=1.0, max_tokens=128))
    print(text[0].outputs[0].text)
    print(tokenizer.encode("Tell me a story."))