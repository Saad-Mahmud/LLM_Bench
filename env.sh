#!/bin/bash
# Script: setup_llm_inference.sh

# Step 1: Create a new conda environment with Python 3.10
module load cuda/12.6
module load conda/latest
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export HF_HOME=/work/pi_shlomo_umass_edu/huggingface
echo "Creating conda environment 'nlp' with Python 3.10..."
conda create -y --name llm_inference python=3.10

# Step 2: Activate the new environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_inference

# Step 3: Install PyTorch and the CUDA toolkit (adjust cudatoolkit version if needed)
echo "Installing PyTorch and related packages..."
conda install cudatoolkit=11.7 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# Step 4: Install main LLM inference packages via pip
echo "Installing additional LLM inference packages..."
pip install transformers accelerate unsloth unsloth-zoo bitsandbytes 
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126 
pip install sentencepiece vllm trl flash-attn
pip install guidance
pip install --upgrade pydantic
conda clean --all

echo "Setup complete! Activate your environment with: conda activate llm_inference"
