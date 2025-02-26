#!/bin/bash
#SBATCH --job-name=EvalGen
#SBATCH -t 32:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --constraint="a100"
#SBATCH --output=Gen_log.txt

# Load required modules
module load cuda/12.6
module load conda/latest

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_inference
source activate llm_inference

# Set CUDA and Python paths (if needed)
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$(conda info --base)/envs/llm_inference/bin:$PATH"
export HF_HOME=/work/pi_shlomo_umass_edu/huggingface
# Debug information
echo "CUDA Version:"
nvcc --version
echo "Python Path: $(which python)"
echo "Accelerate Version:"
python -c "import accelerate; print(accelerate.__version__)"
echo "Torch Version:"
python -c "import torch; print(torch.__version__)"

# Run the training script
echo "Here we go!"
# Start the server in the background.
python EvalGen.py 

# Optionally, after EvalGen finishes, kill the server if it's no longer needed.

echo "Gen complete!"