#!/bin/bash
#================================================================
#  SLURM DIRECTIVES
#================================================================
#SBATCH --job-name=humayun
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodelist=node2
#SBATCH --gres=shard:30
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G

#================================================================
#  ENVIRONMENT SETUP
#================================================================
set -euo pipefail

echo "================================================================"
echo "  JOB STARTED"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $(hostname)"
echo "  Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Load system modules
module load anaconda3-2024.2
module load cuda-12.8

# Activate existing virtual environment
source ~/envs/humayun/bin/activate
echo "[OK] Virtual env activated: $(which python)"

# Project root
PROJECT_DIR=~/hpc-prog/humayun/transformer_mpc
cd "$PROJECT_DIR"
echo "[OK] Working directory: $(pwd)"

#----------------------------------------------------------------
#  Install requirements (safe to re-run, skips if already done)
#----------------------------------------------------------------
echo ""
echo "[SETUP] Checking/Installing requirements..."
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
pip install --quiet -r requirements.txt
echo "[OK] All packages ready."

#----------------------------------------------------------------
#  Create .env if it does not exist
#----------------------------------------------------------------
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[OK] .env file created from .env.example"
else
    echo "[OK] .env file already exists"
fi

#----------------------------------------------------------------
#  Verify GPU
#----------------------------------------------------------------
echo ""
echo "[CHECK] GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
python -c "
import torch
print('  PyTorch version : ' + torch.__version__)
print('  CUDA available  : ' + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print('  GPU             : ' + torch.cuda.get_device_name(0))
else:
    print('  WARNING: CUDA not detected - running on CPU!')
"

#================================================================
#  PIPELINE
#================================================================

# --- STEP 1: Smoke Test ---
echo ""
echo "================================================================"
echo "  STEP 1/5 - Smoke Test"
echo "================================================================"
python tests/test_smoke.py
echo "[OK] Smoke test passed."

# --- STEP 2: Generate Dataset ---
echo ""
echo "================================================================"
echo "  STEP 2/5 - Dataset Generation"
echo "================================================================"
python scripts/run_generate_data.py
echo "[OK] Dataset saved to: $PROJECT_DIR/data/"

# --- STEP 3: Pre-Training ---
echo ""
echo "================================================================"
echo "  STEP 3/5 - Transformer Pre-Training"
echo "================================================================"
python scripts/run_pretrain.py
echo "[OK] Pretrained model saved to: $PROJECT_DIR/models/pretrained.pt"

# --- STEP 4: DAGGER Fine-Tuning ---
echo ""
echo "================================================================"
echo "  STEP 4/5 - DAGGER Fine-Tuning"
echo "================================================================"
python scripts/run_finetune.py
echo "[OK] Fine-tuned model saved to: $PROJECT_DIR/models/finetuned.pt"

# --- STEP 5: Evaluate + Visualize ---
echo ""
echo "================================================================"
echo "  STEP 5/5 - Evaluation + Plot Generation"
echo "================================================================"
python scripts/run_evaluate.py
python scripts/run_visualize.py
echo "[OK] Results and plots saved to: $PROJECT_DIR/figures/"

#================================================================
#  DONE
#================================================================
echo ""
echo "================================================================"
echo "  ALL STEPS COMPLETED SUCCESSFULLY!"
echo "  End Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Output locations:"
echo "    Dataset  -> $PROJECT_DIR/data/"
echo "    Models   -> $PROJECT_DIR/models/"
echo "    Logs     -> $PROJECT_DIR/logs/"
echo "    Figures  -> $PROJECT_DIR/figures/"
echo "================================================================"