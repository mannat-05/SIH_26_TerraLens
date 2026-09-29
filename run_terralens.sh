#!/bin/bash
#================================================================
#  SLURM DIRECTIVES  (must be the first thing after the shebang!)
#================================================================
#SBATCH --job-name=terralens
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=shard:5
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
# Optional: pin to a node (only if you really need to; leaving it out lets the job start sooner)
##SBATCH --nodelist=node2

#================================================================
#  ENVIRONMENT SETUP
#================================================================
set -eo pipefail
export PYTHONUNBUFFERED=1

echo "================================================================"
echo "  JOB STARTED"
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  Node     : $(hostname)"
echo "  Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

module load anaconda3-2024.2 2>/dev/null || true
module load cuda-12.8 2>/dev/null || true

# Optional: activate a conda env that has torch, numpy, pillow, tifffile
#   submit with:  CONDA_ENV=terralens sbatch run_terralens.sh
if [ -n "${CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "[OK] Conda env activated: $CONDA_ENV ($(which python))"
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
echo "[OK] Working directory: $(pwd)"

export DATA_DIR="$PROJECT_DIR/s2_naip_dataset"

#----------------------------------------------------------------
#  Install requirements (compute nodes may have no internet; failures are non-fatal)
#----------------------------------------------------------------
if [ -f "requirements.txt" ]; then
    echo "[SETUP] Installing requirements..."
    pip install --quiet -r requirements.txt || echo "[WARN] pip install failed (no internet?) - continuing"
fi

#----------------------------------------------------------------
#  Verify dataset (download on the LOGIN node beforehand if the compute node is offline):
#    pip install kaggle   # + put kaggle.json in ~/.kaggle/
#    kaggle datasets download -d evgeniyivankin/s2-naip-5k-pairs-rgb-nir -p s2_naip_dataset --unzip
#----------------------------------------------------------------
N_FILES=$( (find "$DATA_DIR" -type f \( -name '*.npy' -o -name '*.tif' -o -name '*.tiff' -o -name '*.png' \) 2>/dev/null || true) | wc -l )
if [ "$N_FILES" -eq 0 ]; then
    if command -v kaggle >/dev/null 2>&1; then
        echo "[DATA] Dataset missing - trying Kaggle download..."
        mkdir -p "$DATA_DIR"
        kaggle datasets download -d evgeniyivankin/s2-naip-5k-pairs-rgb-nir -p "$DATA_DIR" --unzip
    else
        echo "[ERROR] No dataset files in $DATA_DIR and 'kaggle' CLI not found. Download it on the login node first."
        exit 1
    fi
fi
echo "[OK] Dataset files found: $(find "$DATA_DIR" -type f \( -name '*.npy' -o -name '*.tif' -o -name '*.tiff' -o -name '*.png' \) | wc -l)"

#----------------------------------------------------------------
#  Verify GPU
#----------------------------------------------------------------
echo ""
echo "[CHECK] GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -c "
import torch
print('  PyTorch version : ' + torch.__version__)
print('  CUDA available  : ' + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print('  GPU             : ' + torch.cuda.get_device_name(0))
else:
    raise SystemExit('ERROR: CUDA not detected - aborting instead of training on CPU')
"

#================================================================
#  PIPELINE
#================================================================
echo ""
echo "================================================================"
echo "  STEP 1/1 - TerraLens Spatial-Frequency Training"
echo "================================================================"
python -u train_terralens.py
echo "[OK] Training phase completed successfully."

echo ""
echo "================================================================"
echo "  ALL STEPS COMPLETED SUCCESSFULLY!"
echo "  End Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Checkpoints -> $PROJECT_DIR/checkpoints/"
echo "================================================================"
