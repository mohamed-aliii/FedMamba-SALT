#!/usr/bin/env bash
# =============================================================================
# run_centralized_retina.sh -- Launch full centralized Retina experiment
#
# Workflow:
#   1. Activate conda environment
#   2. Run end-to-end smoke test (abort on failure)
#   3. Launch centralized SALT pre-training
#   4. Run linear probe evaluation on the final checkpoint
#
# Usage:
#   bash scripts/run_centralized_retina.sh
#
# Prerequisite:
#   - conda environment "fedmamba" must exist
#   - Retina dataset at DATA_PATH with train/ and test/ subdirectories
#   - MAE ViT-B/16 checkpoint at TEACHER_CKPT
# =============================================================================

set -euo pipefail  # exit on error, undefined vars, pipe failures

# ---- Configuration ----
CONDA_ENV="fedmamba"
DATA_PATH="/path/to/retina"                    # <-- CHANGE THIS
TEACHER_CKPT="data/ckpts/mae_vit_base.pth"
OUTPUT_DIR="outputs/retina_centralized"
NUM_CLASSES=5

# Training hyperparameters
EPOCHS=100
BATCH_SIZE=128
LR=1e-3
WEIGHT_DECAY=0.05
NUM_WORKERS=4
SAVE_EVERY=10

# Evaluation hyperparameters
EVAL_EPOCHS=50
EVAL_BATCH_SIZE=256
EVAL_LR=1e-3

echo "============================================================"
echo "  FedMamba-SALT: Centralized Retina Experiment"
echo "============================================================"
echo "  Data:      ${DATA_PATH}"
echo "  Teacher:   ${TEACHER_CKPT}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Epochs:    ${EPOCHS}"
echo "  Batch:     ${BATCH_SIZE}"
echo ""

# ---- Step 0: Activate environment ----
echo "[Step 0] Activating conda environment: ${CONDA_ENV}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# ---- Step 1: Smoke test ----
echo "[Step 1] Running end-to-end smoke test..."
python -m tests.test_end_to_end
if [ $? -ne 0 ]; then
    echo ""
    echo "[ABORT] Smoke test FAILED. Fix issues before training."
    exit 1
fi
echo ""

# ---- Step 2: Pre-training ----
echo "[Step 2] Launching centralized SALT pre-training..."
python train_centralized.py \
    --data_path "${DATA_PATH}" \
    --teacher_ckpt "${TEACHER_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_workers ${NUM_WORKERS} \
    --save_every ${SAVE_EVERY}

echo ""
echo "[Step 2] Pre-training complete."
echo ""

# ---- Step 3: Linear probe evaluation ----
FINAL_CKPT="${OUTPUT_DIR}/ckpt_latest.pth"
EVAL_OUTPUT="${OUTPUT_DIR}/eval_linear_probe"

echo "[Step 3] Running linear probe evaluation..."
python -m eval.linear_probe \
    --encoder_ckpt "${FINAL_CKPT}" \
    --data_path "${DATA_PATH}" \
    --num_classes ${NUM_CLASSES} \
    --output_dir "${EVAL_OUTPUT}" \
    --epochs ${EVAL_EPOCHS} \
    --batch_size ${EVAL_BATCH_SIZE} \
    --lr ${EVAL_LR} \
    --mode linear_probe

echo ""

# ---- Step 4: Full fine-tuning evaluation ----
FINETUNE_OUTPUT="${OUTPUT_DIR}/eval_full_finetune"

echo "[Step 4] Running full fine-tuning evaluation..."
python -m eval.linear_probe \
    --encoder_ckpt "${FINAL_CKPT}" \
    --data_path "${DATA_PATH}" \
    --num_classes ${NUM_CLASSES} \
    --output_dir "${FINETUNE_OUTPUT}" \
    --epochs ${EVAL_EPOCHS} \
    --batch_size ${EVAL_BATCH_SIZE} \
    --lr ${EVAL_LR} \
    --mode full_finetune

echo ""
echo "============================================================"
echo "  Experiment complete."
echo "  Checkpoints:       ${OUTPUT_DIR}/"
echo "  Linear probe:      ${EVAL_OUTPUT}/"
echo "  Full fine-tune:    ${FINETUNE_OUTPUT}/"
echo "============================================================"
echo ""
echo "  SUCCESS CRITERIA:"
echo "    1. Final SALT loss < 0.5"
echo "    2. embedding_std never dropped below 0.05"
echo "    3. Linear probe accuracy within 5% of Fed-MAE baseline (77.43%)"
echo ""
