#!/bin/bash
# ============================================================================
# FedMamba-SALT PEFT-FedLC Ablation Runs
# ============================================================================
# Objective: Calibrate intrinsic LoRA rank and FedLC temperature (tau)
# Dataset: COVID-FL (12 Clients, Extreme Label Skew)
# ============================================================================

ENCODER_CKPT="/content/drive/MyDrive/fedmamba_salt_COVID/outputs/fedavg_split_real/ckpt_latest.pth"
DATA_PATH="federated/12_clients/split_real"
OUTPUT_BASE="/content/drive/MyDrive/fedmamba_salt_COVID/outputs/peft_fedlc_ablations"

# Common hyperparameters
N_CLIENTS=12
ROUNDS=50
EPOCHS=1
BATCH_SIZE=64
LR=0.001

echo "Starting PEFT-FedLC Ablations..."

# ----------------------------------------------------------------------------
# 1. Rank Sensitivity Ablation (Fixed Tau = 1.0)
# ----------------------------------------------------------------------------
for RANK in 4 8 16; do
    echo "Running LoRA Rank Ablation: r=${RANK}"
    python train_fed_finetune.py \
        --mode peft_fedlc \
        --encoder_ckpt ${ENCODER_CKPT} \
        --data_path ${DATA_PATH} \
        --dataset covidfl \
        --num_classes 3 \
        --n_clients ${N_CLIENTS} \
        --max_rounds ${ROUNDS} \
        --E_epoch ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --lora_rank ${RANK} \
        --fedlc_tau 1.0 \
        --output_dir ${OUTPUT_BASE}/rank_${RANK}_tau_1.0
done

# ----------------------------------------------------------------------------
# 2. Tau Calibration Ablation (Fixed Rank = 8)
# ----------------------------------------------------------------------------
for TAU in 0.5 2.0; do
    echo "Running FedLC Tau Ablation: tau=${TAU}"
    python train_fed_finetune.py \
        --mode peft_fedlc \
        --encoder_ckpt ${ENCODER_CKPT} \
        --data_path ${DATA_PATH} \
        --dataset covidfl \
        --num_classes 3 \
        --n_clients ${N_CLIENTS} \
        --max_rounds ${ROUNDS} \
        --E_epoch ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --lora_rank 8 \
        --fedlc_tau ${TAU} \
        --output_dir ${OUTPUT_BASE}/rank_8_tau_${TAU}
done

echo "Ablations complete. Check ${OUTPUT_BASE} for results."
