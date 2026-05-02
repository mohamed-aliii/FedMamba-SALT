#!/bin/bash
# =============================================================================
# run_fedavg_retina.sh -- Run FedAvg + FedProx experiments across 3 splits.
#
# Produces 6 pre-training checkpoints:
#   outputs/fedavg_split_{1,2,3}/ckpt_latest.pth   (FedAvg, mu=0)
#   outputs/fedprox_split_{1,2,3}/ckpt_latest.pth  (FedProx, mu=0.01)
#
# After pre-training, evaluate each checkpoint with:
#   python eval/linear_probe.py --ckpt outputs/fedavg_split_1/ckpt_latest.pth ...
# =============================================================================

set -e  # Stop on first error

CONFIG="configs/retina_fedavg.yaml"

for SPLIT in split_1 split_2 split_3; do
    echo ""
    echo "============================================================"
    echo "  FedAvg -- ${SPLIT}"
    echo "============================================================"
    python train_fedavg.py \
        --config "${CONFIG}" \
        --split_type "${SPLIT}" \
        --mu 0.0 \
        --output_dir "outputs/fedavg_${SPLIT}"

    echo ""
    echo "============================================================"
    echo "  FedProx (mu=0.01) -- ${SPLIT}"
    echo "============================================================"
    python train_fedavg.py \
        --config "${CONFIG}" \
        --split_type "${SPLIT}" \
        --mu 0.01 \
        --output_dir "outputs/fedprox_${SPLIT}"
done

echo ""
echo "============================================================"
echo "  All 6 experiments complete."
echo "  FedAvg checkpoints:  outputs/fedavg_split_{1,2,3}/"
echo "  FedProx checkpoints: outputs/fedprox_split_{1,2,3}/"
echo "============================================================"
