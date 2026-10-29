#!/usr/bin/env bash
# Script chạy liên tục 8 thuật toán baseline và log lên WandB
set -e

DATASET=${1:-"Cifar10"}
NUM_CLIENTS=${2:-20}
NUM_CLASSES=${3:-10}
GLOBAL_ROUNDS=${4:-200}
MODEL=${5:-"cnn"}
DEVICE=${6:-"cuda"}
WANDB_PROJECT=${7:-"FedSCD-Benchmark"}
WANDB_ENTITY=${8:-"gvquang25-hanoi-university-of-science-and-technology"}
WANDB_GROUP=${9:-"baselines-cifar10"}

PYTHON_BIN="../.venv/Scripts/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

ALGOS=("FedTGP" "FedSA" "FedPLVM" "FedImpro" "FedCOF" "Fed3R" "FedORGP" "FedGMKD")

echo "=========================================================================="
echo " BẮT ĐẦU CHẠY LIÊN TỤC 8 THUẬT TOÁN VÀ LOG LÊN WANDB"
echo " Dataset       : $DATASET ($NUM_CLIENTS clients, $NUM_CLASSES classes)"
echo " Rounds        : $GLOBAL_ROUNDS"
echo " WandB Project : $WANDB_PROJECT"
echo " WandB Entity  : $WANDB_ENTITY"
echo " Danh sách     : ${ALGOS[*]}"
echo "=========================================================================="

for ALGO in "${ALGOS[@]}"; do
    echo -e "\n\033[1;32m===> Đang chạy thuật toán: $ALGO <===\033[0m"
    
    EXTRA_ARGS=""
    if [ "$ALGO" == "FedTGP" ]; then
        EXTRA_ARGS="-lam 10 -se 100 -mart 100"
    fi

    $PYTHON_BIN -u main.py \
        -algo "$ALGO" \
        -data "$DATASET" \
        -m "$MODEL" \
        -nc "$NUM_CLIENTS" \
        -nb "$NUM_CLASSES" \
        -gr "$GLOBAL_ROUNDS" \
        -ls 1 \
        -lbs 10 \
        -lr 0.005 \
        --noniid \
        --alpha_dirich 0.1 \
        -dev "$DEVICE" \
        -did 0 \
        -log \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_group "$WANDB_GROUP" \
        $EXTRA_ARGS
done

echo "=========================================================================="
echo " ĐÃ HOÀN THÀNH TẤT CẢ 8 THUẬT TOÁN!"
echo " Truy cập WandB: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "=========================================================================="
