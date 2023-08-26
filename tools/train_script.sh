#!/bin/sh

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
BATCH_SIZE=8
NUM_EPOCHS=1
EXP_NAME="test"

echo "Number of GPUS: $NUM_GPUS"

# CFG_FILE="cfgs/waymo/mtr+20_percent_data.yaml"
CFG_FILE="cfgs/waymo/ours+10_percent_data.yaml"

bash scripts/torchrun_train.sh ${NUM_GPUS} --cfm --cfg_file ${CFG_FILE} --batch_size ${BATCH_SIZE} --epochs ${NUM_EPOCHS} --extra_tag ${EXP_NAME}
