#!/bin/bash

set -x

# Environment setup
export DS_SKIP_CUDA_CHECK=1
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export CUDA_LAUNCH_BLOCKING=1

# Parameters
MODEL_PATH=""
MODEL_SIZE="7B"
DATA_NAME="sft-vl-demo"
NUM_NODES=1
NUM_GPUS=8
CUTOFF_LEN=20000
LEARNING_RATE=6e-5
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16

# Output directory
model_saved_name="demo-qwen25vl-${MODEL_SIZE}-len_${CUTOFF_LEN}-lr_${LEARNING_RATE}-data_${DATA_NAME}"
OUTPUT_DIR="./output/${MODEL_SIZE}/${model_saved_name}"
tensorboard_dir="${OUTPUT_DIR}/runs_${MODEL_SIZE}/${model_saved_name}"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${tensorboard_dir}

# SFT Training
deepspeed --hostfile=/etc/mpi.host src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path ${MODEL_PATH} \
    --dataset ${DATA_NAME} \
    --dataset_dir ./data \
    --template qwen2_vl \
    --finetuning_type full \
    --freeze_vision_tower True \
    --train_mm_proj_only False \
    --image_resolution 1048576 \
    --video_resolution 16384 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --overwrite_output_dir True \
    --cutoff_len ${CUTOFF_LEN} \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\"min_lr_rate\": 0.1}" \
    --num_train_epochs 1 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --logging_dir "./output/runs_${MODEL_SIZE}/${model_saved_name}" \
    --save_strategy epoch \
    --plot_loss True \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --use_unsloth_gc True \
    --bf16 \
    --flash_attn fa2 \
    --sequence_parallel_size 8 \
    --ddp_timeout 180000000 \
    --report_to tensorboard
