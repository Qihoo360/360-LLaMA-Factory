# 参考文档：https://llamafactory.readthedocs.io/zh-cn/latest/advanced/distributed.html
set -x
## environment
source /home/.bashrc
source /home/miniconda3/etc/profile.d/conda.sh
conda activate 360-llama-factory-higher-tf-new
which python

export FORCE_TORCHRUN=1
export CUDA_LAUNCH_BLOCKING=1
export DISABLE_VERSION_CHECK=1 
export HF_DATASETS_CACHE=/home/common-vl/cache/hf

## 通用目录
CODE_DIR="./"
DATASET_DIR="./data"
MEDIA_DIR="./data"
OUTPUT_DIR="./output"

## 参数: 1. 模型路径, 2. 模型大小, 3. 数据集名称, 4. 节点数
MODEL_PATH=/home/ckpt/sft/7B/qwen-2.5-vl
MODEL_SIZE=7B ## 7B 72B 
DATA_NAME=dpo-vl-demo
## 集群参数
NUM_NODES=1
NUM_GPUS=8
MASTER_PORT=29888
HOSTFILE=/etc/mpi.host

## cutoff_len
CUTOFF_LEN=20000

learning_rate=1e-6
per_device_train_batch_size=1
gradient_accumulation_steps=16

global_batch_size=$((per_device_train_batch_size * gradient_accumulation_steps * NUM_NODES * NUM_GPUS))

## 模型保存目录
model_saved_name="demo-qwen25vl-${MODEL_SIZE}-len_${CUTOFF_LEN}-gbs_${global_batch_size}-lr_${learning_rate}-data_${DATA_NAME}-SP-8-pref_ftx-00"
model_saved_dir=${OUTPUT_DIR}/${MODEL_SIZE}/${model_saved_name}
mkdir -p ${model_saved_dir}

## tensorboard 目录
tensorboard_dir=${OUTPUT_DIR}/runs_${MODEL_SIZE}/${model_saved_name}
mkdir -p ${tensorboard_dir}

## log file
logfile=${model_saved_dir}/training.log

## model args
MODEL_ARGS=" \
--model_name_or_path $MODEL_PATH \
--image_resolution 1048576 \
--video_resolution 16384 \
"

## method args
METHOD_ARGS=" \
--stage dpo \
--do_train \
--finetuning_type full \
--freeze_vision_tower True \
--train_mm_proj_only False \
--deepspeed examples/deepspeed/ds_z2_config.json \
--use_unsloth_gc True \
"

## data args
DATA_ARGS=" \
--dataset $DATA_NAME \
--dataset_dir $DATASET_DIR \
--template qwen2_vl \
--cutoff_len $CUTOFF_LEN \
--overwrite_cache \
"
#--packing True \
#--preprocessing_num_workers 128 \

## training args
TRAIN_ARGS=" \
--per_device_train_batch_size $per_device_train_batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps \
--learning_rate $learning_rate \
--lr_scheduler_type cosine_with_min_lr \
--lr_scheduler_kwargs \"{\\\"min_lr_rate\\\": 0.1}\" \
--num_train_epochs 1 \
--warmup_ratio 0.05 \
--bf16 \
--flash_attn fa2 \
--ddp_timeout 180000000 \
--pref_beta 0.1 \
--pref_ftx 0.0 \
--sequence_parallel_size 8 \
"

## output args
OUTPUT_ARGS=" \
--output_dir $model_saved_dir \
--save_strategy epoch \
--plot_loss True \
--overwrite_output_dir True \
"

## log args
LOG_ARGS=" \
--report_to tensorboard \
--logging_steps 1 \
--logging_dir $tensorboard_dir \
"
## 启动命令
CMD="torchrun --nnodes=$NUM_NODES --node_rank $RANK --nproc-per-node=8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT ${CODE_DIR}/src/train.py \
$MODEL_ARGS \
$METHOD_ARGS \
$DATA_ARGS \
$TRAIN_ARGS \
$OUTPUT_ARGS \
$LOG_ARGS \
"
echo $CMD

## 主节点输出到日志文件，其他节点输出到屏幕，将启动命令输出到日志文件
if [ $RANK -eq 0 ]; then
    echo $CMD > $logfile
    eval $CMD >> $logfile 2>&1
else
    eval $CMD
fi

cmd_exit_code=$?
exit ${cmd_exit_code}
