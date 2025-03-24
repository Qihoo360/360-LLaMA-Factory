#fintune type full lora qlora
    # --optim paged_adamw_32bit \
# export CUDA_HOME=/usr/local/cuda-12.1 
# export PATH=$PATH:/usr/local/cuda-12.1
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DS_SKIP_CUDA_CHECK=1
hostfile="hostfile.4567"
deepspeed --hostfile $hostfile --master_port=29501 src/train.py \
    --stage sft \
    --do_train \
    --max_steps -1 \
    --model_name_or_path /xfr_ceph_sh/open_models/Qwen-R1-Model/Qwen2.5-14B-Instruct  \
    --template qwen \
    --dataset r1-math-7w-stage1-filter \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --flash_attn fa2  \
    --overwrite_cache \
    --cutoff_len 20480 \
    --output_dir /xfr_ceph_sh/xiaofenrui/360-LLaMA-Factory/models/qwen2.5-14b-sft-7w-stage1-5e-5-cosin-random-pack-test \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --save_strategy epoch \
    --logging_steps 1 \
    --warmup_ratio 0.0 \
    --packing True \
    --packing_method "random" \
    --neat_packing True \
    --save_total_limit 100 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --save_only_model True \
    --num_train_epochs 5.0 \
    --bf16 true \
    --plot_loss \
    --seed 42 \
    --do_eval false \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ddp_timeout 180000000 \
