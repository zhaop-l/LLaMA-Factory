set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export COMET_API_KEY=MiIaJfB7mRa3IKnbDc0uzFAfB

current_date=$(date +%Y%m%d)
current_datetime=$(date +%Y%m%d%H)

OUTPUT_DIR='./work_dirs/qwen2-vl-7b/'${current_date}'/qwen2-vl-7b-sft-lora-llmdata-'${current_datetime}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


DISTRIBUTED_ARGS="
--nnodes=1 \
--node_rank=0 \
--master_addr=127.0.0.1 \
--nproc_per_node=${GPUS} \
--master_port=${MASTER_PORT}
"

torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed "/data02/zhaop-l/code/LLaMA-Factory/examples/deepspeed/ds_z3_config.json" \
    --stage sft \
    --do_train \
    --model_name_or_path /data03/public/mllm/Qwen/Qwen2-VL-7B-Instruct/ \
    --dataset processed_data_mixed_3000 processed_RAG \
    --template qwen2_vl \
    --finetuning_type lora \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16 \
    --val_size 0.1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 500 \
    --report_to "comet_ml" "tensorboard" \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"