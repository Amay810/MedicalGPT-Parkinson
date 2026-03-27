#!/bin/bash
#PBS -N grpo_parkinson_m
#PBS -l select=1:ncpus=8:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00
#PBS -q normal
#PBS -P personal
#PBS -o output_grpo_m.log           
#PBS -e error_grpo_m.log            


PROJECT_DIR="/home/users/ntu/s250045/scratch/MedicalGPT"
SFT_MODEL="${PROJECT_DIR}/outputs-sft-parkinson-v2"
GRPO_PROMPT_DIR="${PROJECT_DIR}/data/grpo_prompts_m"
OUTPUT_DIR="${PROJECT_DIR}/grpo_parkinson_run_m"
# ---------------------

module load miniforge3/25.3.1

module load cuda/11.8.0

source activate parkinson

cd ${PROJECT_DIR}

if [ ! -f "${GRPO_PROMPT_DIR}/parkinson_grpo_prompts.jsonl" ]; then
    echo ">>> 准备GRPO prompt数据..."
    python scripts/prepare_grpo_prompts.py \
        --sft_path ./data/parkinson/sft/parkinson_sft_full.jsonl \
        --dpo_path ./data/parkinson/dpo/parkinson_dpo_clean.jsonl \
        --output_dir ${GRPO_PROMPT_DIR}
fi

echo ">>> 开始GRPO训练..."
echo ">>> SFT模型: ${SFT_MODEL}"
echo ">>> Prompt数据: ${GRPO_PROMPT_DIR}"
echo ">>> 输出目录: ${OUTPUT_DIR}"

torchrun --nproc_per_node=4 grpo_training_parkinson.py \
    --model_name_or_path ${SFT_MODEL} \
    --train_file_dir ${GRPO_PROMPT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_generations 4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --warmup_steps 30 \
    --logging_steps 5 \
    --save_steps 50 \
    --eval_strategy steps \
    --eval_steps 25 \
    --save_total_limit 3 \
    --max_completion_length 1024 \
    --bf16 True \
    --use_peft True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules all \
    --report_to tensorboard \

echo ">>> GRPO训练完成!"
echo ">>> 模型保存在: ${OUTPUT_DIR}"


