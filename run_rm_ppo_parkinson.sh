#!/bin/bash
#PBS -N rm_ppo_parkinson
#PBS -P personal
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=6:00:00
#PBS -q normal
#PBS -e error_ppo.log 
#PBS -o output_ppo.log 


PROJECT_DIR="/home/users/ntu/s250045/scratch/MedicalGPT"
SFT_MODEL="${PROJECT_DIR}/merged-sft-parkinson"        
DPO_DATA="${PROJECT_DIR}/data/parkinson/dpo/parkinson_dpo_clean.jsonl"
SFT_DATA="${PROJECT_DIR}/data/parkinson/sft/parkinson_sft_full.jsonl"
RM_OUTPUT="${PROJECT_DIR}/outputs-rm-parkinson"
RM_MERGED="${PROJECT_DIR}/outputs-rm-parkinson-merged"
PPO_OUTPUT="${PROJECT_DIR}/outputs-ppo-parkinson"

module load miniforge3/25.3.1

module load cuda/11.8.0

source activate parkinson

cd ${PROJECT_DIR}

echo "=========================================="
echo "Step 1: 准备RM训练数据"
echo "=========================================="
python scripts/prepare_rm_data.py \
    --dpo_path ${DPO_DATA} \
    --output_dir ./data/reward

echo "=========================================="
echo "Step 2: 训练 Reward Model"
echo "=========================================="

python reward_modeling.py \
    --model_name_or_path ${SFT_MODEL} \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --weight_decay 0.001 \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 100 \
    --save_strategy steps \
    --save_total_limit 3 \
    --max_source_length 512 \
    --max_target_length 512 \
    --output_dir ${RM_OUTPUT} \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 \
    --report_to tensorboard \
    --remove_unused_columns False

echo "=========================================="
echo "Step 3: 合并RM LoRA权重"
echo "=========================================="
python merge_peft_adapter.py \
    --base_model ${SFT_MODEL} \
    --lora_model ${RM_OUTPUT} \
    --output_dir ${RM_MERGED}

echo "=========================================="
echo "Step 4: 准备PPO prompt数据"
echo "=========================================="
python scripts/prepare_ppo_prompts.py \
    --sft_path ${SFT_DATA} \
    --dpo_path ${DPO_DATA} \
    --output_dir ./data/ppo_prompts

echo "=========================================="
echo "Step 5: PPO训练 (RLOO)"
echo "=========================================="

python ppo_training.py \
    --sft_model_path ${SFT_MODEL} \
    --reward_model_path ${RM_MERGED} \
    --train_file_dir ./data/ppo_prompts \
    --validation_file_dir ./data/ppo_prompts \
    --per_device_train_batch_size 2 \
    --num_generations 2 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --warmup_steps 20 \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 3 \
    --output_dir ${PPO_OUTPUT} \
    --template_name qwen \
    --bf16 True \
    --report_to tensorboard

echo "=========================================="
echo "RM + PPO 训练完成!"
echo "RM模型: ${RM_MERGED}"
echo "PPO模型: ${PPO_OUTPUT}"
echo "=========================================="
