#!/bin/bash
#PBS -N ppo_parkinson
#PBS -P personal
#PBS -l select=1:ncpus=8:ngpus=2:mem=64gb
#PBS -l walltime=4:00:00
#PBS -q normal
#PBS -e error_ppo.log
#PBS -o output_ppo.log

module load miniforge3/25.3.1

module load cuda/11.8.0

source activate parkinson

cd /home/users/ntu/s250045/scratch/MedicalGPT

python ppo_training.py \
    --sft_model_path ./merged-sft-parkinson \
    --reward_model_path ./merged-rm-parkinson \
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
    --output_dir ./outputs-ppo-parkinson \
    --template_name qwen \
    --bf16 True \
    --report_to tensorboard
    --do_train \

echo "PPOÑµÁ·Íê³É"