#!/bin/bash
#PBS -N rm_parkinson
#PBS -P personal
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=1:00:00
#PBS -q normal
#PBS -e error_rm.log
#PBS -o output_rm.log

cd /home/users/ntu/s250045/scratch/MedicalGPT

module load miniforge3/25.3.1

module load cuda/11.8.0

source activate parkinson

python reward_modeling.py \
    --model_name_or_path ./merged-sft-parkinson \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train --do_eval \
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
    --save_total_limit 3 \
    --max_source_length 512 \
    --max_target_length 512 \
    --output_dir ./outputs-rm-parkinson \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 \
    --report_to tensorboard \
    --remove_unused_columns False

echo "RM训练完成"