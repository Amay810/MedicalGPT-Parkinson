#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
grpo_training_parkinson.py

基于 MedicalGPT v2.4 的 grpo_training.py 修改，适配帕金森医疗问答场景。

核心改动:
1. 替换 accuracy_reward (数学精确匹配) → medical_entity_reward (医学实体覆盖)
2. 替换 format_reward (<think>/<answer>标签) → structure_reward (CoT结构)
3. 新增 safety_reward (安全性) 和 length_reward (长度合理性)
4. SYSTEM_PROMPT 改为帕金森专科医生角色
5. dataset map 适配不需要 ground truth answer 的场景

用法:
  python grpo_training_parkinson.py \
    --model_name_or_path ./sft_run2_merged \
    --train_file_dir ./data/grpo_prompts \
    --output_dir ./outputs-grpo-parkinson \
    --num_generations 4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
    --bf16 True
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import re
from datasets import load_dataset
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from peft import LoraConfig, TaskType, get_peft_model

# 导入帕金森医疗奖励函数
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rewards.parkinson_rewards import (
    medical_entity_reward,
    structure_reward,
    safety_reward,
    length_reward,
)

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ScriptArguments:
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing training files for local datasets."}
    )
    train_samples: Optional[int] = field(default=-1, metadata={"help": "Number of samples to train on, -1 for all"})
    subset_name: Optional[str] = field(default="main", metadata={"help": "Subset name"})
    dataset_splits: Optional[str] = field(default="train", metadata={"help": "Split name"})
    preprocessing_num_workers: Optional[int] = field(default=10, metadata={"help": "Number of workers"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})


# ============================================================
# 帕金森专科 System Prompt
# ============================================================
SYSTEM_PROMPT = (
    "你是一位帕金森病专科医生助手。回答患者、家属或照护者的问题时，请：\n"
    "1. 先进行临床分析思考，说明考虑要点\n"
    "2. 再给出具体的临床建议\n"
    "3. 回答应基于循证医学证据\n"
    "4. 涉及用药、诊断等建议时，提醒患者咨询专科医生"
)


def find_all_linear_names(peft_model, int4=False, int8=False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def get_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_train(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    is_main_process = training_args.local_rank in [-1, 0]

    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
            f"n_gpu: {training_args.n_gpu}, "
            f"distributed training: {bool(training_args.local_rank != -1)}, "
            f"16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    if script_args.train_file_dir and os.path.exists(script_args.train_file_dir):
        dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")
    else:
        dataset = load_dataset(script_args.dataset_name, script_args.subset_name, split=script_args.dataset_splits)

    if script_args.train_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(script_args.train_samples))

    # Prepare dataset - 帕金森场景不需要ground truth answer
    with training_args.main_process_first(desc="Dataset preparation"):
        dataset = dataset.map(
            lambda x: {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': x['question']}
                ],
                # answer 字段保留但不用于reward计算
                # 原版数学题场景用 answer 做精确匹配，医疗场景不需要
                'answer': x.get('answer', '')
            },
            num_proc=script_args.preprocessing_num_workers,
            desc="Processing dataset" if is_main_process else None,
        )

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    if is_main_process:
        logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(test_dataset)}")
        logger.info("*** Initializing model ***")

    # Model initialization
    torch_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    if script_args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 and QLoRA are incompatible.")

    quantization_config = None
    if script_args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        quantization_config=quantization_config,
    )

    # LoRA setup
    if model_args.use_peft:
        if is_main_process:
            logger.info("Fine-tuning method: LoRA(PEFT)")
        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            logger.warning("Gradient checkpointing with LoRA may cause issues, disabling.")
            training_args.gradient_checkpointing = False

        #target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        #if target_modules == 'all' or (target_modules and 'all' in target_modules):
        #    target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if is_main_process:
            logger.info(f"Peft target_modules: {target_modules}, lora rank: {model_args.lora_r}")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        if is_main_process:
            logger.info("Fine-tuning method: Full parameters training")

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    # ============================================================
    # 核心改动: 使用帕金森医疗奖励函数替换原版数学题奖励
    # ============================================================
    # 原版:  reward_funcs=[accuracy_reward, format_reward]
    # 改为:  四个医疗维度的独立reward函数
    #
    # GRPOTrainer 会对每个prompt采样 num_generations 个response，
    # 对每个response调用所有reward_funcs，将结果相加作为总reward，
    # 然后在组内做相对排名计算advantage。
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            safety_reward,           # 安全性 (最重要)
            medical_entity_reward,   # 医学实体覆盖度
            structure_reward,        # 结构化CoT格式
            length_reward,           # 长度合理性
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
    )
    logger.info("*** GRPO Trainer initialized with Parkinson medical rewards ***")

    # Training
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if is_main_process:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
            f'{training_args.num_train_epochs} epochs ***'
        )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("*** Training complete ***")

    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        kwargs = {"dataset_name": script_args.dataset_name, "tags": ["grpo", "parkinson", "medical"]}
        trainer.create_model_card(**kwargs)
        trainer.model.config.save_pretrained(training_args.output_dir)
        logger.info("*** Training complete! ***")

    trainer.generate_completions()


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
