# -*- coding: utf-8 -*-
"""
@description: Train a model from SFT using RLOO (REINFORCE Leave-One-Out, PPO alternative)

Modified for MedicalGPT-Parkinson: fixed preprocess_function for prompt-only data.
"""

import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
)
from trl import (
    RLOOConfig,
    RLOOTrainer,
    ModelConfig,
    get_peft_config,
)
from template import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class RLOOArguments:
    """
    The name of the Casual LM model we wish to fine with RLOO
    """
    sft_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the SFT model."})
    reward_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the reward model."})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(default=None, metadata={"help": "Dataset configuration name."})
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The template name."})
    max_source_length: Optional[int] = field(default=1024, metadata={"help": "Max length of prompt input text"})
    system_prompt: Optional[str] = field(
        default="你是一位帕金森病专科医生，擅长帕金森病的诊断、用药决策和日常照护指导。请基于循证医学证据，为患者及家属提供专业、结构化的临床建议。",
        metadata={"help": "System prompt for the conversation."}
    )


def main():
    parser = HfArgumentParser((RLOOArguments, RLOOConfig, ModelConfig))
    args, training_args, model_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )[:3]

    # Add distributed training initialization
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = local_rank == 0

    # Only log on main process
    if is_main_process:
        logger.info(f"Parse args: {args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Model args: {model_args}")

    # Load tokenizer
    sft_model_path = args.sft_model_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        sft_model_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.sep_token
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    # Load reward model as reward function
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )

    # Load policy model
    policy = AutoModelForCausalLM.from_pretrained(
        sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)

    # Get datasets
    prompt_template = get_conv_template(args.template_name)
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_train_split
        )
        eval_samples = 100
        train_dataset = dataset.select(range(len(dataset) - eval_samples))
        eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    else:
        data_files = {}
        if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
            train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
            eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        dataset = load_dataset(
            'json',
            data_files=data_files,
        )
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        eval_dataset = val_dataset.select(range(min(100, len(val_dataset))))
    logger.info(f"Get datasets: {train_dataset}, {eval_dataset}")

    # Preprocessing the datasets
    # PPO/RLOO only needs prompts (no gpt responses needed)
    system_prompt = args.system_prompt

    def preprocess_function(examples):
        """
        Convert prompt-only conversations data to formatted prompt strings.
        Input format: {"conversations": [{"from": "human", "value": "..."}]}
        Output format: {"prompt": ["<chat_template_formatted_prompt>"]}
        """
        new_examples = {"prompt": []}

        for i, source in enumerate(examples['conversations']):
            if not source or len(source) < 1:
                continue

            # Get the human message
            first_msg = source[0]
            data_role = first_msg.get("from", "")
            if data_role != "human":
                # Try to find a human message
                human_msgs = [s for s in source if s.get("from", "") == "human"]
                if not human_msgs:
                    continue
                query = human_msgs[0]["value"]
            else:
                query = first_msg["value"]

            if not query or not query.strip():
                continue

            # Build chat messages and apply template
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})

            # Use tokenizer's chat template to format the prompt
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            new_examples["prompt"].append(prompt_text)

        return new_examples

    # Preprocess the dataset
    if is_main_process:
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=getattr(training_args, 'dataset_num_proc', 4),
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )
    train_dataset = train_dataset.filter(lambda x: len(x['prompt']) > 0)

    if is_main_process:
        logger.info(f"Train dataset size after preprocessing: {len(train_dataset)}")
        if len(train_dataset) > 0:
            logger.debug(f"Train prompt[0] (first 200 chars): {train_dataset[0]['prompt'][:200]}")
        else:
            logger.error("Train dataset is EMPTY after preprocessing! Check data format.")

    # Preprocess the dataset for evaluation
    if is_main_process:
        logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=getattr(training_args, 'dataset_num_proc', 4),
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on eval dataset",
    )
    eval_dataset = eval_dataset.filter(lambda x: len(x['prompt']) > 0)

    if is_main_process:
        logger.info(f"Eval dataset size after preprocessing: {len(eval_dataset)}")
        if len(eval_dataset) > 0:
            logger.debug(f"Eval prompt[0] (first 200 chars): {eval_dataset[0]['prompt'][:200]}")
        else:
            logger.error("Eval dataset is EMPTY after preprocessing! Check data format.")
    
    training_args.do_train = True
    
    # We then build the RLOOTrainer, passing the model, the reward function, the tokenizer
    trainer = RLOOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        reward_funcs=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Training
    if training_args.do_train:
        if is_main_process:
            logger.info("*** Train ***")
        trainer.train()

        if is_main_process:
            trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
