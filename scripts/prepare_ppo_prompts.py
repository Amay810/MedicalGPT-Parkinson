#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
准备PPO训练所需的prompt数据。

MedicalGPT的ppo_training.py (实际是RLOO) 使用conversations格式:
  {"conversations": [{"from": "human", "value": "问题"}]}

只需要问题，不需要回答。从SFT和DPO数据中提取第一轮human提问。
"""

import json
import os
import argparse


def extract_first_turn_from_sft(sft_path):
    questions = []
    with open(sft_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            convs = item.get("conversations", [])
            for turn in convs:
                role = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))
                if role in ("human", "user") and value.strip():
                    questions.append(value.strip())
                    break
    return questions


def extract_from_dpo(dpo_path):
    questions = []
    with open(dpo_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q = item.get("question", "").strip()
            if q:
                questions.append(q)
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_path", type=str, required=True)
    parser.add_argument("--dpo_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data/ppo_prompts")
    args = parser.parse_args()

    sft_q = extract_first_turn_from_sft(args.sft_path)
    dpo_q = extract_from_dpo(args.dpo_path)
    all_q = sorted(set(sft_q + dpo_q))
    all_q = [q for q in all_q if len(q) >= 8]

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "parkinson_ppo_prompts.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in all_q:
            item = {
                "conversations": [
                    {"from": "human", "value": q}
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"SFT问题: {len(sft_q)}, DPO问题: {len(dpo_q)}")
    print(f"去重+过滤后: {len(all_q)} 条")
    print(f"输出: {output_path}")


if __name__ == "__main__":
    main()
