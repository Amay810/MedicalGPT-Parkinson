#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从SFT数据和DPO数据中提取unique questions，构建GRPO训练所需的prompt数据集。

GRPO只需要prompt（不需要response），模型会自行采样多个response，
然后用reward function对组内response进行相对排名。

输入:
  - SFT数据: parkinson_sft_169.jsonl (ShareGPT格式)
  - DPO数据: parkinson_dpo_clean.jsonl (question/chosen/rejected格式)
输出:
  - data/grpo_prompts/parkinson_grpo_prompts.jsonl
    格式: {"question": "...", "answer": ""}
    answer字段留空，GRPO不需要ground truth answer（reward由函数计算）
"""

import json
import os
import argparse
from collections import Counter


def extract_questions_from_sft(sft_path):
    """从ShareGPT格式的SFT数据中提取第一轮human提问"""
    questions = []
    with open(sft_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            convs = item.get("conversations", [])
            # 只取第一个human turn
            for turn in convs:
                role = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))
                if role in ("human", "user") and value.strip():
                    questions.append(value.strip())
                    break  # 只取第一轮
    return questions


def extract_questions_from_dpo(dpo_path):
    """从DPO数据中提取question字段"""
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
    parser = argparse.ArgumentParser(description="准备GRPO训练的prompt数据集")
    parser.add_argument("--sft_path", type=str, required=True,
                        help="SFT数据路径 (jsonl, ShareGPT格式)")
    parser.add_argument("--dpo_path", type=str, required=True,
                        help="DPO清洗后数据路径 (jsonl)")
    parser.add_argument("--output_dir", type=str, default="./data/grpo_prompts",
                        help="输出目录")
    parser.add_argument("--min_length", type=int, default=8,
                        help="最小问题长度（字符数），过滤过短的噪声")
    args = parser.parse_args()

    # 提取
    sft_questions = extract_questions_from_sft(args.sft_path)
    dpo_questions = extract_questions_from_dpo(args.dpo_path)

    print(f"SFT提取问题数: {len(sft_questions)}")
    print(f"DPO提取问题数: {len(dpo_questions)}")

    # 合并去重
    all_questions = list(set(sft_questions + dpo_questions))
    print(f"合并去重后: {len(all_questions)}")

    # 过滤过短
    filtered = [q for q in all_questions if len(q) >= args.min_length]
    print(f"过滤min_length={args.min_length}后: {len(filtered)}")

    # 输出
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "parkinson_grpo_prompts.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in sorted(filtered):
            # GRPO数据格式: question + answer(空)
            # grpo_training.py 的 dataset.map 会读取 question 和 answer 字段
            item = {"question": q, "answer": ""}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n输出到: {output_path}")
    print(f"总计: {len(filtered)} 条 unique prompts")

    # 统计来源分布
    sft_set = set(sft_questions)
    dpo_set = set(dpo_questions)
    both = sft_set & dpo_set
    sft_only = sft_set - dpo_set
    dpo_only = dpo_set - sft_set
    print(f"\n来源分布:")
    print(f"  仅SFT: {len(sft_only)}")
    print(f"  仅DPO: {len(dpo_only)}")
    print(f"  两者重叠: {len(both)}")


if __name__ == "__main__":
    main()
