#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import argparse


def convert_dpo_to_rm(dpo_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "parkinson_reward.jsonl")

    pairs = 0
    with open(dpo_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item.get("question", "").strip()
            chosen = item.get("response_chosen", "").strip()
            rejected = item.get("response_rejected", "").strip()
            if not question or not chosen or not rejected:
                continue
            out = {
                "system": "",
                "history": [],
                "question": question,
                "response_chosen": chosen,
                "response_rejected": rejected
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            pairs += 1

    print(f"读取DPO数据，输出: {pairs} 条")
    print(f"保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data/reward")
    args = parser.parse_args()
    convert_dpo_to_rm(args.dpo_path, args.output_dir)
