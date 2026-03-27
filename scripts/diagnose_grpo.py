#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRPO 训练监控与诊断工具

在训练过程中或训练后分析 TensorBoard 日志，
诊断 reward hacking、KL divergence 爆炸、reward 信号无区分度等问题。

用法:
  python scripts/diagnose_grpo.py --log_dir ./outputs-grpo-parkinson

输出诊断结论和建议的超参调整方向。
"""

import argparse
import json
import os
from pathlib import Path


def print_diagnosis_guide():
    """打印GRPO训练诊断指南 - 基于你DPO阶段的经验教训"""

    guide = """
╔══════════════════════════════════════════════════════════════════╗
║          GRPO 帕金森医疗训练 - 诊断与调参指南                   ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一、关键监控指标 (TensorBoard)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. reward/mean
   - 应该: 缓慢上升，说明模型在学习产生更好的回答
   - 异常: 快速飙升 → reward hacking（见问题1）
   - 异常: 不动或下降 → reward信号无区分度（见问题3）

2. reward/std
   - 应该: 训练初期较高，后期逐渐下降
   - 这表示组内response质量趋于一致（模型学到了）
   - 异常: 始终很低 → reward函数区分度不够

3. policy/approx_kl
   - 应该: < 0.1（保守）到 < 0.5（正常范围）
   - 异常: > 1.0 → KL爆炸，policy偏离太远（见问题2）

4. loss/policy_loss
   - 应该: 波动中缓慢下降
   - 异常: 突然跳高或NaN → 学习率太高

5. 各reward分项 (如果用多个reward_funcs)
   - 观察哪个维度在贡献主要梯度信号
   - 如果safety_reward全是1.0 → 对训练无贡献，可以去掉或加严标准

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二、常见问题诊断
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【问题1: Reward Hacking】
  症状: reward/mean快速上升，但生成质量明显下降
  
  你在DPO阶段已经遇过类似问题（Run D中模型通过拉低rejected刷margin）。
  GRPO中的等价现象: 模型学会了堆砌医学术语骗entity_reward，
  或者生成固定长度的模板文本骗length_reward。
  
  排查方法:
    1. 从训练中间采样几条生成结果，人工阅读
    2. 检查reward各维度分项是否不均衡（某一项独大）
  
  修复:
    - 降低lr (如 5e-7 → 1e-7)
    - 增大 kl_coef 提高KL惩罚（默认0.05，可调到0.1-0.2）
    - 修改reward函数，加入重复惩罚:
      ```python
      # 在 parkinson_rewards.py 中添加
      def repetition_penalty_reward(completions, **kwargs):
          rewards = []
          for completion in completions:
              content = completion[0]["content"]
              sentences = content.split("。")
              unique_ratio = len(set(sentences)) / max(len(sentences), 1)
              rewards.append(unique_ratio)
          return rewards
      ```

【问题2: KL Divergence 爆炸】
  症状: policy/approx_kl > 1.0，生成结果与SFT差异极大
  
  对应你DPO Run D的教训: lr=5e-4导致模型崩塌。
  
  修复:
    - 降低lr (5e-7 → 1e-7)
    - 增大 kl_coef (0.05 → 0.1 → 0.2)
    - 减少 num_train_epochs
    - 注意: GRPO的kl_coef是GRPOConfig的参数:
      --kl_coef 0.1

【问题3: Reward信号无区分度】
  症状: reward/mean几乎不动，reward/std非常低
  
  这说明你的reward函数对组内的4个response打分差不多，
  模型没有梯度信号来学习哪个更好。
  
  排查: 手动看几个prompt的4个response和各自的reward分数。
  如果确实难以区分 → reward函数太粗糙。
  
  修复:
    - 加严reward函数阈值（比如entity_reward要求更多实体覆盖才能高分）
    - 增加 num_generations (4→8)，增加组内方差
    - 考虑引入更强的reward信号（如LLM-as-Judge）

【问题4: 显存不足 (OOM)】
  num_generations=4 对 0.5B 模型通常可以，如果OOM:
    - 减小 max_completion_length (1024→512)
    - 减小 per_device_train_batch_size (1已经是最小了)
    - 用 gradient_accumulation_steps 代替大batch
    - 最后考虑 num_generations 降到 2（但会降低训练效果）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三、超参选择建议（基于你的资源和数据量）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

你的场景特征:
  - 模型: Qwen2.5-0.5B-Instruct (小模型)
  - 数据: ~300条 unique prompts (小数据)
  - GPU: PBS集群 (假设单卡A100 40GB或V100 32GB)

推荐超参:
  --num_generations 4          # 每prompt采样4条，组内相对排名
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 4  # 有效batch=4个prompt
  --learning_rate 5e-7         # 比你DPO最终的1e-5还要低，GRPO需要更保守
  --num_train_epochs 3         # 300条×3轮，总约900步
  --kl_coef 0.05               # 默认值，先跑一轮看KL
  --max_completion_length 1024 # 允许模型生成足够长的回答
  --warmup_ratio 0.1
  --lr_scheduler_type cosine
  --save_steps 50              # 约每50步存一个checkpoint
  --eval_steps 25              # 评估频率

迭代策略:
  Run 1: 用上述默认参数跑完，观察TensorBoard
  Run 2: 根据Run 1诊断调整:
    - KL太高 → 加大kl_coef
    - reward不动 → 加大num_generations或修改reward函数
    - 过拟合 → 减少epochs
  Run 3: 尝试reward函数变体（比如去掉length_reward，或替换为combined_reward）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
四、评估方案
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用你已有的15条评测集（丁香医生真实患者提问），对比:

  Base → SFT Run2 → DPO Run G → GRPO Run X

四个checkpoint在同一评测集上的:
  - 准确性 /2
  - 结构性 /2
  - CoT /1
  - 总分 /5

预期:
  - GRPO应该在结构性和CoT上 >= DPO（reward函数直接优化了这两点）
  - 准确性取决于medical_entity_reward的信号质量
  - 如果GRPO准确性也低于SFT（像DPO一样），说明rule-based reward
    不足以替代人工标注的偏好信号，这本身也是一个有价值的发现

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
五、面试叙事框架
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"我在完成DPO训练后，进一步实现了GRPO，目的是对比两种对齐路径:

  路径A (DPO): 人工构造偏好对 → 隐式reward建模 → offline优化
  路径B (GRPO): 自定义reward函数 → 显式reward信号 → online采样优化

关键挑战是设计医疗场景的reward函数——不像数学题有标准答案，
医疗问答需要同时评估安全性、准确性、结构化程度。
我设计了四维度reward（安全性/实体覆盖/结构化/长度），
发现 [你的实验结论]。

最有价值的发现是 [你发现的问题和分析]，
这让我理解了 [对RLHF/对齐的deeper insight]。"
"""
    print(guide)


def check_tensorboard_logs(log_dir):
    """检查TensorBoard日志文件是否存在"""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        return False

    events_files = list(log_dir.rglob("events.out.tfevents.*"))
    if events_files:
        print(f"找到 {len(events_files)} 个TensorBoard日志文件:")
        for f in events_files:
            print(f"  {f}")
        print(f"\n查看命令: tensorboard --logdir {log_dir}")
        return True
    else:
        print(f"目录 {log_dir} 下没有找到TensorBoard日志文件")
        return False


def check_training_metrics(output_dir):
    """检查训练metrics文件"""
    metrics_file = os.path.join(output_dir, "train_results.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
        print("\n训练结果:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print(f"未找到训练结果文件: {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard日志目录")
    parser.add_argument("--guide", action="store_true", help="打印诊断指南")
    args = parser.parse_args()

    if args.guide or args.log_dir is None:
        print_diagnosis_guide()

    if args.log_dir:
        check_tensorboard_logs(args.log_dir)
        check_training_metrics(args.log_dir)
