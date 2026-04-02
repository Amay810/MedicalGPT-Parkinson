# ParkinsonGPT: Domain-Specific LLM Post-Training for Parkinson's Disease QA

> A reproduction and extension of [MedicalGPT](https://github.com/shibing624/MedicalGPT) applied to the Parkinson's disease domain, implementing the full post-training pipeline: **SFT → DPO → Reward Modeling → PPO / GRPO**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen2.5--0.5B--Instruct-orange)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

---

## Overview

This project fine-tunes **Qwen2.5-0.5B-Instruct** through a four-stage post-training pipeline on a Parkinson's disease knowledge corpus constructed from 50 evidence-based systematic reviews (sourced via Consensus AI). The goal is to align a small LLM with domain-specific clinical knowledge and preferred response styles.

**Key results** (15-question clinical evaluation, physician-annotated reference answers):

| Model | Total /5 | Accuracy /2 | Structure /2 | CoT /1 |
|-------|----------|-------------|--------------|--------|
| Base  | 2.73     | 1.07        | 1.00         | 0.67   |
| SFT   | **4.27** ⭐ | **1.33** | 1.93         | 1.00   |
| DPO   | 4.13     | 1.20        | 1.93         | 1.00   |
| GRPO  | 3.93     | 0.93        | **2.00** ✦  | 1.00   |
| PPO   | 3.80     | 0.80        | **2.00** ✦  | 1.00   |

SFT delivers the largest single-stage gain (+1.54). GRPO/PPO achieve perfect structure scores via RL but trade off factual accuracy on a 0.5B model — a clear alignment tax at small scale.

---

## Architecture

```
50 Evidence-Based SRs (Consensus AI)
        │
        ▼
  Data Construction Pipeline
  ├── SFT data:  169 QA pairs (CoT format, DeepSeek-R1 distillation)
  └── DPO data:  293 preference pairs (chosen/rejected, Huatuo26M filtered)
        │
        ▼
Stage 1: SFT  (Qwen2.5-0.5B-Instruct + LoRA r=8)
        │
        ▼
Stage 2: DPO  (293 preference pairs, Run G: max_steps=70, lr=1e-5)
        │
        ├──► Path A: DPO as final aligned model
        │
        ├──► Path B: Reward Modeling → PPO (RLOO)
        │
        └──► Path C: GRPO (rule-based reward, 4-dimensional)
```

---

## Repository Structure

```
MedicalGPT-Parkinson/
│
├── data/
│   ├── parkinson/
│   │   ├── sft/parkinson_sft_full.jsonl        # 169 SFT samples (ShareGPT format)
│   │   ├── dpo/parkinson_dpo_clean.jsonl        # 293 DPO preference pairs
│   │   ├── grpo_prompts/                        # GRPO prompt set
│   │   ├── ppo_prompts/                         # PPO prompt set
│   │   └── reward/                              # RM training data
│   └── eval/                                    # Evaluation questions & results
│       ├── questions.txt
│       ├── parkinson_real_questions_v2.jsonl
│       ├── grpo_eval_results.jsonl
│       └── ppo_eval_results.jsonl
│
├── rewards/
│   ├── __init__.py
│   └── parkinson_rewards.py                     # 4-dim rule-based GRPO rewards
│
├── scripts/
│   ├── prepare_rm_data.py                       # DPO data → RM training format
│   ├── prepare_grpo_prompts.py                  # Build GRPO prompt set
│   ├── prepare_ppo_prompts.py                   # Build PPO prompt set
│   └── diagnose_grpo.py                         # GRPO training diagnostics
│
├── supervised_finetuning.py                     # SFT trainer
├── dpo_training.py                              # DPO trainer
├── reward_modeling.py                           # Reward model trainer
├── ppo_training.py                              # PPO (RLOO) trainer
├── grpo_training_parkinson.py                   # Custom GRPO trainer (new)
├── merge_peft_adapter.py                        # Merge LoRA weights
├── inference.py                                 # Batch inference
├── template.py                                  # Prompt templates
│
├── run_sft.sh                                   # SFT launch
├── run_dpo.sh                                   # DPO launch (Run G config)
├── run_rm.sh                                    # RM training (standalone)
├── run_ppo.sh                                   # PPO training (standalone)
├── run_rm_ppo_parkinson.sh                      # Full RM+PPO pipeline
├── run_grpo_parkinson.sh                        # GRPO launch (4×GPU)
│
├── train_parkinson.pbs                          # PBS: SFT/DPO job
├── infer_all_models.pbs                         # PBS: inference for all 5 models
│
├── requirements.txt
├── README.md
├── LICENSE
└── DISCLAIMER
```

---

## Data Construction

### SFT Data (169 samples)

Three-source construction strategy:

1. **SR distillation (62 samples)**: Knowledge-distilled from 50 evidence-based systematic reviews via DeepSeek-R1, generating structured `【思考过程】/【临床建议】` CoT QA pairs.
2. **Persona synthesis (93 samples)**: Script-generated covering patient / family / caregiver perspectives.
3. **Huatuo augmentation (14 samples)**: Filtered from Huatuo26M-Lite to fill the colloquial family-caregiver distribution gap.

### DPO Data (293 pairs) — Three-Stage Pipeline

**Stage 1 — Keyword filtering**: Streamed Huatuo26M-Lite with 34 Parkinson-specific keywords → 534 candidates from 177,703 records.

**Stage 2 — Semantic filtering (with bug fix)**:
- *Original bug*: Embedding on `question + answer[:300]` caused 74% false positives due to shared consultation tone — style similarity, not topic similarity.
- *Fix*: Embed `question` field only; switch from Top-5 mean to max similarity; fix `utf-8-sig` CSV encoding; use local `paraphrase-multilingual-MiniLM-L12-v2` (DeepSeek embedding API returned 404).

**Stage 3 — LLM judge (two rounds)**:
- Round 1: Binary yes/no → 26% pass rate, 139 samples.
- Round 2: 1–3 scoring, keep ≥2 → 37% pass rate, 200 final samples.

**Chosen/Rejected generation**: DeepSeek-R1 (8 concurrent workers, ~44 min). Rejected responses are **length-matched to chosen** to eliminate the length-shortcut bias in DPO training.

**Data audit**: Discovered 150 placeholder-contaminated samples (3 template variables unreplaced, each appearing 50×) via `Counter` analysis → 326 clean samples → 293 training + 33 eval.

---

## Training Details

### Stage 1: SFT

```bash
bash run_sft.sh
```

- Model: `Qwen/Qwen2.5-0.5B-Instruct` + LoRA (r=8, α=16, dropout=0.05)
- Data: 169 samples, 10% eval split
- Key: Established dual-label CoT format (`【思考过程】/【临床建议】`)

### Stage 2: DPO (Run G — Best of 7 iterations)

```bash
bash run_dpo.sh
```

- `max_steps=70`, `lr=1e-5`, `warmup_steps=10`, cosine schedule
- eval loss: **0.149**, eval margin: **1.963**, rewards/chosen: **+0.346** (no collapse)
- 7 iterations (Run A–G); key insight: completing the full cosine decay curve (steps 55→70) was critical for stable convergence.

### Stage 3B: Reward Modeling + PPO

```bash
# Option 1: run RM and PPO separately (more stable)
bash run_rm.sh
bash run_ppo.sh

# Option 2: full pipeline in one job
bash run_rm_ppo_parkinson.sh
```

Converts DPO preference pairs → RM training format, trains RM via LoRA, merges weights, then runs PPO (RLOO) from SFT model.

### Stage 3C: GRPO

```bash
bash run_grpo_parkinson.sh   # requires 4× GPU
```

- 4-dimensional rule-based reward: safety, medical entity coverage, CoT structure, length
- `torchrun --nproc_per_node=4`, 483 prompts from combined SFT+DPO data

---

## Evaluation

Inference for all 5 model checkpoints is managed via a single PBS script:

```bash
# Edit infer_all_models.pbs: change "if false" to "if true" for the model you want
qsub infer_all_models.pbs
```

Results are saved to `data/eval/output_*.jsonl`.

---

## Environment

```bash
conda create -n parkinson python=3.10
conda activate parkinson
pip install -r requirements.txt
```

Tested on: NVIDIA A100 (NTU HPC cluster), CUDA 11.8, `miniforge3/25.3.1`.

HuggingFace mirror (China):
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Citation

```bibtex
@misc{ParkinsonGPT2025,
  title  = {ParkinsonGPT: Domain-Specific LLM Post-Training for Parkinson's Disease QA},
  year   = {2025},
  note   = {\url{https://github.com/Amay810/MedicalGPT-Parkinson}}
}
```

Based on [MedicalGPT](https://github.com/shibing624/MedicalGPT) by Ming Xu (Apache 2.0).

---

## Disclaimer

This project is for research purposes only and should not be used as a substitute for professional medical advice. Model outputs may contain errors or hallucinations.
