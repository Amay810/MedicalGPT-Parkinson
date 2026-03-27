#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
帕金森医疗场景 GRPO 奖励函数

替换 MedicalGPT 原版 grpo_training.py 中面向数学题的 accuracy_reward + format_reward，
适配开放式医疗问答场景。

设计哲学:
- 医疗问答没有 ground truth answer 可以精确匹配，因此不能用原版的 parse+verify
- 改用多维度 rule-based reward: 医学实体覆盖 + 结构化程度 + 安全性 + 长度合理性
- 每个维度独立打分，由 GRPOTrainer 接收多个 reward_funcs 做组内相对排名

奖励函数签名要求 (TRL GRPOTrainer):
  def reward_func(completions, **kwargs) -> list[float]
  - completions: list of list of dict, 每个元素是 [{"role": "assistant", "content": "..."}]
  - 返回: list of float, 长度 = len(completions)
"""

import re
from loguru import logger

# ============================================================
# 帕金森领域关键医学实体词典
# 来源: 50篇循证医学SR + 评测中发现的关键知识点
# ============================================================

# 核心药物实体 (出现任一即表明回答涉及帕金森药物知识)
DRUG_ENTITIES = {
    "左旋多巴", "levodopa", "美多芭", "息宁",
    "多巴胺受体激动剂", "普拉克索", "罗匹尼罗", "罗替戈汀",
    "MAO-B抑制剂", "司来吉兰", "雷沙吉兰",
    "COMT抑制剂", "恩他卡朋", "托卡朋",
    "金刚烷胺", "安坦", "苯海索",
    "抗胆碱能", "多巴胺",
}

# 核心症状实体
SYMPTOM_ENTITIES = {
    "静止性震颤", "震颤", "僵直", "肌强直",
    "运动迟缓", "动作缓慢", "步态", "冻结",
    "姿势不稳", "平衡",
    # 非运动症状
    "便秘", "嗅觉减退", "睡眠障碍", "REM",
    "抑郁", "焦虑", "认知", "幻觉",
    "体位性低血压", "自主神经",
}

# 诊断与检查实体
DIAGNOSIS_ENTITIES = {
    "帕金森", "PD", "帕金森病",
    "DaTSCAN", "DAT扫描", "多巴胺转运体",
    "MRI", "CT", "影像",
    "Hoehn-Yahr", "UPDRS", "MDS",
    "鉴别诊断", "继发性", "帕金森综合征",
    "多系统萎缩", "MSA", "进行性核上性麻痹", "PSP",
}

# 照护与生活管理实体
CARE_ENTITIES = {
    "康复", "运动疗法", "物理治疗",
    "太极", "有氧运动", "步态训练",
    "营养", "吞咽", "防跌倒",
    "照护者", "家属", "护理",
    "生活质量", "日常活动",
}

# 安全性关键词 - 出现表示模型有安全意识
SAFETY_PHRASES = [
    "咨询医生", "就诊", "专科医生", "神经内科",
    "遵医嘱", "医生指导", "个体化", "因人而异",
    "不建议自行", "切勿自行", "请勿擅自",
    "及时就医", "定期随访", "定期复查",
]

# 危险模式 - 出现应受惩罚
DANGEROUS_PATTERNS = [
    "可以停药", "停药没问题", "不需要吃药",
    "自行调整剂量", "自己加量", "自己减量",
    "不需要看医生", "不用去医院", "没必要就诊",
    "保证能治好", "一定能治愈", "包治",
    "偏方", "祖传秘方",
]


# ============================================================
# Reward Function 1: 医学实体覆盖度
# ============================================================
def medical_entity_reward(completions, **kwargs):
    """
    评估回答中医学实体的覆盖度。
    
    逻辑:
    - 扫描回答中出现的实体类别数(药物/症状/诊断/照护)
    - 覆盖越多类别 → 回答越全面 → 分数越高
    - 单类别内命中多个实体有额外加分但递减
    
    分数范围: [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        if not content:
            rewards.append(0.0)
            continue

        score = 0.0
        categories_hit = 0

        # 检查每个实体类别
        for entities, weight in [
            (DRUG_ENTITIES, 0.25),
            (SYMPTOM_ENTITIES, 0.25),
            (DIAGNOSIS_ENTITIES, 0.25),
            (CARE_ENTITIES, 0.25),
        ]:
            hits = sum(1 for e in entities if e in content)
            if hits > 0:
                categories_hit += 1
                # 第一个命中得满分，后续递减
                category_score = min(1.0, 0.6 + 0.1 * hits)
                score += weight * category_score

        rewards.append(round(score, 4))

    logger.debug(f"medical_entity rewards: {rewards}")
    return rewards


# ============================================================
# Reward Function 2: 结构化CoT格式
# ============================================================
def structure_reward(completions, **kwargs):
    """
    评估回答的结构化程度。
    
    你的SFT数据是【思考过程】+【临床建议】双段结构，
    GRPO应该鼓励模型保持这种结构。
    
    检查维度:
    - 是否有分析/思考段落
    - 是否有明确的建议/总结
    - 是否有分点论述 (1. 2. 3. 或 一、二、三)
    - 段落组织是否合理 (不是一整块)
    
    分数范围: [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        if not content:
            rewards.append(0.0)
            continue

        score = 0.0

        # 1. 有思考/分析段落 (0.3)
        think_patterns = ["思考过程", "分析", "从临床角度", "需要考虑", "综合来看",
                          "首先", "其次", "此外", "另外"]
        think_hits = sum(1 for p in think_patterns if p in content)
        if think_hits >= 2:
            score += 0.3
        elif think_hits >= 1:
            score += 0.15

        # 2. 有建议/总结 (0.3)
        advice_patterns = ["建议", "临床建议", "总结", "综上", "方案",
                           "推荐", "注意事项", "需要注意"]
        advice_hits = sum(1 for p in advice_patterns if p in content)
        if advice_hits >= 2:
            score += 0.3
        elif advice_hits >= 1:
            score += 0.15

        # 3. 分点论述 (0.2)
        numbered = len(re.findall(r'[1-9][.、]|[一二三四五六七八九十][、.]', content))
        if numbered >= 3:
            score += 0.2
        elif numbered >= 2:
            score += 0.1

        # 4. 段落组织 (0.2) - 有换行分段
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        if len(paragraphs) >= 3:
            score += 0.2
        elif len(paragraphs) >= 2:
            score += 0.1

        rewards.append(round(score, 4))

    logger.debug(f"structure rewards: {rewards}")
    return rewards


# ============================================================
# Reward Function 3: 安全性
# ============================================================
def safety_reward(completions, **kwargs):
    """
    评估回答的安全性。
    
    医疗场景安全性是底线要求:
    - 包含安全提醒 (咨询医生等) → 加分
    - 包含危险建议 (停药/自行调整等) → 重罚
    
    设计: 默认0.5分起步，安全提醒加分，危险建议扣分。
    这样区分度在于: 有安全意识的回答 > 中性回答 > 危险回答
    
    分数范围: [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        if not content:
            rewards.append(0.0)
            continue

        score = 0.5  # 中性起步

        # 安全提醒加分
        safety_hits = sum(1 for p in SAFETY_PHRASES if p in content)
        if safety_hits >= 3:
            score += 0.5
        elif safety_hits >= 2:
            score += 0.35
        elif safety_hits >= 1:
            score += 0.2

        # 危险模式重罚
        danger_hits = sum(1 for p in DANGEROUS_PATTERNS if p in content)
        score -= danger_hits * 0.3

        rewards.append(round(max(0.0, min(1.0, score)), 4))

    logger.debug(f"safety rewards: {rewards}")
    return rewards


# ============================================================
# Reward Function 4: 长度合理性
# ============================================================
def length_reward(completions, **kwargs):
    """
    评估回答长度的合理性。
    
    基于你的评测经验:
    - 太短 (<100字): 信息量不足，无法覆盖必要的临床建议
    - 适中 (200-1500字): 最佳区间
    - 偏长 (1500-2500字): 轻微惩罚
    - 过长 (>2500字): 可能是重复/废话，重罚
    
    用分段线性函数，不是硬截断。
    
    分数范围: [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        length = len(content)

        if length < 50:
            score = 0.0
        elif length < 100:
            score = 0.2
        elif length < 200:
            score = 0.5 + 0.3 * (length - 100) / 100  # 0.5 → 0.8
        elif length <= 1500:
            score = 1.0  # 最佳区间
        elif length <= 2500:
            score = 1.0 - 0.3 * (length - 1500) / 1000  # 1.0 → 0.7
        else:
            score = max(0.1, 0.7 - 0.3 * (length - 2500) / 1000)  # 递减

        rewards.append(round(score, 4))

    logger.debug(f"length rewards: {rewards}")
    return rewards


# ============================================================
# 组合奖励函数 (可选 - 如果你想用单个reward_func而不是多个)
# ============================================================
def combined_medical_reward(completions, **kwargs):
    """
    加权组合四个维度的奖励。
    
    权重设计:
    - 安全性 0.30 (底线，最重要)
    - 医学实体 0.30 (准确性代理指标)
    - 结构化 0.25 (可读性)
    - 长度 0.15 (辅助)
    
    如果用多个 reward_funcs 传给 GRPOTrainer，则不需要这个函数。
    GRPOTrainer 会自动对多个 reward 做加权平均。
    但如果你想自定义权重，用这个单函数更灵活。
    """
    entity_scores = medical_entity_reward(completions, **kwargs)
    struct_scores = structure_reward(completions, **kwargs)
    safety_scores = safety_reward(completions, **kwargs)
    length_scores = length_reward(completions, **kwargs)

    combined = []
    for e, s, sf, l in zip(entity_scores, struct_scores, safety_scores, length_scores):
        score = 0.30 * sf + 0.30 * e + 0.25 * s + 0.15 * l
        combined.append(round(score, 4))

    logger.debug(f"combined rewards: {combined}")
    return combined


# ============================================================
# 测试工具
# ============================================================
if __name__ == "__main__":
    """快速测试: 模拟3条不同质量的回答，验证reward函数的区分度"""

    test_completions = [
        # 高质量回答: 结构化、有实体、有安全提醒
        [{"role": "assistant", "content": """
【思考过程】
患者询问帕金森病的用药时机，需要综合考虑以下几点：
1. 帕金森病是慢性进展性疾病，左旋多巴是金标准药物
2. 早期可考虑多巴胺受体激动剂（如普拉克索）单药治疗
3. 需根据患者年龄、症状严重程度和生活质量影响个体化决策

【临床建议】
建议您：
1. 尽早到神经内科就诊，完善DaTSCAN等检查明确诊断
2. 用药方案需由专科医生根据Hoehn-Yahr分期制定
3. 配合康复运动如太极拳、步态训练，改善运动症状
4. 定期随访，监测药物疗效及副作用

请勿擅自调整药量，遵医嘱用药。
"""}],
        # 中等质量: 有些信息但缺乏结构
        [{"role": "assistant", "content": """
帕金森病可以吃左旋多巴，也可以吃普拉克索。
具体吃什么药要看你的情况。震颤和僵直是主要症状。
建议去医院看看。
"""}],
        # 低质量: 信息错误、无结构、危险建议
        [{"role": "assistant", "content": """
你这个不用太担心，可以停药试试看，自己减量就行。
偏方可以治好帕金森，不需要看医生。
"""}],
    ]

    print("=" * 60)
    print("GRPO Reward Function 测试")
    print("=" * 60)

    for name, func in [
        ("医学实体", medical_entity_reward),
        ("结构化", structure_reward),
        ("安全性", safety_reward),
        ("长度", length_reward),
        ("综合", combined_medical_reward),
    ]:
        scores = func(test_completions)
        print(f"\n{name}奖励:")
        for i, s in enumerate(scores):
            quality = ["高质量", "中等", "低质量"][i]
            print(f"  {quality}: {s:.4f}")

    print("\n" + "=" * 60)
    print("预期: 高质量 > 中等 > 低质量 (每个维度)")
    print("如果排序不符合预期，需要调整阈值或权重")
