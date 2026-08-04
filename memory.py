# memory.py 存储FYP背景，只做为字典使用
# 包括主要目标，四个研究方向，三类题目类型，衡量最后标准值，用户背景
def get_FYP_memory():
    return {
        "project_title": "CL-1 Reasoning Distillation for Compact Language Models",
        "project_type": "Empirical Research",
        "main_goal":(
            "Study whether training compact language models with rationale-enriched data"
            "can improve reasoning performance"
        ),
        "research_focus": [
            "baseline without rationale",
            "single-rationale distillation",
            "multi-rationale distillation",
            "diverse multi-rationale distillation"
        ],

        "possible_datasets": [
            "GSM8K",
            "SVAMP",
            "StrategyQA"

        ],
        "possible_metrics": [
            "accuracy",
            "precision",
            "recall",
            "F1-score",
            "confusion matrix"
        ],

        "current_user_background": (
            "The user has experience with Python, PyTorch, text classification, "
            "DistilBERT fine-tuning, data cleaning, and evaluation metrics."
        )
    }