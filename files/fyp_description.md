# CL-1 Reasoning Distillation for Compact Language Models

This project studies whether compact language models can improve their reasoning ability by learning from teacher-generated rationales.

The main experimental conditions are:

1. Baseline without rationale
2. Single-rationale distillation
3. Multi-rationale distillation
4. Diverse multi-rationale distillation

Possible datasets include GSM8K, SVAMP, and StrategyQA.

Possible evaluation metrics include accuracy, precision, recall, F1-score, exact match, and confusion matrix.

The expected contribution is to investigate whether rationale diversity helps compact models generalize better on reasoning tasks.