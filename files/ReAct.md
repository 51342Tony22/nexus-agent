# ReAct: Synergizing Reasoning and Acting in Language Models


## Problem

Large language models can generate reasoning steps,
but they may suffer from hallucination when solving
knowledge-intensive tasks.


## Method

ReAct combines reasoning and acting.

The model generates:

Thought:
internal reasoning process

Action:
interaction with external tools

Observation:
information returned from tools


The final answer is produced based on the reasoning
and observations.


## Dataset

The paper evaluates ReAct on:

- HotpotQA
- Fever
- ALFWorld


## Result

ReAct improves factual accuracy and reduces hallucination
compared with traditional Chain-of-Thought prompting.