# v1.8 Evaluation Guide

This directory turns the six local FYP papers into repeatable evaluation
inputs. Tests and evaluations answer different questions:

- Unit tests ask whether code follows its contracts.
- Retrieval evaluation asks whether relevant evidence appears sufficient.
- Live Agent evaluation measures structured-output reliability and runtime cost.
- None of these automatically proves that every generated claim is correct.

## 1. Inspect the evaluation manifest (no model calls)

```bash
.venv/bin/python evaluate_agent.py
```

This validates and lists `paper_analysis_cases.json`. It does not load the
Embedding model and does not call the LLM API.

## 2. Run the local retrieval-sufficiency baseline

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python evaluate_context_sufficiency.py
```

This parses the six PDFs and uses the locally cached Embedding model. It runs
one relevant query and one deliberately unrelated query for each paper.

Baseline recorded on 2026-08-03:

- Positive cases judged sufficient: 6/6
- Negative cases judged insufficient: 6/6
- Total decisions matching their expected labels: 12/12

This is a small calibration set, not a general retrieval benchmark. The
queries are known examples and should not be used to claim universal accuracy.

## Live smoke baseline

One end-to-end case was run on 2026-08-04 with the same 11,881-character
prompt and the same paper evidence:

- `deepseek-ai/DeepSeek-V4-Pro`: timed out after 60.29 seconds.
- `deepseek-ai/DeepSeek-V4-Flash`: valid on the first attempt in 31.65 seconds.
- Flash provider usage: 3,444 prompt tokens, 1,829 completion tokens, 5,273
  total tokens.
- Flash result: retrieval sufficient, three citation identities valid, no
  structured-output repair required.

The Pro health check succeeded on a tiny prompt, so this comparison does not
prove that the Pro endpoint is unavailable. It shows that Flash was the more
responsive choice for this project's current long structured prompt.

## 3. Run one live end-to-end smoke case

```bash
.venv/bin/python evaluate_agent.py --live --limit 1
```

`--live` is intentionally explicit because it calls the configured LLM and
therefore uses time and tokens. After the one-case smoke test works, run all
six and save the report:

```bash
.venv/bin/python evaluate_agent.py \
  --live \
  --output evaluations/paper_analysis_live_report.json
```

## Metrics

- `structured_valid_rate`: final output passed the Pydantic contract.
- `first_pass_valid_rate`: output passed without a repair call.
- `repair_attempt_rate`: fraction of cases requiring the one allowed repair.
- `repair_success_rate`: repaired successfully / repairs attempted.
- `fallback_rate`: final output still failed and used safe text fallback.
- `citation_identity_valid_rate`: cited `source + chunk_id` existed in that
  turn's retrieved evidence.
- `retrieval_sufficient_rate`: the Context Evaluator accepted the evidence.
- `total_model_calls`, token metrics and latency metrics: actual runtime cost.

Citation identity is deliberately not called semantic groundedness. A real
chunk can be cited for a claim it does not support. Measuring that requires
human labels or a separate claim-evidence evaluator in a later version.
