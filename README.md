# Nexus Agent

A modular research assistant for literature retrieval and analysis. Nexus Agent combines scholarly search, local PDF/Markdown retrieval, persistent memory, typed tools, and bounded execution loops. Previously named **FYP Research Mini Agent**; the internal Python class remains `FYPResearchAgent` for compatibility.

## Architecture

```mermaid
flowchart LR
    U[Research request] --> S[Skill routing and planner]
    S --> T[Typed tool registry and permission checks]
    T --> R[Scholarly search and local document retrieval]
    R --> C[Budgeted context and selected memory]
    C --> L[Configured language model]
    L --> V[Schema and citation identity checks]
    V --> O[Answer or bounded repair/fallback]
```

- **Hybrid retrieval:** BM25 and Sentence Transformers embeddings fused with reciprocal rank fusion, FAISS indexing, and BM25 fallback.
- **Bounded execution:** evidence-sufficiency checks, query refinement, and observation-driven tool loops with hard limits.
- **Context and memory:** separate character budgets, recent history, relevance-filtered persistent memories, and explicit memory controls.
- **Typed tool boundaries:** Pydantic schemas, deterministic permission decisions, and single-use approvals bound to exact arguments.
- **Structured analysis:** output validation, source/chunk identity checks, one bounded repair path, and fallback.
- **Integrations:** Semantic Scholar search, selected open-access PDF imports, and an allowlisted read-only MCP library.

## Quick start

The checked-in lock files describe the Python 3.14 environment used by this project.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
cp .env.example .env
# Edit .env locally to configure the provider and model.
python main.py
```

Interactive commands include `paper-json <request>`, `memory-search <query>`, `memories`, `approvals`, `approve <request_id>`, and `memory-context on/off/status`. Use `exit` to quit. Model-backed requests call the configured provider; they can incur charges.

## Verification

```bash
python -m pip install -r requirements-dev-lock.txt
make check
```

The deterministic gate runs Ruff, Bandit, and unit tests without paid model calls. The repository also contains a GitHub Actions gate. See [HARNESS_ENGINEERING.md](HARNESS_ENGINEERING.md) for the architecture and [evaluations/README.md](evaluations/README.md) for the evaluation protocol.

Saved validation covers six papers and 12 relevant/irrelevant calibration queries. One separately recorded live smoke case passed structured-output and three citation-identity checks on its first attempt. These are small feasibility checks, not evidence of universal retrieval accuracy or claim-level factual correctness.

## Scope and limits

This is an independent research/learning project. Routing and planning are primarily rule based; automatic rolling-summary updates are not implemented. Context budgets are mainly character based with token estimates. Citation identity validation does not prove that a source supports every generated claim.

Provider credentials and local memory/index state are excluded from Git. Existing paper files are research inputs, not project-authored publications; review upstream redistribution terms before making this private repository public. The mathematical reasoning distillation study is a separate project.
