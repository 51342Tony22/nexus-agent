# v2.0 Harness Engineering

## Purpose

The Agent core chooses actions. The Harness makes those actions legible,
bounded, testable, reviewable, and recoverable. This project uses two layers:

- Runtime Harness: tool contracts, permission decisions, context budgets,
  memory policy, evaluators, retry bounds, structured errors, and MCP trust.
- Development Harness: `AGENTS.md`, static checks, tests, pull requests, CI,
  branch protection, and human review.

## The repository feedback loop

```text
AGENTS.md -> code change -> static scan -> tests -> pull request
          -> CI gate -> human judgment -> merge
          -> recurring feedback becomes a rule/test
```

`AGENTS.md` communicates intent; it is not enforcement. Ruff, Bandit,
architecture tests, tool contracts, and CI enforce machine-checkable parts.

## Local commands

```bash
# Install development-only tools
.venv/bin/python -m pip install -r requirements-dev-lock.txt

# Required deterministic gate
make check

# Dependency vulnerability audit (run when dependencies change)
make audit
```

The short requirements files express direct dependency intent; the lock files
freeze the complete resolved graph. CI uses Python 3.14 and installs the full
development lock, matching the version used to resolve and verify it locally.

The default gate does not call a live LLM. Live six-paper evaluation is a
separate, explicit experiment because it costs money and depends on a network
provider.

## Pull-request contract

`.github/pull_request_template.md` asks for scope, architecture, risk,
verification evidence, and the decision that still needs human judgment. A PR
is a review packet, not a correctness guarantee.

`.github/workflows/ci.yml` runs the deterministic gate in a clean environment.
The workflow becomes a real merge gate only after the repository is pushed to
GitHub and `deterministic-quality-gate` is configured as a required status
check for `main`.

Recommended `main` rules:

1. Require a pull request before merging.
2. Require `deterministic-quality-gate` to pass.
3. Require all conversations to be resolved.
4. Block force pushes and branch deletion.
5. For a team repository, require one approval and optionally CODEOWNERS.

## Human approval boundaries

Human approval happens in different places:

- Runtime: before a sensitive, external-write, code-execution, or irreversible
  tool action.
- Change delivery: before merging a PR that changes architecture or policy.
- Deployment: before a workflow receives production secrets or deploys.

An approval must be bound to the exact tool name and validated arguments. It
must not authorize a later call whose arguments were silently changed.

The local CLI exposes the approval lifecycle:

```text
approvals                 list pending requests
approve <request_id>      execute the exact stored call once
reject <request_id>       reject it permanently
```

Approval is a resumable control-plane pause rather than the end of the turn:

```text
pending request
  -> approve exact tool name + validated arguments
  -> execute once
  -> create ToolObservation
  -> resume tool loop / retrieval / context building / LLM answer
```

The resumed run receives the completed observation directly, so it does not
re-plan or execute the approved tool a second time.

The full request is Harness control-plane state and is not sent to the LLM.
The model only sees that an action was blocked and human approval is required.
