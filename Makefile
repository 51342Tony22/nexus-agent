PYTHON ?= .venv/bin/python
RUFF_CACHE_DIR ?= /tmp/fyp-mini-agent-ruff-cache
PIP_AUDIT_CACHE_DIR ?= /tmp/fyp-mini-agent-pip-audit-cache

.PHONY: check lint security test audit

check: lint security test

lint:
	RUFF_CACHE_DIR=$(RUFF_CACHE_DIR) $(PYTHON) -m ruff check .

security:
	$(PYTHON) -m bandit -q -lll -r . -x ./.venv,./tests

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests -q

audit:
	$(PYTHON) -m pip_audit --cache-dir $(PIP_AUDIT_CACHE_DIR) -r requirements-lock.txt
