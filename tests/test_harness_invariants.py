"""Mechanical repository invariants for the v2.0 development Harness."""

import ast
import unittest
from pathlib import Path

from skills import SKILL_REGISTRY
from tool_registry import TOOL_SPECS


PROJECT_DIR = Path(__file__).resolve().parents[1]


class RepositoryHarnessTests(unittest.TestCase):
    def test_required_harness_files_exist(self):
        required = {
            "AGENTS.md",
            "HARNESS_ENGINEERING.md",
            "Makefile",
            "pyproject.toml",
            "requirements-dev.txt",
            ".github/pull_request_template.md",
            ".github/workflows/ci.yml",
        }
        missing = [
            path
            for path in sorted(required)
            if not (PROJECT_DIR / path).is_file()
        ]
        self.assertEqual(missing, [])

    def test_secrets_and_local_state_are_gitignored(self):
        gitignore = (PROJECT_DIR / ".gitignore").read_text(encoding="utf-8")
        entries = {
            line.strip()
            for line in gitignore.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn(".env", entries)
        self.assertIn(".venv/", entries)
        self.assertIn("data/", entries)

    def test_every_non_read_only_tool_requires_confirmation(self):
        for name, spec in TOOL_SPECS.items():
            with self.subTest(tool=name):
                if spec.risk_level != "read_only":
                    self.assertTrue(spec.requires_confirmation)

    def test_every_skill_tool_is_registered(self):
        for skill in SKILL_REGISTRY:
            for tool_name in skill.get("allowed_tools", []):
                with self.subTest(skill=skill["name"], tool=tool_name):
                    self.assertIn(tool_name, TOOL_SPECS)

    def test_mcp_server_does_not_import_agent_or_model_runtime(self):
        source = (PROJECT_DIR / "mcp_library_server.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])

        forbidden = {
            "agent",
            "llm_client",
            "memory_store",
            "permission_manager",
        }
        self.assertEqual(imported_roots.intersection(forbidden), set())


if __name__ == "__main__":
    unittest.main()
