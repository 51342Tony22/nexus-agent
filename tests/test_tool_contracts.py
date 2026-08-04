import unittest
from unittest.mock import patch

from agent import FYPResearchAgent
from context_builder import build_context
from memory import get_FYP_memory
from permission_manager import check_tool_permission
from prompts import build_user_prompt_sections
from skills import SKILL_REGISTRY, select_skill
from tool_contracts import ToolInputValidationError
from tool_registry import (
    TOOL_REGISTRY,
    TOOL_SPECS,
    build_tool_call,
    get_tool_spec,
    run_tool,
)


class ToolSpecTests(unittest.TestCase):
    def test_retrieval_spec_describes_schema_and_risk(self):
        spec = get_tool_spec("retrieve_from_file")

        self.assertIsNotNone(spec)
        self.assertEqual(spec.risk_level, "read_only")
        self.assertFalse(spec.requires_confirmation)
        schema = spec.input_schema.model_json_schema()
        self.assertEqual(
            set(schema["required"]),
            {"file_path", "query"},
        )
        self.assertEqual(schema["properties"]["top_k"]["maximum"], 10)

    def test_build_tool_call_validates_and_normalizes_arguments(self):
        tool_call = build_tool_call(
            "retrieve_from_file",
            {
                "file_path": "  ReAct.md  ",
                "query": "  method  ",
            },
        )

        self.assertEqual(tool_call["args"]["file_path"], "ReAct.md")
        self.assertEqual(tool_call["args"]["query"], "method")
        self.assertEqual(tool_call["args"]["top_k"], 3)

    def test_invalid_name_or_range_is_rejected_before_execution(self):
        cases = [
            {
                "file_path": "ReAct.md",
                "query": "method",
                "top-k": 3,
            },
            {
                "file_path": "ReAct.md",
                "query": "method",
                "top_k": 1000,
            },
        ]

        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ToolInputValidationError):
                    build_tool_call("retrieve_from_file", arguments)

    def test_paper_tool_specs_validate_provider_arguments(self):
        search_spec = get_tool_spec("search_papers")
        detail_spec = get_tool_spec("get_paper_details")
        import_spec = get_tool_spec("import_paper_pdf")

        self.assertEqual(search_spec.risk_level, "read_only")
        self.assertEqual(detail_spec.risk_level, "read_only")
        self.assertEqual(import_spec.risk_level, "local_write")
        self.assertTrue(import_spec.requires_confirmation)

        search_call = build_tool_call(
            "search_papers",
            {"query": "  agentic RAG  ", "year": "2024-"},
        )
        detail_call = build_tool_call(
            "get_paper_details",
            {"paper_id": "A" * 40},
        )
        self.assertEqual(search_call["args"]["query"], "agentic RAG")
        self.assertEqual(search_call["args"]["limit"], 5)
        self.assertEqual(detail_call["args"]["paper_id"], "a" * 40)

        invalid_calls = [
            ("search_papers", {"query": "x", "limit": 5}),
            ("search_papers", {"query": "valid", "limit": 11}),
            ("search_papers", {"query": "valid", "year": "last year"}),
            ("get_paper_details", {"paper_id": "not-an-id"}),
        ]
        for tool_name, arguments in invalid_calls:
            with self.subTest(tool_name=tool_name, arguments=arguments):
                with self.assertRaises(ToolInputValidationError):
                    build_tool_call(tool_name, arguments)

    def test_tool_specs_are_the_single_complete_registry(self):
        self.assertEqual(set(TOOL_REGISTRY), set(TOOL_SPECS))
        for name, spec in TOOL_SPECS.items():
            with self.subTest(tool_name=name):
                self.assertEqual(spec.name, name)
                self.assertIs(TOOL_REGISTRY[name], spec.function)
                self.assertIsNotNone(spec.input_schema)
                self.assertEqual(spec.risk_level in {
                    "read_only",
                    "local_write",
                    "external_write",
                }, True)
                self.assertIsInstance(
                    spec.input_schema.model_json_schema(),
                    dict,
                )

        allowed_tools = {
            tool_name
            for skill in SKILL_REGISTRY
            for tool_name in skill.get("allowed_tools", [])
        }
        self.assertTrue(allowed_tools.issubset(TOOL_SPECS))

    def test_remaining_teaching_tool_arguments_are_validated(self):
        dataset_call = build_tool_call(
            "compare_datasets",
            {"dataset_names": ["gsm8k", "Strategy QA", "gsm8k"]},
        )
        student_call = build_tool_call("suggest_student_models", {})
        pdf_call = build_tool_call(
            "read_pdf_file",
            {"file_path": " paper.pdf ", "max_pages": 10},
        )

        self.assertEqual(
            dataset_call["args"]["dataset_names"],
            ["GSM8K", "StrategyQA"],
        )
        self.assertEqual(student_call["args"]["task_type"], "generation")
        self.assertEqual(pdf_call["args"]["file_path"], "paper.pdf")
        self.assertEqual(pdf_call["args"]["max_characters"], 8000)

        invalid_calls = [
            ("compare_datasets", {"dataset_names": ["unknown"]}),
            ("suggest_teacher_models", {"unexpected": True}),
            ("suggest_student_models", {"task_type": "embedding"}),
            ("generate_experiment_groups", {"extra": "value"}),
            ("read_text_file", {"file_path": ""}),
            ("read_pdf_file", {"file_path": "paper.pdf", "max_pages": 0}),
            (
                "read_pdf_file",
                {"file_path": "paper.pdf", "max_characters": 999999},
            ),
        ]
        for tool_name, arguments in invalid_calls:
            with self.subTest(tool_name=tool_name, arguments=arguments):
                with self.assertRaises(ToolInputValidationError):
                    build_tool_call(tool_name, arguments)


class PermissionManagerTests(unittest.TestCase):
    def test_read_only_tool_does_not_require_confirmation(self):
        decision = check_tool_permission(
            get_tool_spec("search_papers"),
            user_input=None,
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason_code, "confirmation_not_required")

    def test_import_requires_current_explicit_user_intent(self):
        spec = get_tool_spec("import_paper_pdf")

        missing = check_tool_permission(spec, user_input=None)
        explicit = check_tool_permission(spec, user_input="下载第1篇")
        negated = check_tool_permission(
            spec,
            user_input="不要下载论文，只查看信息",
        )

        self.assertFalse(missing.allowed)
        self.assertEqual(missing.reason_code, "confirmation_required")
        self.assertTrue(explicit.allowed)
        self.assertEqual(explicit.confirmation_source, "current_user_message")
        self.assertFalse(negated.allowed)
        self.assertEqual(negated.reason_code, "write_explicitly_negated")


class ToolExecutionContractTests(unittest.TestCase):
    def test_invalid_plan_becomes_observation_instead_of_crashing(self):
        agent = FYPResearchAgent.__new__(FYPResearchAgent)
        plan = [{
            "step_id": 1,
            "description": "Invalid retrieval request.",
            "tool_name": "retrieve_from_file",
            "tool_args": {
                "file_path": "ReAct.md",
                "query": "method",
                "top-k": 3,
            },
        }]

        with patch("agent.run_tool") as mocked_runner:
            result = agent.execute_plan(plan)[0]

        mocked_runner.assert_not_called()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_tool_arguments")
        self.assertFalse(result["error"]["retryable"])
        self.assertIsNone(result["tool_result"])
        self.assertEqual(result["tool_name"], result["tool_used"])
        self.assertEqual(result["arguments"], result["tool_args"])

    def test_uncaught_tool_exception_is_normalized(self):
        def explode():
            raise RuntimeError("simulated failure")

        result = run_tool({
            "name": "explode",
            "function": explode,
            "args": {},
        })

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "tool_execution_error")
        self.assertIn("simulated failure", result["error"]["message"])
        self.assertIsNone(result["tool_result"])
        self.assertGreaterEqual(result["latency_seconds"], 0.0)

    def test_tool_reported_error_is_normalized_without_losing_result(self):
        result = run_tool({
            "name": "provider_tool",
            "function": lambda: {
                "error": {
                    "code": "rate_limited",
                    "message": "Try again later.",
                    "retryable": True,
                }
            },
            "args": {},
        })

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "rate_limited")
        self.assertTrue(result["error"]["retryable"])
        self.assertEqual(
            result["tool_result"]["error"]["code"],
            "rate_limited",
        )

    def test_context_exposes_structured_tool_error(self):
        agent = FYPResearchAgent.__new__(FYPResearchAgent)
        observations = agent.execute_plan([{
            "step_id": 1,
            "description": "Invalid retrieval request.",
            "tool_name": "retrieve_from_file",
            "tool_args": {
                "file_path": "ReAct.md",
                "query": "method",
                "top_k": 0,
            },
        }])
        user_input = "分析 ReAct.md 的方法"
        context = build_context(
            user_input=user_input,
            memory=get_FYP_memory(),
            skill=select_skill(user_input),
            tool_observations=observations,
        )
        tool_section = build_user_prompt_sections(context)["tools"]

        self.assertIn('"status": "error"', tool_section)
        self.assertIn("invalid_tool_arguments", tool_section)

    def test_planner_cannot_grant_itself_local_write_permission(self):
        agent = FYPResearchAgent.__new__(FYPResearchAgent)
        plan = [{
            "step_id": 1,
            "description": "Attempt an unconfirmed paper import.",
            "tool_name": "import_paper_pdf",
            "tool_args": {"paper_id": "a" * 40},
            # This field is deliberately ignored by Permission Manager.
            "permission_granted": True,
        }]

        with patch("agent.run_tool") as mocked_runner:
            result = agent.execute_plan(plan, user_input="分析这篇论文")[0]

        mocked_runner.assert_not_called()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "permission_denied")
        self.assertEqual(
            result["permission"]["reason_code"],
            "confirmation_required",
        )

    def test_explicit_import_request_grants_local_write_permission(self):
        agent = FYPResearchAgent.__new__(FYPResearchAgent)
        plan = [{
            "step_id": 1,
            "description": "Import the selected paper.",
            "tool_name": "import_paper_pdf",
            "tool_args": {"paper_id": "a" * 40},
        }]
        fake_runtime_result = {
            "tool_used": "import_paper_pdf",
            "tool_args": {"paper_id": "a" * 40},
            "tool_result": {
                "imported": True,
                "file_path": "paper.pdf",
                "error": None,
            },
            "status": "success",
            "error": None,
            "latency_seconds": 0.01,
        }

        with patch("agent.run_tool", return_value=fake_runtime_result) as runner:
            result = agent.execute_plan(
                plan,
                user_input="下载论文并保存到项目",
            )[0]

        runner.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["permission"]["allowed"])
        self.assertEqual(
            result["permission"]["reason_code"],
            "explicit_import_request",
        )


if __name__ == "__main__":
    unittest.main()
