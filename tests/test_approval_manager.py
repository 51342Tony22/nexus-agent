import unittest
from unittest.mock import patch

from agent import FYPResearchAgent
from approval_manager import ApprovalManager, tool_call_digest
from context_builder import build_context
from memory import get_FYP_memory
from prompts import build_user_prompt_sections
from skills import select_skill


class ApprovalManagerTests(unittest.TestCase):
    def test_digest_is_stable_for_argument_order(self):
        first = tool_call_digest("write", {"b": 2, "a": 1})
        second = tool_call_digest("write", {"a": 1, "b": 2})
        self.assertEqual(first, second)

    def test_duplicate_pending_call_reuses_request(self):
        manager = ApprovalManager()
        first = manager.create_request(
            tool_name="import_paper_pdf",
            arguments={"paper_id": "a" * 40},
            risk_level="local_write",
            reason="Explicit approval is required.",
        )
        second = manager.create_request(
            tool_name="import_paper_pdf",
            arguments={"paper_id": "a" * 40},
            risk_level="local_write",
            reason="Explicit approval is required.",
        )
        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(len(manager.list_requests(status="pending")), 1)

    def test_approval_is_single_use_and_rejection_cannot_execute(self):
        manager = ApprovalManager()
        approved_request = manager.create_request(
            tool_name="write",
            arguments={"path": "allowed.txt"},
            risk_level="local_write",
            reason="Writes a local file.",
        )
        manager.authorize(approved_request.request_id)
        manager.complete(
            approved_request.request_id,
            execution_status="success",
        )
        with self.assertRaises(ValueError):
            manager.authorize(approved_request.request_id)

        rejected_request = manager.create_request(
            tool_name="write",
            arguments={"path": "rejected.txt"},
            risk_level="local_write",
            reason="Writes a local file.",
        )
        manager.reject(rejected_request.request_id)
        with self.assertRaises(ValueError):
            manager.authorize(rejected_request.request_id)


class AgentApprovalIntegrationTests(unittest.TestCase):
    def make_import_plan(self):
        return [{
            "step_id": 1,
            "description": "Import one paper.",
            "tool_name": "import_paper_pdf",
            "tool_args": {"paper_id": "a" * 40},
        }]

    def test_unconfirmed_write_creates_request_without_execution(self):
        agent = FYPResearchAgent.__new__(FYPResearchAgent)
        with patch("agent.run_tool") as runner:
            observation = agent.execute_plan(
                self.make_import_plan(),
                user_input="分析这篇论文",
            )[0]

        runner.assert_not_called()
        request = observation["permission"]["approval_request"]
        self.assertEqual(request["status"], "pending")
        self.assertEqual(request["tool_name"], "import_paper_pdf")
        self.assertEqual(request["arguments"], {"paper_id": "a" * 40})

    def test_approval_executes_stored_arguments_once(self):
        agent = FYPResearchAgent.__new__(FYPResearchAgent)
        observation = agent.execute_plan(
            self.make_import_plan(),
            user_input="分析这篇论文",
        )[0]
        request_id = observation["permission"]["approval_request"]["request_id"]
        fake_result = {
            "tool_used": "import_paper_pdf",
            "tool_args": {"paper_id": "a" * 40},
            "tool_result": {"imported": True, "error": None},
            "status": "success",
            "error": None,
            "latency_seconds": 0.01,
        }

        resumed_result = {"response": "resumed answer"}
        with (
            patch("agent.run_tool", return_value=fake_result) as runner,
            patch.object(agent, "run", return_value=resumed_result) as run,
        ):
            result = agent.approve_tool_request(request_id)

        called_tool = runner.call_args.args[0]
        self.assertEqual(called_tool["name"], "import_paper_pdf")
        self.assertEqual(called_tool["args"], {"paper_id": "a" * 40})
        self.assertEqual(result["approval_request"]["status"], "executed")
        self.assertTrue(result["resumed"])
        self.assertEqual(result["agent_result"], resumed_result)
        resumed_kwargs = run.call_args.kwargs
        self.assertEqual(resumed_kwargs["_resume_execution_plan"][0]["tool_name"], "import_paper_pdf")
        self.assertEqual(
            resumed_kwargs["_resume_tool_observations"][0]["status"],
            "success",
        )
        with self.assertRaises(ValueError):
            agent.approve_tool_request(request_id)

    def test_negated_write_does_not_create_approvable_request(self):
        agent = FYPResearchAgent.__new__(FYPResearchAgent)
        observation = agent.execute_plan(
            self.make_import_plan(),
            user_input="不要下载论文",
        )[0]

        self.assertNotIn(
            "approval_request",
            observation["permission"],
        )
        self.assertEqual(agent.list_tool_approval_requests(), [])

    def test_approval_control_plane_state_does_not_enter_llm_prompt(self):
        agent = FYPResearchAgent.__new__(FYPResearchAgent)
        observation = agent.execute_plan(
            self.make_import_plan(),
            user_input="分析这篇论文",
        )[0]
        request = observation["permission"]["approval_request"]
        user_input = "分析这篇论文"
        context = build_context(
            user_input=user_input,
            memory=get_FYP_memory(),
            skill=select_skill(user_input),
            tool_observations=[observation],
        )
        tool_section = build_user_prompt_sections(context)["tools"]

        self.assertIn('"approval_required": true', tool_section)
        self.assertNotIn(request["request_id"], tool_section)
        self.assertNotIn(request["argument_digest"], tool_section)
        self.assertNotIn('"paper_id"', tool_section)


if __name__ == "__main__":
    unittest.main()
