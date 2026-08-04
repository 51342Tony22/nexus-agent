import tempfile
import unittest
from pathlib import Path

from agent import FYPResearchAgent
from memory_write_policy import (
    MAX_PENDING_SUGGESTIONS,
    evaluate_memory_write,
)


class MemoryWritePolicyTests(unittest.TestCase):
    def test_stable_research_profile_is_suggested(self):
        decision = evaluate_memory_write(
            "我的研究方向是 reasoning distillation"
        )

        self.assertEqual(decision["status"], "suggested")
        self.assertEqual(decision["reason_code"], "research_profile")
        self.assertEqual(
            decision["suggestion"]["tags"],
            ["profile", "research"],
        )

    def test_response_preference_is_suggested(self):
        decision = evaluate_memory_write(
            "我希望回答时解释为什么这样设计"
        )

        self.assertEqual(decision["status"], "suggested")
        self.assertEqual(decision["reason_code"], "response_preference")
        self.assertIn("preference", decision["suggestion"]["tags"])

    def test_ordinary_question_is_not_memorable(self):
        decision = evaluate_memory_write("什么是 Agentic RAG？")

        self.assertEqual(decision["status"], "not_memorable")
        self.assertEqual(
            decision["reason_code"],
            "no_stable_memory_signal",
        )
        self.assertIsNone(decision["suggestion"])

    def test_sensitive_information_is_rejected_before_rule_matching(self):
        decision = evaluate_memory_write(
            "请记住我的 API key 是 sk-example-secret-value"
        )

        self.assertEqual(decision["status"], "rejected")
        self.assertEqual(decision["reason_code"], "sensitive_information")
        self.assertIsNone(decision["suggestion"])

    def test_existing_equivalent_memory_is_not_suggested_again(self):
        decision = evaluate_memory_write(
            "我的研究方向是 RAG",
            existing_records=[{
                "memory_id": "mem_existing",
                "content": "我的研究方向是 RAG",
            }],
        )

        self.assertEqual(decision["status"], "duplicate")
        self.assertEqual(decision["duplicate_of"], "mem_existing")

    def test_agent_writes_only_after_suggestion_is_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            agent = FYPResearchAgent(memory_store_path=str(path))

            decision = agent.propose_memory_write(
                "我的学习目标是理解 Agent harness"
            )
            suggestion = decision["suggestion"]

            self.assertEqual(decision["status"], "suggested")
            self.assertFalse(path.exists())
            self.assertEqual(len(agent.list_memory_suggestions()), 1)

            saved = agent.approve_memory_suggestion(
                suggestion["suggestion_id"]
            )

            self.assertTrue(path.exists())
            self.assertEqual(
                saved["source"],
                "user_approved_policy_suggestion",
            )
            self.assertEqual(saved["metadata"]["category"], "long_term_goal")
            self.assertEqual(agent.list_memory_suggestions(), [])

    def test_dismissed_suggestion_is_never_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            agent = FYPResearchAgent(memory_store_path=str(path))
            decision = agent.propose_memory_write(
                "我希望回答时提供代码示例"
            )
            suggestion_id = decision["suggestion"]["suggestion_id"]

            self.assertTrue(agent.dismiss_memory_suggestion(suggestion_id))
            self.assertFalse(agent.dismiss_memory_suggestion(suggestion_id))
            self.assertFalse(path.exists())

    def test_agent_run_returns_suggestion_without_calling_an_external_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            agent = FYPResearchAgent(memory_store_path=str(path))
            agent.call_llm = lambda context: {
                "content": "已了解你的研究方向。",
                "latency": 0.0,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "context_stats": {},
            }

            result = agent.run("我的研究方向是 reasoning distillation")

            self.assertEqual(
                result["memory_suggestion"]["category"],
                "research_profile",
            )
            self.assertEqual(
                result["debug"]["memory_write_decision"]["status"],
                "suggested",
            )
            self.assertFalse(path.exists())

    def test_pending_suggestions_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = FYPResearchAgent(
                memory_store_path=str(Path(directory) / "memories.json")
            )

            for index in range(MAX_PENDING_SUGGESTIONS + 3):
                unique_goal = chr(ord("a") + index) * 24
                agent.propose_memory_write(
                    f"我的学习目标是完成 {unique_goal}"
                )

            self.assertEqual(
                len(agent.list_memory_suggestions()),
                MAX_PENDING_SUGGESTIONS,
            )


if __name__ == "__main__":
    unittest.main()
