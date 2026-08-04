import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent import FYPResearchAgent
from context_builder import build_context
from memory import get_FYP_memory
from prompts import build_user_prompt_sections
from skills import select_skill
from tool_loop import (
    build_imported_paper_query,
    decide_next_tool_action,
)


PAPER_ID_1 = "a" * 40
PAPER_ID_2 = "b" * 40


def observation(tool_name, result, step_id=1):
    return {
        "step_id": step_id,
        "description": f"Run {tool_name}.",
        "tool_used": tool_name,
        "tool_args": {},
        "tool_result": result,
    }


class ToolLoopDecisionTests(unittest.TestCase):
    def test_search_requires_user_selection_and_never_auto_downloads(self):
        decision = decide_next_tool_action(
            user_input="搜索 reasoning distillation 论文",
            skill={"name": "paper_search_skill", "allowed_tools": ["search_papers"]},
            observations=[observation("search_papers", {
                "papers": [
                    {"paper_id": PAPER_ID_1, "title": "Paper One"},
                    {"paper_id": PAPER_ID_2, "title": "Paper Two"},
                ],
                "error": None,
            })],
            dynamic_steps_used=0,
            max_steps=2,
        )

        self.assertEqual(decision["status"], "needs_user_input")
        self.assertEqual(decision["reason_code"], "paper_selection_required")
        self.assertEqual(len(decision["choices"]), 2)
        self.assertIsNone(decision["next_step"])

    def test_import_plus_analysis_proposes_grounded_file_retrieval(self):
        decision = decide_next_tool_action(
            user_input="下载第1篇并分析方法",
            skill={
                "name": "paper_import_skill",
                "allowed_tools": ["import_paper_pdf", "retrieve_from_file"],
            },
            observations=[observation("import_paper_pdf", {
                "paper_id": PAPER_ID_1,
                "imported": True,
                "file_path": "Imported_Paper.pdf",
                "error": None,
            })],
            dynamic_steps_used=0,
            max_steps=2,
        )

        self.assertEqual(decision["status"], "continue")
        self.assertEqual(decision["next_step"]["tool_name"], "retrieve_from_file")
        self.assertEqual(
            decision["next_step"]["tool_args"]["file_path"],
            "Imported_Paper.pdf",
        )
        self.assertEqual(decision["next_step"]["tool_args"]["query"], "方法")
        self.assertTrue(decision["retrieval_decision_update"]["needs_retrieval"])

    def test_import_without_analysis_stops_and_budget_is_enforced(self):
        imported = observation("import_paper_pdf", {
            "paper_id": PAPER_ID_1,
            "imported": True,
            "file_path": "Imported_Paper.pdf",
            "error": None,
        })
        no_analysis = decide_next_tool_action(
            "下载第1篇",
            {"name": "paper_import_skill", "allowed_tools": []},
            [imported],
            0,
            2,
        )
        exhausted = decide_next_tool_action(
            "下载第1篇并分析方法",
            {"name": "paper_import_skill", "allowed_tools": []},
            [imported],
            2,
            2,
        )

        self.assertEqual(no_analysis["reason_code"], "paper_imported")
        self.assertEqual(exhausted["reason_code"], "max_dynamic_steps_reached")

    def test_failed_import_never_proposes_retrieval(self):
        decision = decide_next_tool_action(
            "下载第1篇并分析方法",
            {"name": "paper_import_skill", "allowed_tools": []},
            [observation("import_paper_pdf", {
                "imported": False,
                "file_path": None,
                "error": {"code": "no_open_access_pdf"},
            })],
            0,
            2,
        )

        self.assertEqual(decision["status"], "stop")
        self.assertEqual(decision["reason_code"], "import_paper_pdf_failed")
        self.assertIsNone(decision["next_step"])

    def test_query_builder_has_content_fallback(self):
        self.assertEqual(
            build_imported_paper_query(f"下载论文 {PAPER_ID_1}", PAPER_ID_1),
            "main ideas methods datasets results relevance to FYP",
        )


class PaperSelectionStateTests(unittest.TestCase):
    def setUp(self):
        self.agent = FYPResearchAgent.__new__(FYPResearchAgent)
        self.agent.last_paper_search_results = [
            {"paper_id": PAPER_ID_1, "title": "One"},
            {"paper_id": PAPER_ID_2, "title": "Two"},
        ]

    def test_nth_result_is_resolved_to_provider_id_for_planning(self):
        planning_input, trace = self.agent.resolve_paper_selection(
            "下载第2篇并分析方法"
        )

        self.assertIn(PAPER_ID_2, planning_input)
        self.assertEqual(trace["status"], "resolved")
        self.assertEqual(trace["selected_index"], 2)

    def test_out_of_range_selection_does_not_fabricate_an_id(self):
        planning_input, trace = self.agent.resolve_paper_selection("下载第3篇")

        self.assertEqual(planning_input, "下载第3篇")
        self.assertEqual(trace["status"], "index_out_of_range")
        self.assertIsNone(trace["paper_id"])


class AgentToolLoopTests(unittest.TestCase):
    def test_skill_permission_guard_blocks_proposed_dynamic_tool(self):
        agent = FYPResearchAgent.__new__(FYPResearchAgent)
        agent.last_paper_search_results = []
        agent.execute_plan = Mock()
        imported = observation("import_paper_pdf", {
            "paper_id": PAPER_ID_1,
            "imported": True,
            "file_path": "Imported_Paper.pdf",
            "error": None,
        })

        _, observations, _, trace = agent.run_observation_tool_loop(
            user_input="下载论文并分析方法",
            skill={"name": "restricted", "allowed_tools": ["import_paper_pdf"]},
            retrieval_decision={"needs_retrieval": False},
            execution_plan=[],
            tool_observations=[imported],
        )

        self.assertEqual(trace["stopped_reason"], "tool_not_allowed")
        self.assertEqual(len(observations), 1)
        agent.execute_plan.assert_not_called()

    def test_search_select_import_and_retrieve_across_two_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = FYPResearchAgent(
                memory_store_path=str(Path(directory) / "memories.json")
            )
            captured_contexts = []

            def fake_llm(context):
                captured_contexts.append(context)
                return {
                    "content": "测试回答",
                    "latency": 0.0,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "context_stats": {},
                }

            def fake_run_tool(tool_call):
                name = tool_call["name"]
                args = tool_call["args"]
                if name == "search_papers":
                    result = {
                        "provider": "Semantic Scholar",
                        "query": "reasoning distillation",
                        "result_count": 1,
                        "papers": [{
                            "paper_id": PAPER_ID_1,
                            "title": "A Retrieved Paper",
                            "year": 2024,
                            "url": "https://example.org/paper",
                        }],
                        "error": None,
                    }
                elif name == "import_paper_pdf":
                    result = {
                        "provider": "Semantic Scholar",
                        "paper_id": args["paper_id"],
                        "paper": {"title": "A Retrieved Paper"},
                        "imported": True,
                        "already_existed": False,
                        "file_path": "A_Retrieved_Paper_aaaaaaaa.pdf",
                        "bytes_written": 1234,
                        "error": None,
                    }
                elif name == "retrieve_from_file":
                    result = {
                        "file_path": args["file_path"],
                        "file_type": ".pdf",
                        "retrieval": {
                            "query": args["query"],
                            "expanded_query": args["query"],
                            "retrieval_method": "Hybrid (BM25+Embedding)",
                            "fusion_method": "RRF",
                            "semantic_score_query_aligned": True,
                            "retrieved_chunks": [{
                                "chunk_id": 0,
                                "start": 0,
                                "end": 200,
                                "retrieval_sources": ["bm25", "embedding"],
                                "source_ranks": {"bm25": 1, "embedding": 1},
                                "source_scores": {"embedding": 0.90},
                                "content": "方法 method architecture evidence " * 8,
                            }],
                        },
                    }
                else:
                    raise AssertionError(f"Unexpected tool: {name}")
                return {
                    "tool_used": name,
                    "tool_args": args,
                    "tool_result": result,
                }

            agent.call_llm = fake_llm
            with patch("agent.run_tool", side_effect=fake_run_tool):
                search_result = agent.run("搜索 reasoning distillation 论文")
                analysis_result = agent.run("下载第1篇并分析方法")

        self.assertEqual(
            search_result["agent_tool_loop"]["status"],
            "needs_user_input",
        )
        self.assertEqual(len(agent.last_paper_search_results), 1)
        self.assertEqual(
            analysis_result["debug"]["tools_used"],
            ["import_paper_pdf", "retrieve_from_file"],
        )
        self.assertEqual(
            analysis_result["debug"]["paper_selection"]["paper_id"],
            PAPER_ID_1,
        )
        self.assertEqual(
            analysis_result["agent_tool_loop"]["dynamic_steps_used"],
            1,
        )
        self.assertTrue(analysis_result["context_evaluation"]["is_sufficient"])
        self.assertEqual(len(captured_contexts[-1]["evidence"]), 1)


class ToolLoopContextTests(unittest.TestCase):
    def test_tool_loop_has_an_independent_prompt_budget(self):
        context = build_context(
            user_input="搜索论文",
            memory=get_FYP_memory(),
            skill=select_skill("搜索论文"),
            agent_tool_loop={
                "status": "needs_user_input",
                "stopped_reason": "paper_selection_required",
                "iterations": [{"choices": [{"title": "x" * 2000}]}],
            },
        )
        context["context_budget"]["agent_tool_loop"] = 250
        sections = build_user_prompt_sections(context)

        self.assertIn("agent_tool_loop", sections)
        self.assertIn("[context truncated]", sections["agent_tool_loop"])
        self.assertIn("paper_selection_required", sections["agent_tool_loop"])


if __name__ == "__main__":
    unittest.main()
