import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from agent import FYPResearchAgent
from context_builder import build_context
from memory import get_FYP_memory
from output_contracts import (
    build_output_schema_context,
    validate_paper_analysis_output,
)
from prompts import build_prompt_bundle, build_user_prompt_sections
from skills import select_skill


def valid_payload(source="ReAct.md", chunk_id=2):
    return {
        "summary": "ReAct 将推理轨迹与行动交错起来。",
        "problem": "只推理或只行动都难以处理交互式任务。",
        "method": "模型交替生成 reasoning trace 和 action。",
        "datasets": ["HotpotQA"],
        "results": ["检索到的片段报告该方法提升了任务表现。"],
        "limitations": ["效果依赖行动空间和外部反馈。"],
        "fyp_relevance": "可用于设计带工具反馈的 reasoning distillation。",
        "evidence_references": [{
            "source": source,
            "chunk_id": chunk_id,
            "claim": "该片段说明 reasoning 与 action 被交错使用。",
        }],
    }


class StructuredOutputContractTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [{
            "source": "ReAct.md",
            "chunk_id": 2,
            "content": "ReAct interleaves reasoning traces and actions.",
        }]

    def test_valid_json_is_parsed_and_validated(self):
        result = validate_paper_analysis_output(
            json.dumps(valid_payload(), ensure_ascii=False),
            self.evidence,
        )

        self.assertEqual(result.status, "valid")
        self.assertEqual(result.schema_name, "PaperAnalysisOutput")
        self.assertEqual(result.data["evidence_references"][0]["chunk_id"], 2)

    def test_markdown_json_fence_is_tolerated(self):
        raw = "```json\n" + json.dumps(valid_payload()) + "\n```"

        result = validate_paper_analysis_output(raw, self.evidence)

        self.assertEqual(result.status, "valid")

    def test_invalid_json_and_missing_fields_fall_back(self):
        invalid_json = validate_paper_analysis_output("not json", self.evidence)
        missing_fields = validate_paper_analysis_output(
            json.dumps({"summary": "only one field"}),
            self.evidence,
        )

        self.assertEqual(invalid_json.status, "fallback")
        self.assertEqual(invalid_json.error.code, "invalid_json")
        self.assertEqual(missing_fields.status, "fallback")
        self.assertEqual(
            missing_fields.error.code,
            "schema_validation_failed",
        )

    def test_missing_or_fabricated_evidence_reference_falls_back(self):
        missing = valid_payload()
        missing["evidence_references"] = []
        fabricated = valid_payload(source="invented.pdf", chunk_id=99)

        missing_result = validate_paper_analysis_output(
            json.dumps(missing),
            self.evidence,
        )
        fabricated_result = validate_paper_analysis_output(
            json.dumps(fabricated),
            self.evidence,
        )

        self.assertEqual(
            missing_result.error.code,
            "missing_evidence_references",
        )
        self.assertEqual(
            fabricated_result.error.code,
            "invalid_evidence_reference",
        )

    def test_schema_forbids_unknown_fields(self):
        payload = valid_payload()
        payload["unsupported_extra"] = "must not silently pass"

        result = validate_paper_analysis_output(
            json.dumps(payload),
            self.evidence,
        )

        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.error.code, "schema_validation_failed")


class StructuredOutputPromptTests(unittest.TestCase):
    def build_context(self, structured=True):
        user_input = "分析 ReAct.md 的方法"
        skill = select_skill(user_input)
        schema = (
            build_output_schema_context("paper_analysis", skill)
            if structured
            else None
        )
        return build_context(
            user_input=user_input,
            memory=get_FYP_memory(),
            skill=skill,
            structured_output_schema=schema,
        )

    def test_schema_has_an_independent_prompt_section_and_budget(self):
        context = self.build_context()
        sections = build_user_prompt_sections(context)
        bundle = build_prompt_bundle(context)

        self.assertIn("## Structured Output Schema", sections["output_schema"])
        self.assertIn("PaperAnalysisOutput", sections["output_schema"])
        self.assertIn("只返回一个", bundle["system_prompt"])
        self.assertIn("Return only the JSON object", sections["final_instruction"])
        self.assertIn("output_schema", context["context_budget"])

    def test_default_text_mode_does_not_change_the_prompt(self):
        context = self.build_context(structured=False)
        sections = build_user_prompt_sections(context)
        bundle = build_prompt_bundle(context)

        self.assertNotIn("output_schema", sections)
        self.assertNotIn("Structured Output Schema", bundle["user_prompt"])
        self.assertIn("Answer the current user task", sections["final_instruction"])

    def test_paper_schema_is_only_available_to_analysis_skills(self):
        dataset_skill = select_skill("比较 GSM8K 和 SVAMP 数据集")

        with self.assertRaises(ValueError):
            build_output_schema_context("paper_analysis", dataset_skill)


class StructuredOutputAgentTests(unittest.TestCase):
    def make_agent(self, model_content):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        agent = FYPResearchAgent(
            memory_store_path=str(
                Path(temporary_directory.name) / "memories.json"
            )
        )
        plan = [{
            "step_id": 1,
            "description": "Retrieve relevant chunks.",
            "tool_name": "retrieve_from_file",
            "tool_args": {
                "file_path": "ReAct.md",
                "query": "方法",
                "top_k": 3,
            },
        }]
        observations = [{
            "step_id": 1,
            "description": "Retrieve relevant chunks.",
            "tool_used": "retrieve_from_file",
            "tool_name": "retrieve_from_file",
            "tool_args": plan[0]["tool_args"],
            "status": "success",
            "error": None,
            "latency_seconds": 0.0,
            "tool_result": {
                "file_path": "ReAct.md",
                "file_type": ".md",
                "retrieval": {
                    "query": "方法",
                    "expanded_query": "方法 method architecture",
                    "retrieval_method": "Hybrid (BM25+Embedding)",
                    "fusion_method": "RRF",
                    "semantic_score_query_aligned": True,
                    "retrieved_chunks": [{
                        "chunk_id": 2,
                        "start": 0,
                        "end": 120,
                        "retrieval_sources": ["bm25", "embedding"],
                        "source_ranks": {"bm25": 1, "embedding": 1},
                        "source_scores": {"embedding": 0.91},
                        "content": (
                            "ReAct interleaves reasoning traces and actions. "
                            "The method uses observations from tools."
                        ),
                    }],
                },
            },
        }]
        context_evaluation = {
            "status": "sufficient",
            "is_sufficient": True,
            "reason_code": "strong_semantic_support",
            "reason": "Retrieved evidence supports the answer.",
            "recommended_action": "answer",
            "signals": {"chunks_returned": 1},
        }
        agent.execute_plan = Mock(return_value=observations)
        agent.run_observation_tool_loop = Mock(return_value=(
            plan,
            observations,
            {
                "needs_retrieval": True,
                "query": "方法",
                "source": "ReAct.md",
            },
            {"status": "stopped", "steps": []},
        ))
        agent.run_retrieval_loop = Mock(return_value=(
            plan,
            observations,
            context_evaluation,
            {"attempt_count": 1, "stop_reason": "sufficient"},
        ))
        captured = {"contexts": []}
        model_contents = (
            list(model_content)
            if isinstance(model_content, list)
            else [model_content]
        )

        def fake_llm(context):
            captured["contexts"].append(context)
            call_index = len(captured["contexts"]) - 1
            content = model_contents[min(call_index, len(model_contents) - 1)]
            return {
                "status": "success",
                "content": content,
                "model": "test-model",
                "latency": 0.25,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "context_stats": {},
            }

        agent.call_llm = fake_llm
        return agent, captured

    def test_agent_returns_validated_data_and_human_rendering(self):
        raw_json = json.dumps(valid_payload(), ensure_ascii=False)
        agent, captured = self.make_agent(raw_json)

        result = agent.run(
            "分析 ReAct.md 的方法",
            output_mode="paper_analysis",
        )

        self.assertEqual(result["structured_output"]["status"], "valid")
        self.assertEqual(
            result["structured_output"]["data"]["method"],
            valid_payload()["method"],
        )
        self.assertIn("Method", result["response"])
        self.assertNotEqual(result["response"], raw_json)
        self.assertEqual(
            captured["contexts"][0]["structured_output_schema"]["name"],
            "PaperAnalysisOutput",
        )
        self.assertEqual(result["model_call"]["call_count"], 1)
        self.assertEqual(result["usage"]["total_tokens"], 15)
        self.assertEqual(
            agent.conversation_memory.get_recent_history()[-1]["assistant"],
            result["response"],
        )

    def test_agent_falls_back_without_claiming_invalid_json_is_valid(self):
        agent, _ = self.make_agent("普通文本，不是 JSON")

        result = agent.run(
            "分析 ReAct.md 的方法",
            output_mode="paper_analysis",
        )

        self.assertEqual(result["structured_output"]["status"], "fallback")
        self.assertEqual(
            result["structured_output"]["error"]["code"],
            "invalid_json",
        )
        self.assertIn("安全回退", result["response"])
        self.assertIn("普通文本，不是 JSON", result["response"])
        self.assertEqual(result["structured_output"]["attempt_count"], 2)
        self.assertTrue(result["structured_output_repair"]["attempted"])
        self.assertEqual(result["model_call"]["call_count"], 2)
        self.assertEqual(result["usage"]["latency"], 0.5)
        self.assertEqual(result["usage"]["total_tokens"], 30)

    def test_agent_repairs_invalid_json_once_then_revalidates(self):
        repaired_json = json.dumps(valid_payload(), ensure_ascii=False)
        agent, captured = self.make_agent([
            "第一次不是 JSON",
            repaired_json,
        ])

        result = agent.run(
            "分析 ReAct.md 的方法",
            output_mode="paper_analysis",
        )

        self.assertEqual(result["structured_output"]["status"], "valid")
        self.assertEqual(result["structured_output"]["attempt_count"], 2)
        self.assertTrue(result["structured_output_repair"]["attempted"])
        self.assertEqual(
            result["structured_output_repair"]["initial_error"]["code"],
            "invalid_json",
        )
        self.assertEqual(len(captured["contexts"]), 2)
        repair_context = captured["contexts"][1]
        self.assertEqual(
            repair_context["structured_output_repair"]["previous_candidate"],
            "第一次不是 JSON",
        )
        repair_sections = build_user_prompt_sections(repair_context)
        self.assertIn(
            "## Structured Output Repair",
            repair_sections["output_repair"],
        )
        self.assertEqual(result["usage"]["total_tokens"], 30)


if __name__ == "__main__":
    unittest.main()
