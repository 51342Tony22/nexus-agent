import unittest

from agent_evaluator import (
    PROJECT_DIR,
    aggregate_evaluation,
    evaluate_agent_result,
    load_evaluation_cases,
    run_live_evaluation,
)


def fake_result(
    *,
    status="valid",
    attempt_count=1,
    repair_attempted=False,
    source="paper.pdf",
    cited_source="paper.pdf",
    chunk_id=3,
    cited_chunk_id=3,
    sufficient=True,
    tokens=100,
):
    data = None
    if status == "valid":
        data = {
            "evidence_references": [{
                "source": cited_source,
                "chunk_id": cited_chunk_id,
                "claim": "A supported claim.",
            }],
        }
    return {
        "structured_output": {
            "status": status,
            "attempt_count": attempt_count,
            "data": data,
            "error": None if status == "valid" else {"code": "invalid_json"},
        },
        "structured_output_repair": {
            "attempted": repair_attempted,
        },
        "tool_observations": [{
            "tool_result": {
                "file_path": source,
                "retrieval": {
                    "retrieved_chunks": [{"chunk_id": chunk_id}],
                },
            },
        }],
        "context_evaluation": {"is_sufficient": sufficient},
        "usage": {
            "latency": 0.5,
            "prompt_tokens": 70,
            "completion_tokens": 30,
            "total_tokens": tokens,
        },
        "model_call": {"call_count": attempt_count},
        "debug": {
            "context_stats": {
                "total": {
                    "characters": 8000,
                    "estimated_tokens": 2400,
                },
            },
            "retrievals": [{
                "retrieval_method": "Hybrid (BM25+Embedding)",
            }],
        },
    }


class AgentEvaluationCaseTests(unittest.TestCase):
    def test_manifest_contains_six_unique_existing_papers(self):
        cases = load_evaluation_cases()

        self.assertEqual(len(cases), 6)
        self.assertEqual(len({case["case_id"] for case in cases}), 6)
        for case in cases:
            self.assertTrue((PROJECT_DIR / "files" / case["paper"]).is_file())
            self.assertIn(case["paper"], case["question"])

    def test_result_metrics_check_reference_against_runtime_chunks(self):
        case = {
            "case_id": "case-one",
            "paper": "paper.pdf",
            "evaluation_goal": "test",
        }

        measured = evaluate_agent_result(case, fake_result())

        self.assertTrue(measured["structured_valid"])
        self.assertTrue(measured["first_pass_valid"])
        self.assertTrue(measured["citation_identity_valid"])
        self.assertEqual(
            measured["retrieval_method"],
            "Hybrid (BM25+Embedding)",
        )
        self.assertEqual(measured["prompt_estimated_tokens"], 2400)

    def test_fabricated_reference_is_detected_independently(self):
        case = {
            "case_id": "case-one",
            "paper": "paper.pdf",
            "evaluation_goal": "test",
        }

        measured = evaluate_agent_result(
            case,
            fake_result(cited_source="invented.pdf", cited_chunk_id=99),
        )

        self.assertTrue(measured["structured_valid"])
        self.assertFalse(measured["citation_identity_valid"])


class AgentEvaluationAggregateTests(unittest.TestCase):
    def make_case(self, case_id, **overrides):
        result = {
            "case_id": case_id,
            "paper": f"{case_id}.pdf",
            "evaluation_goal": "test",
            "completed": True,
            "structured_status": "valid",
            "structured_valid": True,
            "first_pass_valid": True,
            "repair_attempted": False,
            "repair_success": False,
            "fallback": False,
            "citation_count": 1,
            "citation_identity_valid": True,
            "retrieval_sufficient": True,
            "retrieval_method": "Hybrid (BM25+Embedding)",
            "model_call_count": 1,
            "prompt_characters": 8000,
            "prompt_estimated_tokens": 2400,
            "latency_seconds": 1.0,
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
            "error": None,
        }
        result.update(overrides)
        return result

    def test_aggregate_uses_correct_denominators_and_cost_totals(self):
        cases = [
            self.make_case("one"),
            self.make_case(
                "two",
                first_pass_valid=False,
                repair_attempted=True,
                repair_success=True,
                model_call_count=2,
                total_tokens=180,
                latency_seconds=2.0,
            ),
            self.make_case(
                "three",
                structured_status="fallback",
                structured_valid=False,
                first_pass_valid=False,
                repair_attempted=True,
                repair_success=False,
                fallback=True,
                citation_count=0,
                citation_identity_valid=False,
                retrieval_sufficient=False,
                model_call_count=2,
                total_tokens=None,
                latency_seconds=3.0,
            ),
        ]

        aggregate = aggregate_evaluation(cases)

        self.assertAlmostEqual(aggregate["structured_valid_rate"], 2 / 3)
        self.assertAlmostEqual(aggregate["first_pass_valid_rate"], 1 / 3)
        self.assertAlmostEqual(aggregate["repair_attempt_rate"], 2 / 3)
        self.assertEqual(aggregate["repair_success_rate"], 0.5)
        self.assertEqual(aggregate["citation_identity_valid_rate"], 1.0)
        self.assertEqual(aggregate["total_model_calls"], 5)
        self.assertEqual(aggregate["usage_known_cases"], 2)
        self.assertEqual(aggregate["total_tokens"], 280)
        self.assertEqual(aggregate["average_tokens"], 140)
        self.assertEqual(aggregate["total_latency_seconds"], 6.0)


class LiveEvaluationRunnerTests(unittest.TestCase):
    def test_one_case_failure_does_not_abort_later_cases(self):
        class FakeAgent:
            def __init__(self):
                self.calls = 0
                self.clears = 0

            def run(self, question, output_mode):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("provider secret must not be copied")
                return fake_result(source="second.pdf", cited_source="second.pdf")

            def clear_conversation(self):
                self.clears += 1

        cases = [
            {
                "case_id": "first",
                "paper": "first.pdf",
                "question": "first question",
                "evaluation_goal": "failure isolation",
            },
            {
                "case_id": "second",
                "paper": "second.pdf",
                "question": "second question",
                "evaluation_goal": "continue evaluation",
            },
        ]
        agent = FakeAgent()

        report = run_live_evaluation(cases, agent)

        self.assertFalse(report["case_results"][0]["completed"])
        self.assertEqual(
            report["case_results"][0]["error"]["exception_type"],
            "RuntimeError",
        )
        self.assertNotIn(
            "provider secret",
            str(report["case_results"][0]["error"]),
        )
        self.assertTrue(report["case_results"][1]["completed"])
        self.assertEqual(agent.clears, 2)


if __name__ == "__main__":
    unittest.main()
