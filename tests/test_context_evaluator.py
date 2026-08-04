import unittest

from context_evaluator import evaluate_retrieval_context


def retrieval_decision(needs_retrieval=True):
    return {
        "needs_retrieval": needs_retrieval,
        "source": "files/paper.pdf",
        "query": "bidirectional KL divergence consistency",
    }


def retrieval_observation(
    chunks,
    *,
    query="bidirectional KL divergence consistency",
    expanded_query="bidirectional KL divergence consistency",
    method="Hybrid (BM25+Embedding)",
):
    return [{
        "tool_used": "retrieve_from_file",
        "tool_result": {
            "retrieval": {
                "query": query,
                "expanded_query": expanded_query,
                "retrieval_method": method,
                "retrieved_chunks": chunks,
            }
        },
    }]


def chunk(content, *, embedding_score=None, sources=None):
    source_scores = {}
    if embedding_score is not None:
        source_scores["embedding"] = embedding_score

    return {
        "content": content,
        "retrieval_sources": sources or [],
        "source_scores": source_scores,
    }


class ContextEvaluatorTests(unittest.TestCase):
    def test_retrieval_not_required_is_not_evaluated(self):
        result = evaluate_retrieval_context(
            retrieval_decision(needs_retrieval=False),
            [],
        )

        self.assertEqual(result["status"], "not_required")
        self.assertIsNone(result["is_sufficient"])
        self.assertEqual(
            result["recommended_action"],
            "answer_without_retrieval",
        )

    def test_strong_query_coverage_is_sufficient(self):
        evidence = (
            "The proposed method minimizes bidirectional KL divergence "
            "to improve consistency between multiple rationales. " * 3
        )
        result = evaluate_retrieval_context(
            retrieval_decision(),
            retrieval_observation([chunk(evidence)]),
        )

        self.assertTrue(result["is_sufficient"])
        self.assertEqual(result["reason_code"], "strong_query_coverage")
        self.assertEqual(result["recommended_action"], "answer")

    def test_semantic_support_can_cover_low_lexical_overlap(self):
        evidence = (
            "The student is trained so that explanations agreeing on the "
            "same output produce compatible probability distributions. " * 3
        )
        result = evaluate_retrieval_context(
            retrieval_decision(),
            retrieval_observation([
                chunk(
                    evidence,
                    embedding_score=0.86,
                    sources=["bm25", "embedding"],
                )
            ]),
        )

        self.assertTrue(result["is_sufficient"])
        self.assertEqual(result["reason_code"], "strong_semantic_support")

    def test_empty_retrieval_requests_query_rewrite(self):
        result = evaluate_retrieval_context(
            retrieval_decision(),
            retrieval_observation([]),
        )

        self.assertFalse(result["is_sufficient"])
        self.assertEqual(result["reason_code"], "no_retrieved_chunks")
        self.assertEqual(result["recommended_action"], "rewrite_query")

    def test_weak_unrelated_evidence_is_insufficient(self):
        evidence = (
            "Quantum hardware uses superconducting qubits and cryogenic "
            "control electronics for physical gate operations. " * 3
        )
        result = evaluate_retrieval_context(
            retrieval_decision(),
            retrieval_observation([
                chunk(
                    evidence,
                    embedding_score=0.72,
                    sources=["bm25", "embedding"],
                )
            ]),
        )

        self.assertFalse(result["is_sufficient"])
        self.assertEqual(result["reason_code"], "weak_evidence_support")
        self.assertEqual(result["recommended_action"], "rewrite_query")

    def test_evaluation_uses_original_query_not_generic_expansion_terms(self):
        evidence = (
            "This method presents an approach, framework, architecture, "
            "algorithm, and methodology for language processing. " * 3
        )
        result = evaluate_retrieval_context(
            {
                "needs_retrieval": True,
                "source": "files/paper.pdf",
                "query": "superconducting qubits",
            },
            retrieval_observation(
                [chunk(evidence)],
                query="superconducting qubits",
                expanded_query=(
                    "superconducting qubits method methodology approach "
                    "framework architecture algorithm"
                ),
            ),
        )

        self.assertFalse(result["is_sufficient"])
        self.assertEqual(
            result["signals"]["evaluation_query"],
            "superconducting qubits",
        )
        self.assertEqual(result["signals"]["query_term_coverage"], 0.0)

    def test_retry_is_still_evaluated_against_original_decision_query(self):
        evidence = (
            "A generic definition and explanation with supporting evidence. " * 3
        )
        result = evaluate_retrieval_context(
            {
                "needs_retrieval": True,
                "source": "files/paper.pdf",
                "query": "superconducting qubits",
            },
            retrieval_observation(
                [
                    chunk(
                        evidence,
                        embedding_score=0.95,
                        sources=["bm25", "embedding"],
                    )
                ],
                query="definition explanation evidence",
                expanded_query="definition explanation evidence",
            ),
        )

        self.assertFalse(result["is_sufficient"])
        self.assertEqual(
            result["signals"]["evaluation_query"],
            "superconducting qubits",
        )
        self.assertFalse(
            result["signals"]["semantic_score_query_aligned"]
        )

    def test_retrieval_tool_error_requests_source_fix(self):
        result = evaluate_retrieval_context(
            retrieval_decision(),
            [{
                "tool_used": "retrieve_from_file",
                "tool_result": {"error": "The PDF is empty."},
            }],
        )

        self.assertFalse(result["is_sufficient"])
        self.assertEqual(result["reason_code"], "retrieval_tool_error")
        self.assertEqual(result["recommended_action"], "fix_source")

    def test_bm25_evidence_can_pass_without_embedding_score(self):
        evidence = (
            "Bidirectional KL divergence enforces consistency across "
            "rationales that share the same answer. " * 3
        )
        result = evaluate_retrieval_context(
            retrieval_decision(),
            retrieval_observation(
                [chunk(evidence, sources=["bm25"])],
                method="BM25 fallback",
            ),
        )

        self.assertTrue(result["is_sufficient"])
        self.assertEqual(result["reason_code"], "strong_query_coverage")


if __name__ == "__main__":
    unittest.main()
