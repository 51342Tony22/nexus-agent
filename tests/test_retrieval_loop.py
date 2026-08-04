import unittest
from unittest.mock import Mock

from agent import FYPResearchAgent
from retrieval_loop import (
    build_refined_query,
    merge_retrieval_observations,
    should_retry,
)


def make_chunk(chunk_id, content, *, embedding_score=None):
    source_scores = {}
    sources = []
    if embedding_score is not None:
        source_scores["embedding"] = embedding_score
        sources = ["embedding"]

    return {
        "chunk_id": chunk_id,
        "start": chunk_id * 100,
        "end": chunk_id * 100 + len(content),
        "content": content,
        "retrieval_sources": sources,
        "source_ranks": {
            source: 1
            for source in sources
        },
        "source_scores": source_scores,
    }


def make_observation(query, chunks, *, step_id=1):
    return {
        "step_id": step_id,
        "description": "Retrieve evidence.",
        "tool_used": "retrieve_from_file",
        "tool_args": {
            "file_path": "paper.pdf",
            "query": query,
            "top_k": 3,
        },
        "tool_result": {
            "file_path": "paper.pdf",
            "file_type": ".pdf",
            "character_count": 4000,
            "retrieval": {
                "query": query,
                "expanded_query": query,
                "retrieval_method": "Hybrid (BM25+Embedding)",
                "fusion_method": "RRF",
                "embedding_model": "fake-model",
                "candidate_k": 3,
                "bm25_candidates": 3,
                "embedding_candidates": 3,
                "rank_constant": 60,
                "retrieved_chunks": chunks,
            },
        },
    }


class RetrievalLoopTests(unittest.TestCase):
    def setUp(self):
        self.original_query = "bidirectional KL divergence consistency"
        self.decision = {
            "needs_retrieval": True,
            "source": "paper.pdf",
            "query": self.original_query,
        }
        self.plan = [{
            "step_id": 1,
            "description": "Retrieve evidence.",
            "tool_name": "retrieve_from_file",
            "tool_args": {
                "file_path": "paper.pdf",
                "query": self.original_query,
                "top_k": 3,
            },
        }]

    def test_refined_query_focuses_missing_terms_and_intent(self):
        evaluation = {
            "signals": {
                "matched_query_terms": ["bidirectional"],
                "missing_query_terms": ["kl", "divergence", "consistency"],
            }
        }

        refinement = build_refined_query(
            "how does bidirectional KL divergence ensure consistency",
            evaluation,
        )

        self.assertIn("kl", refinement["refined_query"])
        self.assertIn("consistency", refinement["refined_query"])
        self.assertIn("mechanism", refinement["refined_query"])
        self.assertEqual(refinement["intent"], "mechanism")

    def test_retry_guard_requires_rewrite_action_and_available_budget(self):
        evaluation = {
            "status": "insufficient",
            "recommended_action": "rewrite_query",
        }

        self.assertTrue(should_retry(evaluation, retries_used=0, max_retries=1))
        self.assertFalse(should_retry(evaluation, retries_used=1, max_retries=1))

        evaluation["recommended_action"] = "fix_source"
        self.assertFalse(should_retry(evaluation, retries_used=0, max_retries=1))

    def test_merge_deduplicates_chunks_and_tracks_attempts(self):
        shared = make_chunk(1, "shared evidence " * 10, embedding_score=0.80)
        stronger_shared = make_chunk(
            1,
            "shared evidence " * 10,
            embedding_score=0.86,
        )
        second = make_chunk(2, "new evidence " * 10, embedding_score=0.84)

        merged = merge_retrieval_observations(
            [
                make_observation("first query", [shared]),
                make_observation("second query", [stronger_shared, second], step_id=2),
            ],
            original_query=self.original_query,
            max_chunks=6,
        )
        retrieval = merged["tool_result"]["retrieval"]

        self.assertEqual(retrieval["attempt_count"], 2)
        self.assertEqual(len(retrieval["retrieved_chunks"]), 2)
        self.assertEqual(
            retrieval["retrieved_chunks"][0]["retrieval_attempts"],
            [1, 2],
        )
        self.assertEqual(
            retrieval["retrieved_chunks"][0]["source_scores"]["embedding"],
            0.86,
        )

    def test_agent_retries_merges_evidence_and_stops_when_sufficient(self):
        initial = make_observation(
            self.original_query,
            [make_chunk(1, "Bidirectional training signal. " * 5)],
        )
        retry = make_observation(
            "kl divergence consistency mechanism",
            [
                make_chunk(
                    2,
                    "KL divergence improves consistency across rationales. " * 4,
                )
            ],
            step_id=2,
        )
        agent = FYPResearchAgent.__new__(FYPResearchAgent)
        agent.execute_plan = Mock(return_value=[retry])

        plan, observations, evaluation, trace = agent.run_retrieval_loop(
            retrieval_decision=self.decision,
            execution_plan=list(self.plan),
            tool_observations=[initial],
        )

        self.assertEqual(agent.execute_plan.call_count, 1)
        self.assertEqual(len(plan), 2)
        self.assertTrue(evaluation["is_sufficient"])
        self.assertEqual(trace["retries_used"], 1)
        self.assertEqual(trace["stopped_reason"], "sufficient_after_retry")
        self.assertEqual(trace["final_evidence"], "merged_attempts")
        self.assertEqual(
            observations[0]["tool_result"]["retrieval"]["attempt_count"],
            2,
        )

    def test_agent_stops_after_retry_budget_is_exhausted(self):
        initial = make_observation(
            self.original_query,
            [make_chunk(1, "unrelated hardware details " * 5)],
        )
        retry = make_observation(
            "kl divergence consistency mechanism",
            [make_chunk(2, "still unrelated material " * 5)],
            step_id=2,
        )
        agent = FYPResearchAgent.__new__(FYPResearchAgent)
        agent.execute_plan = Mock(return_value=[retry])

        _, _, evaluation, trace = agent.run_retrieval_loop(
            retrieval_decision=self.decision,
            execution_plan=list(self.plan),
            tool_observations=[initial],
        )

        self.assertFalse(evaluation["is_sufficient"])
        self.assertEqual(agent.execute_plan.call_count, 1)
        self.assertEqual(trace["retries_used"], 1)
        self.assertEqual(trace["stopped_reason"], "max_retries_reached")


if __name__ == "__main__":
    unittest.main()
