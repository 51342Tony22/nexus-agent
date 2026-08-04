import unittest

from retrieval_evaluator import (
    calculate_retrieval_metrics,
    evaluate_retriever,
)


class RetrievalEvaluatorTests(unittest.TestCase):
    def test_calculates_precision_recall_and_hit(self):
        metrics = calculate_retrieval_metrics(
            retrieved_chunk_ids=[2, 5],
            relevant_chunk_ids=[2, 3],
        )

        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["hit"], 1.0)

    def test_empty_retrieval_has_zero_metrics(self):
        metrics = calculate_retrieval_metrics(
            retrieved_chunk_ids=[],
            relevant_chunk_ids=[2],
        )

        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["hit"], 0.0)

    def test_evaluates_multiple_cases_with_macro_average(self):
        cases = [
            {"name": "one", "query": "q1", "relevant_chunk_ids": [1]},
            {"name": "two", "query": "q2", "relevant_chunk_ids": [2]},
        ]

        def fake_retrieve(query, top_k):
            if query == "q1":
                return [{"chunk_id": 1}]
            return [{"chunk_id": 3}]

        report = evaluate_retriever(
            retriever_name="fake",
            cases=cases,
            retrieve=fake_retrieve,
            top_k=1,
        )

        self.assertEqual(report["aggregate"]["precision"], 0.5)
        self.assertEqual(report["aggregate"]["recall"], 0.5)
        self.assertEqual(report["aggregate"]["hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
