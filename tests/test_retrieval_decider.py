import unittest

from context_builder import build_context
from memory import get_FYP_memory
from planner import create_plan
from prompts import build_prompt_bundle
from retrieval_decider import decide_retrieval
from skills import select_skill


class RetrievalDeciderTests(unittest.TestCase):
    def test_explicit_file_overrides_generic_paper_keywords(self):
        user_input = "分析 ReAct.pdf 中 method 和 dataset"
        skill = select_skill(user_input)
        decision = decide_retrieval(user_input, skill)

        self.assertEqual(skill["name"], "file_analysis_skill")
        self.assertTrue(decision["needs_retrieval"])
        self.assertEqual(decision["source"], "ReAct.pdf")
        self.assertEqual(decision["query"], "method 和 dataset")
        self.assertEqual(decision["reason_code"], "explicit_local_file")

    def test_file_reference_supports_arrow_in_paper_name(self):
        user_input = "分析 1_CotBasic_FromLLM->LM_DIstill.pdf 的方法"
        skill = select_skill(user_input)
        decision = decide_retrieval(user_input, skill)

        self.assertEqual(skill["name"], "file_analysis_skill")
        self.assertEqual(
            decision["source"],
            "1_CotBasic_FromLLM->LM_DIstill.pdf",
        )
        self.assertEqual(decision["query"], "方法")

    def test_general_knowledge_does_not_require_retrieval(self):
        user_input = "解释 Transformer 的基本结构"
        skill = select_skill(user_input)
        decision = decide_retrieval(user_input, skill)
        plan = create_plan(skill, user_input, decision)

        self.assertFalse(decision["needs_retrieval"])
        self.assertEqual(decision["strategy"], "none")
        self.assertIsNone(plan[0]["tool_name"])

    def test_dataset_tool_is_not_document_retrieval(self):
        user_input = "比较 GSM8K 和 SVAMP 数据集"
        skill = select_skill(user_input)
        decision = decide_retrieval(user_input, skill)
        plan = create_plan(skill, user_input, decision)

        self.assertFalse(decision["needs_retrieval"])
        self.assertEqual(plan[0]["tool_name"], "compare_datasets")

    def test_file_analysis_without_filename_uses_default_source(self):
        user_input = "分析文件"
        skill = select_skill(user_input)
        decision = decide_retrieval(user_input, skill)

        self.assertTrue(decision["needs_retrieval"])
        self.assertEqual(decision["source"], "fyp_description.md")
        self.assertFalse(decision["source_was_explicit"])
        self.assertEqual(decision["reason_code"], "default_project_file")

    def test_planner_consumes_clean_retrieval_query(self):
        user_input = "read ReAct.pdf and explain its method"
        skill = select_skill(user_input)
        decision = decide_retrieval(user_input, skill)
        plan = create_plan(skill, user_input, decision)

        self.assertEqual(plan[0]["tool_args"]["file_path"], "ReAct.pdf")
        self.assertEqual(plan[0]["tool_args"]["query"], "explain its method")
        self.assertNotEqual(plan[0]["tool_args"]["query"], user_input)

    def test_decision_context_does_not_duplicate_retrieval_query(self):
        user_input = "分析 ReAct.md 中 reasoning 和 action 如何结合"
        skill = select_skill(user_input)
        decision = decide_retrieval(user_input, skill)
        plan = create_plan(skill, user_input, decision)
        context = build_context(
            user_input=user_input,
            memory=get_FYP_memory(),
            skill=skill,
            retrieval_decision=decision,
            execution_plan=plan,
            tool_observations=[],
        )
        bundle = build_prompt_bundle(context)

        self.assertEqual(bundle["user_prompt"].count(user_input), 1)
        self.assertNotIn("query", context["retrieval_decision_context"])
        self.assertIn("## Retrieval Decision", bundle["user_prompt"])


if __name__ == "__main__":
    unittest.main()
