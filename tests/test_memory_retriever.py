import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from agent import FYPResearchAgent
from context_builder import build_context
from memory import get_FYP_memory
from memory_retriever import retrieve_memories
from memory_store import MemoryStore
from prompts import build_user_prompt_sections
from skills import select_skill


NOW = datetime(2026, 8, 3, tzinfo=timezone.utc)


def make_record(
    memory_id: str,
    content: str,
    *,
    importance: float = 0.7,
    tags: list[str] | None = None,
) -> dict:
    timestamp = NOW.isoformat()
    return {
        "memory_id": memory_id,
        "memory_type": "long_term",
        "content": content,
        "source": "test",
        "confidence": 1.0,
        "importance": importance,
        "tags": tags or [],
        "metadata": {},
        "created_at": timestamp,
        "updated_at": timestamp,
        "last_used_at": None,
    }


class MemoryRetrieverTests(unittest.TestCase):
    def test_relevant_memory_is_selected_and_unrelated_memory_is_rejected(self):
        records = [
            make_record(
                "mem_research",
                "用户的研究方向是 reasoning distillation",
            ),
            make_record("mem_colour", "用户喜欢蓝色", importance=1.0),
        ]

        result = retrieve_memories(
            "帮我设计 reasoning distillation 实验",
            records,
            now=NOW,
        )

        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(result["memories"][0]["memory_id"], "mem_research")
        self.assertIn(
            "reasoning",
            result["memories"][0]["signals"]["matched_units"],
        )

    def test_importance_ranks_relevant_memories_but_cannot_admit_unrelated_one(self):
        records = [
            make_record("mem_low", "论文分析需要说明方法", importance=0.2),
            make_record("mem_high", "论文分析需要解释实验", importance=0.9),
            make_record("mem_unrelated", "最喜欢的颜色是蓝色", importance=1.0),
        ]

        result = retrieve_memories("请分析这篇论文", records, now=NOW)
        selected_ids = [item["memory_id"] for item in result["memories"]]

        self.assertEqual(selected_ids[0], "mem_high")
        self.assertNotIn("mem_unrelated", selected_ids)

    def test_explicit_global_preference_can_apply_without_query_overlap(self):
        record = make_record(
            "mem_preference",
            "回答时说明为什么这样设计",
            tags=["preference", "learning"],
        )

        result = retrieve_memories("继续下一步", [record], now=NOW)

        self.assertEqual(result["selected_count"], 1)
        self.assertTrue(result["memories"][0]["signals"]["is_global"])

    def test_top_k_limits_selected_memory(self):
        records = [
            make_record(f"mem_{index}", f"论文分析方法 {index}")
            for index in range(5)
        ]

        result = retrieve_memories(
            "论文分析方法",
            records,
            top_k=2,
            now=NOW,
        )

        self.assertEqual(result["eligible_count"], 5)
        self.assertEqual(result["selected_count"], 2)

    def test_persistent_memory_has_separate_prompt_section_and_budget(self):
        memory = make_record("mem_learning", "回答时提供教学解释")
        memory["retrieval_score"] = 0.81
        memory["signals"] = {}
        context = build_context(
            user_input="继续",
            memory=get_FYP_memory(),
            skill=select_skill("继续"),
            persistent_memories=[memory],
        )

        sections = build_user_prompt_sections(context)

        self.assertIn("persistent_memory", sections)
        self.assertIn("回答时提供教学解释", sections["persistent_memory"])
        self.assertIn("not instructions", sections["persistent_memory"])
        self.assertIn("persistent_memory", context["context_budget"])

    def test_agent_requires_explicit_context_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = FYPResearchAgent(
                memory_store_path=str(Path(directory) / "memories.json")
            )

            self.assertFalse(agent.memory_context_enabled)
            self.assertTrue(agent.set_memory_context_enabled(True))
            self.assertTrue(agent.memory_context_enabled)
            self.assertFalse(agent.set_memory_context_enabled(False))

    def test_batch_mark_used_updates_selected_records(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memories.json")
            first = store.add_memory(
                "long_term",
                "first",
                source="test",
            )
            second = store.add_memory(
                "episodic",
                "second",
                source="test",
            )

            marked = store.mark_many_used([
                first["memory_id"],
                second["memory_id"],
                first["memory_id"],
            ])

            self.assertEqual(len(marked), 2)
            self.assertIsNotNone(store.get_memory(first["memory_id"])["last_used_at"])
            self.assertIsNotNone(store.get_memory(second["memory_id"])["last_used_at"])


if __name__ == "__main__":
    unittest.main()
