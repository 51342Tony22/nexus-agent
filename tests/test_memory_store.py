import json
import tempfile
import unittest
from pathlib import Path

from agent import FYPResearchAgent
from memory_store import MemoryStore, MemoryStoreError


class MemoryStoreTests(unittest.TestCase):
    def test_add_memory_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            store = MemoryStore(path)
            record = store.add_memory(
                memory_type="long_term",
                content="用户研究 Reasoning Distillation",
                source="explicit_user_statement",
                confidence=1.0,
                importance=0.9,
                tags=["FYP", "research-topic", "fyp"],
            )

            reloaded = MemoryStore(path)
            loaded_record = reloaded.get_memory(record["memory_id"])

            self.assertEqual(
                loaded_record["content"],
                "用户研究 Reasoning Distillation",
            )
            self.assertEqual(loaded_record["tags"], ["fyp", "research-topic"])
            self.assertEqual(reloaded.count_by_type()["long_term"], 1)

    def test_update_and_delete_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            store = MemoryStore(path)
            record = store.add_memory(
                memory_type="long_term",
                content="用户喜欢简短回答",
                source="explicit_user_statement",
            )
            created_at = record["created_at"]

            updated = store.update_memory(
                record["memory_id"],
                content="用户喜欢带教学解释的回答",
                importance=0.8,
            )

            self.assertEqual(updated["created_at"], created_at)
            self.assertEqual(updated["content"], "用户喜欢带教学解释的回答")
            self.assertEqual(updated["importance"], 0.8)
            self.assertTrue(store.delete_memory(record["memory_id"]))
            self.assertFalse(store.delete_memory(record["memory_id"]))
            self.assertEqual(store.count_by_type()["total"], 0)

    def test_memory_validation_rejects_unsafe_records(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memories.json")

            with self.assertRaises(ValueError):
                store.add_memory(
                    memory_type="short_term",
                    content="should stay in conversation memory",
                    source="test",
                )

            with self.assertRaises(ValueError):
                store.add_memory(
                    memory_type="long_term",
                    content="invalid confidence",
                    source="test",
                    confidence=1.5,
                )

            with self.assertRaises(ValueError):
                store.add_memory(
                    memory_type="long_term",
                    content="   ",
                    source="test",
                )

    def test_corrupt_json_is_reported_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            path.write_text("{broken json", encoding="utf-8")

            with self.assertRaises(MemoryStoreError):
                MemoryStore(path)

            self.assertEqual(path.read_text(encoding="utf-8"), "{broken json")

    def test_schema_is_explicit_in_saved_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            store = MemoryStore(path)
            store.add_memory(
                memory_type="episodic",
                content="Task completed",
                source="task_result",
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["memories"]), 1)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_agent_only_writes_memory_after_explicit_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            agent = FYPResearchAgent(memory_store_path=str(path))

            self.assertFalse(path.exists())
            saved = agent.remember_long_term(
                "分析论文时解释为什么这样设计",
                tags=["preference", "learning"],
            )
            self.assertTrue(path.exists())

            restarted_agent = FYPResearchAgent(memory_store_path=str(path))
            restored = restarted_agent.list_persistent_memories()
            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0]["memory_id"], saved["memory_id"])

    def test_agent_can_record_manual_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            agent = FYPResearchAgent(memory_store_path=str(path))

            episode = agent.record_episode(
                task="Implement v1.3.4",
                outcome="Document index cache passed all tests",
                tags=["agentic-rag", "cache"],
            )

            self.assertEqual(episode["memory_type"], "episodic")
            self.assertEqual(episode["source"], "task_result")
            self.assertEqual(
                episode["metadata"]["task"],
                "Implement v1.3.4",
            )


if __name__ == "__main__":
    unittest.main()
