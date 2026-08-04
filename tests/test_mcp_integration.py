import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextResourceContents

from agent import FYPResearchAgent
from mcp_client_adapter import (
    read_fyp_catalog_via_mcp,
    search_fyp_library_via_mcp,
)
from mcp_library_server import server
from planner import create_plan
from skills import select_skill
from tool_contracts import ToolInputValidationError
from tool_registry import build_tool_call, get_tool_spec, run_tool


PROJECT_DIR = Path(__file__).resolve().parents[1]


class MCPServerProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_discovers_tools_resource_and_prompt(self):
        async with Client(server) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()

            self.assertEqual(client.server_info.name, "fyp-library")
            self.assertEqual(client.server_info.version, "1.0.0")
            self.assertEqual(client.protocol_version, "2026-07-28")
            self.assertEqual(
                {tool.name for tool in tools.tools},
                {"search_fyp_library", "get_fyp_paper_record"},
            )
            self.assertIn(
                "fyp://library/catalog",
                {str(resource.uri) for resource in resources.resources},
            )
            self.assertIn(
                "analyze_fyp_paper",
                {prompt.name for prompt in prompts.prompts},
            )

    async def test_tool_returns_structured_content(self):
        async with Client(server) as client:
            result = await client.call_tool(
                "search_fyp_library",
                {"query": "多 rationale 一致性", "limit": 6},
            )

        self.assertFalse(result.is_error)
        self.assertIsInstance(result.structured_content, dict)
        filenames = {
            paper["filename"]
            for paper in result.structured_content["papers"]
        }
        self.assertIn("5_MCC-KD_Mutiltationales_SameAnswer.pdf", filenames)
        self.assertEqual(
            result.structured_content["trust_level"],
            "untrusted_mcp_server_data",
        )

    async def test_resource_contains_only_the_six_allowlisted_papers(self):
        async with Client(server) as client:
            result = await client.read_resource("fyp://library/catalog")

        text = next(
            item.text
            for item in result.contents
            if isinstance(item, TextResourceContents)
        )
        catalog = json.loads(text)
        self.assertEqual(len(catalog), 6)
        self.assertTrue(all(item["exists"] for item in catalog))
        self.assertNotIn(".env", text)

    async def test_server_tool_error_is_a_result_not_a_client_crash(self):
        async with Client(server) as client:
            result = await client.call_tool(
                "get_fyp_paper_record",
                {"filename": "../../.env"},
            )

        self.assertTrue(result.is_error)
        self.assertIsNone(result.structured_content)

    async def test_stdio_subprocess_negotiates_and_calls_tool(self):
        """Prove the standalone server works across a real process boundary."""
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(PROJECT_DIR / "mcp_library_server.py")],
            cwd=str(PROJECT_DIR),
        )

        async with asyncio.timeout(15):
            async with Client(stdio_client(parameters)) as client:
                result = await client.call_tool(
                    "search_fyp_library",
                    {"query": "knowledge distillation", "limit": 2},
                )
                server_name = client.server_info.name
                protocol_version = client.protocol_version

        self.assertEqual(server_name, "fyp-library")
        self.assertEqual(protocol_version, "2026-07-28")
        self.assertFalse(result.is_error)
        self.assertLessEqual(result.structured_content["result_count"], 2)


class MCPAgentAdapterTests(unittest.TestCase):
    def test_sync_adapter_preserves_protocol_metadata(self):
        result = search_fyp_library_via_mcp(
            query="rationale 一致性",
            limit=6,
        )

        self.assertIsNone(result["error"])
        self.assertEqual(result["mcp"]["server"], "fyp-library")
        self.assertEqual(result["mcp"]["transport"], "in_memory")
        self.assertIn("search_fyp_library", result["mcp"]["discovered_tools"])
        self.assertGreaterEqual(result["data"]["result_count"], 1)

    def test_resource_adapter_reads_catalog(self):
        result = read_fyp_catalog_via_mcp()

        self.assertIsNone(result["error"])
        self.assertEqual(result["mcp"]["resource"], "fyp://library/catalog")
        self.assertEqual(len(json.loads(result["contents"][0])), 6)

    def test_skill_planner_and_registry_use_the_mcp_adapter(self):
        user_input = "用 MCP 在本地论文库搜索多 rationale 一致性"
        skill = select_skill(user_input)
        plan = create_plan(skill, user_input)
        spec = get_tool_spec("mcp_search_fyp_library")

        self.assertEqual(skill["name"], "mcp_library_skill")
        self.assertEqual(plan[0]["tool_name"], "mcp_search_fyp_library")
        self.assertEqual(plan[0]["tool_args"]["limit"], 6)
        self.assertIn("rationale", plan[0]["tool_args"]["query"])
        self.assertEqual(spec.risk_level, "read_only")
        self.assertFalse(spec.requires_confirmation)

        result = run_tool(build_tool_call(
            "mcp_search_fyp_library",
            plan[0]["tool_args"],
        ))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_result"]["mcp"]["server"], "fyp-library")

    def test_registry_rejects_invalid_mcp_arguments_before_calling(self):
        with self.assertRaises(ToolInputValidationError):
            build_tool_call(
                "mcp_search_fyp_library",
                {"query": "", "limit": 99},
            )

    def test_full_agent_places_mcp_data_in_untrusted_external_context(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = FYPResearchAgent(
                memory_store_path=str(Path(directory) / "memories.json")
            )
            captured = {}

            def fake_llm(context):
                captured["context"] = context
                return {
                    "status": "success",
                    "content": "本地目录中找到相关论文。",
                    "model": "test-model",
                    "latency": 0.0,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "context_stats": {},
                }

            agent.call_llm = fake_llm
            result = agent.run(
                "用 MCP 在本地论文库搜索多 rationale 一致性"
            )

        self.assertEqual(result["debug"]["selected_skill"], "mcp_library_skill")
        self.assertEqual(result["debug"]["tools_used"], [
            "mcp_search_fyp_library"
        ])
        self.assertEqual(result["debug"]["mcp_calls"][0]["server"], "fyp-library")
        external = captured["context"]["external_tool_data"][0]
        self.assertEqual(external["provider"], "MCP")
        self.assertEqual(
            external["trust_level"],
            "untrusted_mcp_server_data",
        )


if __name__ == "__main__":
    unittest.main()
