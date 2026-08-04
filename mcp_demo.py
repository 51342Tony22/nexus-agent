"""Inspect the local MCP server without calling an LLM or opening a port.

Run without arguments for the lightweight in-memory transport, or pass
``--stdio`` to launch the MCP server as a separate subprocess.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextResourceContents

from mcp_library_server import server


PROJECT_DIR = Path(__file__).resolve().parent


async def inspect_server(*, use_stdio: bool = False) -> None:
    if use_stdio:
        transport = stdio_client(StdioServerParameters(
            command=sys.executable,
            args=[str(PROJECT_DIR / "mcp_library_server.py")],
            cwd=str(PROJECT_DIR),
        ))
    else:
        transport = server

    async with Client(transport) as client:
        print(f"Transport: {'stdio subprocess' if use_stdio else 'in memory'}")
        print(f"Server: {client.server_info.name} v{client.server_info.version}")
        print(f"Protocol: {client.protocol_version}")

        tools = await client.list_tools()
        print("\nTools discovered by the client:")
        for tool in tools.tools:
            print(f"- {tool.name}: {tool.description}")
            print(json.dumps(tool.input_schema, ensure_ascii=False, indent=2))

        resources = await client.list_resources()
        print("\nResources discovered by the client:")
        for resource in resources.resources:
            print(f"- {resource.uri}")

        prompts = await client.list_prompts()
        print("\nPrompts discovered by the client:")
        for prompt in prompts.prompts:
            print(f"- {prompt.name}")

        result = await client.call_tool(
            "search_fyp_library",
            {"query": "多 rationale 一致性", "limit": 6},
        )
        print("\nStructured tool result:")
        print(json.dumps(
            result.structured_content,
            ensure_ascii=False,
            indent=2,
        ))

        catalog = await client.read_resource("fyp://library/catalog")
        catalog_text = next(
            (
                item.text
                for item in catalog.contents
                if isinstance(item, TextResourceContents)
            ),
            "[]",
        )
        print(f"\nCatalog records: {len(json.loads(catalog_text))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="launch mcp_library_server.py as a separate stdio subprocess",
    )
    args = parser.parse_args()
    asyncio.run(inspect_server(use_stdio=args.stdio))
