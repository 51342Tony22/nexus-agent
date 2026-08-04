"""Synchronous Agent adapter around the asynchronous MCP v2 Client."""

import asyncio

from mcp import Client
from mcp.types import TextContent, TextResourceContents

from mcp_library_server import server


async def _call_tool(tool_name: str, arguments: dict) -> dict:
    async with Client(server) as client:
        discovered = await client.list_tools()
        available = {tool.name for tool in discovered.tools}
        if tool_name not in available:
            return {
                "error": {
                    "code": "mcp_tool_not_discovered",
                    "message": f"MCP server did not advertise '{tool_name}'.",
                    "retryable": False,
                },
                "mcp": {
                    "server": client.server_info.name,
                    "protocol_version": client.protocol_version,
                    "transport": "in_memory",
                    "tool": tool_name,
                },
            }

        result = await client.call_tool(tool_name, arguments)
        metadata = {
            "server": client.server_info.name,
            "server_version": client.server_info.version,
            "protocol_version": client.protocol_version,
            "transport": "in_memory",
            "tool": tool_name,
            "discovered_tools": sorted(available),
        }
        if result.is_error:
            messages = [
                block.text
                for block in result.content
                if isinstance(block, TextContent)
            ]
            return {
                "error": {
                    "code": "mcp_tool_error",
                    "message": "MCP tool execution failed.",
                    "retryable": False,
                    "details": {"messages": messages[:3]},
                },
                "mcp": metadata,
            }

        return {
            "mcp": metadata,
            "data": result.structured_content,
            "error": None,
        }


async def _read_catalog_resource() -> dict:
    async with Client(server) as client:
        resources = await client.list_resources()
        available = {str(resource.uri) for resource in resources.resources}
        uri = "fyp://library/catalog"
        if uri not in available:
            return {
                "error": {
                    "code": "mcp_resource_not_discovered",
                    "message": f"MCP server did not advertise '{uri}'.",
                    "retryable": False,
                }
            }

        result = await client.read_resource(uri)
        texts = [
            item.text
            for item in result.contents
            if isinstance(item, TextResourceContents)
        ]
        return {
            "mcp": {
                "server": client.server_info.name,
                "protocol_version": client.protocol_version,
                "transport": "in_memory",
                "resource": uri,
            },
            "contents": texts,
            "error": None,
        }


def _run(coroutine) -> dict:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.run(coroutine)
        except Exception as error:
            return {
                "error": {
                    "code": "mcp_client_error",
                    "message": f"MCP client failed with {type(error).__name__}.",
                    "retryable": False,
                }
            }

    coroutine.close()
    return {
        "error": {
            "code": "mcp_event_loop_conflict",
            "message": (
                "The synchronous Agent adapter cannot run inside an active "
                "asyncio event loop."
            ),
            "retryable": False,
        }
    }


def search_fyp_library_via_mcp(query: str, limit: int = 5) -> dict:
    return _run(_call_tool(
        "search_fyp_library",
        {"query": query, "limit": limit},
    ))


def get_fyp_paper_record_via_mcp(filename: str) -> dict:
    return _run(_call_tool(
        "get_fyp_paper_record",
        {"filename": filename},
    ))


def read_fyp_catalog_via_mcp() -> dict:
    return _run(_read_catalog_resource())
