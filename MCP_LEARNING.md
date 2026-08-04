# v1.9 MCP Learning Notes

## What MCP changes

Before MCP, this project imports a Python function directly from its own tool
registry. That creates a custom connection between one Agent and one tool.

With MCP, the boundary is explicit:

```text
Agent
  -> MCP Client
      -> protocol discovery and validation
          -> MCP Server
              -> allowlisted FYP library capability
```

The server does not become an Agent. It does not choose goals, plan, decide
whether retrieval is needed, or generate a final answer. It only advertises
and executes capabilities using a standard protocol.

## The four roles in this project

- **Host:** `FYPResearchAgent`, which decides when a capability is useful.
- **Client:** `mcp_client_adapter.py`, which discovers and calls the server.
- **Server:** `mcp_library_server.py`, which owns the capability boundary.
- **Transport:** in-memory for the current Agent and tests; stdio is available
  when an external MCP host launches `mcp_library_server.py`.

## Tools, Resources and Prompts

- A **Tool** performs an operation. `search_fyp_library` searches allowlisted
  metadata and returns typed structured content.
- A **Resource** is read-only addressable data. `fyp://library/catalog` exposes
  the complete catalog.
- A **Prompt** is a reusable message template. `analyze_fyp_paper` tells an MCP
  host how to request grounded analysis; rendering it does not run retrieval.

## Why the allowlist matters

The MCP server reads filenames only from
`evaluations/paper_analysis_cases.json`. It does not accept arbitrary paths,
write files, run shell commands or expose `.env`. MCP standardizes transport;
it does not automatically make a server safe or its returned content trusted.

All MCP data enters the model context with
`trust_level=untrusted_mcp_server_data`.

## Try it without an LLM

```bash
.venv/bin/python mcp_demo.py
```

The command above uses the in-memory transport. To prove that the same server
also works as an independent MCP process over standard input/output, run:

```bash
.venv/bin/python mcp_demo.py --stdio
```

Expected observations:

```text
Server: fyp-library v1.0.0
Protocol: 2026-07-28
Tools discovered by the client:
- get_fyp_paper_record
- search_fyp_library
Catalog records: 6
```

## Try it through the Agent

Start `main.py`, then ask:

```text
用 MCP 在本地论文库搜索多 rationale 一致性
```

The expected Debug chain is:

```text
Selected Skill: mcp_library_skill
Tool: mcp_search_fyp_library
Server: fyp-library
Transport: in_memory
```

## Run the checks

The MCP-specific tests cover capability discovery, typed results, the six-file
allowlist, path rejection, Agent context isolation and a real stdio subprocess:

```bash
.venv/bin/python -m unittest tests.test_mcp_integration -v
```

Run the complete regression suite after changing any shared prompt or tool
contract:

```bash
.venv/bin/python -m unittest discover -s tests -q
```

This version deliberately leaves `retrieve_from_file` as the existing local
tool. Migrating every working tool to MCP would teach less and create more
risk. A later version can compare direct retrieval with MCP retrieval using
the same tests before choosing one architecture.
