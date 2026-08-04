#main.py
#启动Agent，等待用户输入
#把输入交给agent.run(),打印agent回复并等待下一轮输入


import json

from agent import FYPResearchAgent
from config import SHOW_DEBUG


def print_debug(debug_info: dict):
    print("\n[Debug]")
    print(f"Selected Skill: {debug_info['selected_skill']}")
    print(f"Skill Description: {debug_info['skill_description']}")
    print(f"Allowed Tools: {debug_info['allowed_tools']}")
    print(f"Output Style: {debug_info['output_style']}")
    print(f"Model: {debug_info['model']}")
    model_call = debug_info.get("model_call", {})
    if model_call:
        print(
            "Model Call: "
            f"status={model_call.get('status')} "
            f"calls={model_call.get('call_count', 1)} "
            f"timeout={model_call.get('timeout_seconds')}s "
            f"max_retries={model_call.get('max_retries_configured')} "
            "max_output_tokens="
            f"{model_call.get('max_output_tokens_configured')}"
        )
        if model_call.get("error"):
            print(f"Model Error: {model_call['error']}")
    structured = debug_info.get("structured_output", {})
    if structured.get("status") != "not_requested":
        print(
            "Structured Output: "
            f"status={structured.get('status')} "
            f"attempts={structured.get('attempt_count')}"
        )
        repair = debug_info.get("structured_output_repair", {})
        print(
            "Structured Repair: "
            f"attempted={repair.get('attempted')} "
            f"final={repair.get('final_status')}"
        )
    print(f"Context Keys: {debug_info['context_keys']}")
    print(f"Memory Keys: {debug_info['memory_keys']}")
    print(f"Relevant Memory Keys: {debug_info['relevant_memory_keys']}")
    print(f"Conversation Turns: {debug_info['conversation_turns']}")
    print(
        "Persistent Memories: "
        f"{debug_info.get('persistent_memory_counts', {})}"
    )
    memory_retrieval = debug_info.get("memory_retrieval", {})
    if memory_retrieval:
        print("\n[Memory Retrieval]")
        print(
            "Context Consent: "
            f"{'on' if debug_info.get('memory_context_enabled') else 'off'}"
        )
        print(f"Method: {memory_retrieval.get('retrieval_method')}")
        print(
            f"Candidates: {memory_retrieval.get('candidate_count')} "
            f"| Eligible: {memory_retrieval.get('eligible_count')} "
            f"| Selected: {memory_retrieval.get('selected_count')}"
        )
        for selected in memory_retrieval.get("selected", []):
            signals = selected["signals"]
            print(
                f"{selected['memory_id']} | {selected['memory_type']} "
                f"| score={selected['retrieval_score']:.4f} "
                f"| relevance={signals['relevance']:.4f} "
                f"| prompt={selected['included_in_prompt']}"
            )
    memory_write = debug_info.get("memory_write_decision", {})
    if memory_write:
        print("\n[Memory Write Policy]")
        print(f"Status: {memory_write.get('status')}")
        print(f"Reason Code: {memory_write.get('reason_code')}")
        print(f"Suggestion: {memory_write.get('suggestion_id')}")
        print(f"Pending Suggestions: {memory_write.get('pending_count')}")
    print(f"Tools Used: {debug_info['tools_used']}")
    for execution in debug_info.get("tool_executions", []):
        print(
            "Tool Contract: "
            f"step={execution.get('step_id')} "
            f"tool={execution.get('tool_name')} "
            f"status={execution.get('status')} "
            f"latency={execution.get('latency_seconds', 0.0):.4f}s"
        )
        if execution.get("error"):
            print(f"  Structured Error: {execution['error']}")
        permission = execution.get("permission")
        if permission:
            print(
                f"  Permission: allowed={permission.get('allowed')} "
                f"risk={permission.get('risk_level')} "
                f"reason={permission.get('reason_code')}"
            )

    agent_tool_loop = debug_info.get("agent_tool_loop")
    if agent_tool_loop:
        print("\n[Observation-driven Tool Loop]")
        print(f"Status: {agent_tool_loop.get('status')}")
        print(
            "Dynamic Steps: "
            f"{agent_tool_loop.get('dynamic_steps_used')} / "
            f"{agent_tool_loop.get('max_dynamic_steps')}"
        )
        print(f"Stopped Reason: {agent_tool_loop.get('stopped_reason')}")
        selection = agent_tool_loop.get("paper_selection", {})
        if selection.get("status") != "not_requested":
            print(
                f"Paper Selection: {selection.get('status')} "
                f"| index={selection.get('selected_index')} "
                f"| paper_id={selection.get('paper_id')}"
            )
        for iteration in agent_tool_loop.get("iterations", []):
            print(
                f"Iteration {iteration['iteration']}: "
                f"observe={iteration.get('observed_tool')} "
                f"-> {iteration.get('status')} "
                f"({iteration.get('reason_code')}) "
                f"-> tool={iteration.get('proposed_tool')}"
            )
            if iteration.get("user_input_request"):
                print(f"  Needs User: {iteration['user_input_request']}")

    retrieval_decision = debug_info.get("retrieval_decision")
    if retrieval_decision:
        print("\n[Retrieval Decision]")
        print(f"Needs Retrieval: {retrieval_decision['needs_retrieval']}")
        print(f"Reason Code: {retrieval_decision['reason_code']}")
        print(f"Reason: {retrieval_decision['reason']}")
        print(f"Strategy: {retrieval_decision['strategy']}")
        print(f"Source: {retrieval_decision['source']}")
        print(f"Retrieval Query: {retrieval_decision['query']}")

    context_evaluation = debug_info.get("context_evaluation")
    if context_evaluation:
        print("\n[Context Evaluation]")
        print(f"Status: {context_evaluation['status']}")
        print(f"Sufficient: {context_evaluation['is_sufficient']}")
        print(f"Reason Code: {context_evaluation['reason_code']}")
        print(f"Reason: {context_evaluation['reason']}")
        print(f"Recommended Action: {context_evaluation['recommended_action']}")
        signals = context_evaluation.get("signals", {})
        if signals:
            print(f"Query Coverage: {signals.get('query_term_coverage')}")
            print(f"Max Embedding Score: {signals.get('maximum_embedding_score')}")
            print(
                "Semantic Score Query-aligned: "
                f"{signals.get('semantic_score_query_aligned')}"
            )
            print(f"Dual-source Chunks: {signals.get('dual_source_chunks')}")
            print(f"Missing Terms: {signals.get('missing_query_terms')}")

    retrieval_loop = debug_info.get("retrieval_loop")
    if retrieval_loop and retrieval_loop.get("enabled"):
        print("\n[Agentic Retrieval Loop]")
        print(f"Max Retries: {retrieval_loop['max_retries']}")
        print(f"Retries Used: {retrieval_loop['retries_used']}")
        print(f"Final Evidence: {retrieval_loop['final_evidence']}")
        print(f"Stopped Reason: {retrieval_loop['stopped_reason']}")
        for attempt in retrieval_loop["attempts"]:
            evaluation = attempt["evaluation"]
            print(
                f"Attempt {attempt['attempt']}: {attempt['query']} "
                f"| Status: {evaluation['status']} "
                f"| Coverage: {evaluation['query_term_coverage']}"
            )
            attempt_cache = attempt.get("index_cache")
            if attempt_cache:
                print(
                    f"  Index Cache: hit={attempt_cache.get('hit')} "
                    f"size={attempt_cache.get('cache_size')} "
                    f"build={attempt_cache.get('build_seconds', 0):.3f}s"
                )
            refinement = attempt.get("query_refinement")
            if refinement:
                print(
                    f"  Refinement: missing={refinement['missing_terms']} "
                    f"intent={refinement['intent']} "
                    f"hints={refinement['added_hints']}"
                )

    context_stats = debug_info.get("context_stats")
    if context_stats:
        print("\n[Context]")
        print(
            f"Total: {context_stats['total']['characters']} chars "
            f"(~{context_stats['total']['estimated_tokens']} estimated tokens)"
        )
        for section_name, stats in context_stats["sections"].items():
            print(
                f"{section_name}: {stats['characters']} chars "
                f"(~{stats['estimated_tokens']} tokens)"
            )

    for retrieval in debug_info.get("retrievals", []):
        print("\n[Retrieval]")
        print(f"Method: {retrieval['retrieval_method']}")
        if retrieval["fusion_method"]:
            print(f"Fusion: {retrieval['fusion_method']}")
        print(f"Query: {retrieval['query']}")
        print(f"Expanded Query: {retrieval['expanded_query']}")
        print(f"Embedding Model: {retrieval['embedding_model']}")
        print(f"Chunks Returned: {retrieval['chunks_returned']}")
        index_cache = retrieval.get("index_cache")
        if index_cache:
            if "hit_sequence" in index_cache:
                print(f"Index Cache Hit Sequence: {index_cache['hit_sequence']}")
            else:
                print(f"Index Cache Hit: {index_cache.get('hit')}")
                print(f"Index Cache Size: {index_cache.get('cache_size')}")
                print(
                    "Index Build Time: "
                    f"{index_cache.get('build_seconds', 0):.3f}s"
                )
        if retrieval["candidate_k"] is not None:
            print(f"Candidate K: {retrieval['candidate_k']}")
            print(f"BM25 Candidates: {retrieval['bm25_candidates']}")
            print(f"Embedding Candidates: {retrieval['embedding_candidates']}")
            print(f"RRF Rank Constant: {retrieval['rank_constant']}")
        if retrieval["fallback_reason"]:
            print(f"Fallback Reason: {retrieval['fallback_reason']}")

    for search in debug_info.get("paper_searches", []):
        print("\n[Paper Search]")
        print(f"Provider: {search['provider']}")
        print(f"Query: {search['query']}")
        print(f"Year Filter: {search['year_filter']}")
        print(f"Results: {search['result_count']}")
        print(f"Total Estimate: {search['total_estimate']}")
        if search["error"]:
            print(f"Error: {search['error']}")

    for paper_import in debug_info.get("paper_imports", []):
        print("\n[Paper Detail / Import]")
        print(f"Tool: {paper_import['tool']}")
        print(f"Provider: {paper_import['provider']}")
        print(f"Paper ID: {paper_import['paper_id']}")
        print(f"Imported: {paper_import['imported']}")
        print(f"Already Existed: {paper_import['already_existed']}")
        print(f"File: {paper_import['file_path']}")
        print(f"Bytes: {paper_import['bytes_written']}")
        if paper_import["error"]:
            print(f"Error: {paper_import['error']}")

    for mcp_call in debug_info.get("mcp_calls", []):
        print("\n[MCP]")
        print(
            f"Server: {mcp_call.get('server')} "
            f"v{mcp_call.get('server_version')}"
        )
        print(f"Protocol: {mcp_call.get('protocol_version')}")
        print(f"Transport: {mcp_call.get('transport')}")
        print(f"Discovered Tools: {mcp_call.get('discovered_tools')}")
        print(f"Called Tool: {mcp_call.get('tool')}")
        print(f"Results: {mcp_call.get('result_count')}")
        if mcp_call.get("error"):
            print(f"Error: {mcp_call['error']}")

    print("\n[Plan]")
    for step in debug_info["plan_steps"]:
        print(
            f"Step {step['step_id']}: {step['description']} "
            f"| Tool: {step['tool_name']} "
            f"| Args: {step['tool_args']}"
        )

def print_usage(usage_info: dict):
    print("\n[Usage]")

    latency = usage_info["latency"]
    print(f"Latency: {latency:.2f}s")

    prompt_tokens = usage_info["prompt_tokens"]
    completion_tokens = usage_info["completion_tokens"]
    total_tokens = usage_info["total_tokens"]

    if total_tokens is None:
        print("Token usage: not available from provider.")
        return

    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Completion Tokens: {completion_tokens}")
    print(f"Total Tokens: {total_tokens}")


def print_memories(memories: list[dict]):
    if not memories:
        print("No persistent memories stored.")
        return

    print("\n[Persistent Memories]")
    for memory in memories:
        print(
            f"{memory['memory_id']} | {memory['memory_type']} "
            f"| importance={memory['importance']:.2f}"
        )
        print(f"  {memory['content']}")
        if memory["tags"]:
            print(f"  tags={memory['tags']}")


def print_memory_search(result: dict):
    print(
        "\n[Local Memory Search] "
        f"candidates={result['candidate_count']} "
        f"eligible={result['eligible_count']} "
        f"selected={result['selected_count']}"
    )
    if not result["memories"]:
        print("No relevant persistent memory found.")
        return

    for memory in result["memories"]:
        signals = memory["signals"]
        print(
            f"{memory['memory_id']} | {memory['memory_type']} "
            f"| score={memory['retrieval_score']:.4f} "
            f"| relevance={signals['relevance']:.4f}"
        )
        print(f"  {memory['content']}")
        print(f"  matched={signals['matched_units']}")


def print_memory_suggestions(suggestions: list[dict]):
    if not suggestions:
        print("No pending memory suggestions.")
        return

    print("\n[Pending Memory Suggestions]")
    for suggestion in suggestions:
        print(
            f"{suggestion['suggestion_id']} | {suggestion['category']} "
            f"| importance={suggestion['importance']:.2f}"
        )
        print(f"  {suggestion['content']}")
        print(f"  reason={suggestion['reason']}")


def print_tool_approvals(requests: list[dict]):
    if not requests:
        print("No pending tool approval requests.")
        return

    print("\n[Pending Tool Approvals]")
    for request in requests:
        print(
            f"{request['request_id']} | {request['tool_name']} "
            f"| risk={request['risk_level']}"
        )
        print(f"  arguments={json.dumps(request['arguments'], ensure_ascii=False)}")
        print(f"  digest={request['argument_digest']}")
        print(f"  reason={request['reason']}")


def main():
    agent = FYPResearchAgent()

    print("FYP Research Mini Agent started.")
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'clear' to clear short-term conversation memory.")
    print(
        "Type 'paper-json <paper analysis request>' to request validated "
        "PaperAnalysisOutput."
    )
    print("Type 'remember <text>' to save explicit long-term memory.")
    print(
        "Type 'remember-pref <text>' to save a globally applicable "
        "response preference."
    )
    print("Type 'memories' to list persistent memories.")
    print("Type 'memory-search <query>' to preview local memory retrieval.")
    print("Type 'memory-suggestions' to list pending write suggestions.")
    print("Type 'memory-save <suggestion_id>' to approve a suggestion.")
    print("Type 'memory-dismiss <suggestion_id>' to reject a suggestion.")
    print("Type 'approvals' to list pending high-risk tool requests.")
    print("Type 'approve <request_id>' to execute the exact approved tool call.")
    print("Type 'reject <request_id>' to reject a pending tool call.")
    print("Type 'forget <memory_id>' to delete one persistent memory.\n")
    print(
        "Type 'memory-context on/off/status' to control whether selected "
        "persistent memories may be sent to the external LLM.\n"
    )

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Agent stopped.")
            break

        if user_input.lower() == "clear":
            agent.clear_conversation()
            print("Short-term conversation memory cleared.")
            continue

        if user_input.lower() == "memories":
            print_memories(agent.list_persistent_memories())
            continue

        if user_input.lower() == "memory-suggestions":
            print_memory_suggestions(agent.list_memory_suggestions())
            continue

        if user_input.lower() == "approvals":
            print_tool_approvals(agent.list_tool_approval_requests())
            continue

        if user_input.lower().startswith("approve "):
            request_id = user_input[len("approve "):].strip()
            try:
                approved = agent.approve_tool_request(request_id)
            except (KeyError, ValueError) as error:
                print(f"Tool request was not approved: {error}")
            else:
                print("\n[Approved Tool Execution]")
                execution = approved["execution"]
                print(json.dumps(execution, ensure_ascii=False, indent=2))
                resumed = approved.get("agent_result")
                if resumed:
                    print("\n[Resumed Agent]")
                    print(resumed["response"])
                    if SHOW_DEBUG:
                        print_debug(resumed["debug"])
                        print_usage(resumed["usage"])
            continue

        if user_input.lower().startswith("reject "):
            request_id = user_input[len("reject "):].strip()
            try:
                rejected = agent.reject_tool_request(request_id)
            except (KeyError, ValueError) as error:
                print(f"Tool request was not rejected: {error}")
            else:
                print(f"Tool request rejected: {rejected['request_id']}")
            continue

        if user_input.lower().startswith("memory-save "):
            suggestion_id = user_input[len("memory-save "):].strip()
            try:
                memory = agent.approve_memory_suggestion(suggestion_id)
            except (KeyError, ValueError) as error:
                print(f"Memory suggestion was not saved: {error}")
            else:
                print(
                    f"Memory suggestion approved and saved: "
                    f"{memory['memory_id']}"
                )
            continue

        if user_input.lower().startswith("memory-dismiss "):
            suggestion_id = user_input[len("memory-dismiss "):].strip()
            if agent.dismiss_memory_suggestion(suggestion_id):
                print(f"Memory suggestion dismissed: {suggestion_id}")
            else:
                print(f"Memory suggestion not found: {suggestion_id}")
            continue

        if user_input.lower() == "memory-context status":
            status = "on" if agent.memory_context_enabled else "off"
            print(f"Persistent memory context is {status}.")
            continue

        if user_input.lower() == "memory-context on":
            agent.set_memory_context_enabled(True)
            print(
                "Persistent memory context enabled for this run. Relevant "
                "stored memories may be sent to the configured external LLM."
            )
            continue

        if user_input.lower() == "memory-context off":
            agent.set_memory_context_enabled(False)
            print("Persistent memory context disabled.")
            continue

        if user_input.lower().startswith("memory-search "):
            query = user_input[len("memory-search "):].strip()
            if not query:
                print("Nothing to search.")
                continue
            print_memory_search(agent.retrieve_persistent_memories(query))
            continue

        if user_input.lower().startswith("remember-pref "):
            content = user_input[len("remember-pref "):].strip()
            if not content:
                print("Nothing to remember.")
                continue
            memory = agent.remember_long_term(
                content,
                tags=["preference"],
                importance=0.8,
            )
            print(f"Global response preference saved: {memory['memory_id']}")
            continue

        if user_input.lower().startswith("remember "):
            content = user_input[len("remember "):].strip()
            if not content:
                print("Nothing to remember.")
                continue
            memory = agent.remember_long_term(content)
            print(f"Long-term memory saved: {memory['memory_id']}")
            continue

        if user_input.lower().startswith("forget "):
            memory_id = user_input[len("forget "):].strip()
            if agent.forget_memory(memory_id):
                print(f"Memory deleted: {memory_id}")
            else:
                print(f"Memory not found: {memory_id}")
            continue

        if not user_input:
            continue

        output_mode = "text"
        if user_input.lower().startswith("paper-json "):
            user_input = user_input[len("paper-json "):].strip()
            if not user_input:
                print("Nothing to analyze.")
                continue
            output_mode = "paper_analysis"

        try:
            result = agent.run(user_input, output_mode=output_mode)
        except ValueError as error:
            print(f"Cannot run requested output mode: {error}")
            continue

        if SHOW_DEBUG:
            print_debug(result["debug"])
            print_usage(result["usage"])

        print("\nAgent:")
        print(result["response"])
        structured = result.get("structured_output", {})
        if structured.get("status") == "valid":
            print("\n[Validated Structured Output]")
            print(json.dumps(
                structured["data"],
                ensure_ascii=False,
                indent=2,
            ))
        elif structured.get("status") == "fallback":
            print("\n[Structured Output Fallback]")
            print(structured.get("error"))
        suggestion = result.get("memory_suggestion")
        if suggestion:
            print("\n[Memory Suggestion - not saved yet]")
            print(
                f"{suggestion['suggestion_id']} | "
                f"{suggestion['category']}"
            )
            print(f"  {suggestion['content']}")
            print(
                "Use 'memory-save "
                f"{suggestion['suggestion_id']}' to approve, or "
                "'memory-dismiss "
                f"{suggestion['suggestion_id']}' to reject."
            )
        for approval in result.get("approval_requests", []):
            print("\n[Tool Approval Required - not executed]")
            print(
                f"{approval['request_id']} | {approval['tool_name']} "
                f"| risk={approval['risk_level']}"
            )
            print(
                "Use 'approve "
                f"{approval['request_id']}' to execute these exact arguments, "
                "or 'reject "
                f"{approval['request_id']}' to reject."
            )
        print("-" * 60)


if __name__ == "__main__":
    main()
