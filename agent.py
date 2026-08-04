#agent.oy
#串联memory->skill->构建context
#最终返回main


import re

from memory import get_FYP_memory
from skills import select_skill
from context_builder import build_context
from llm_client import LLMClient
from config import (
    AGENT_TOOL_LOOP_MAX_STEPS,
    AGENTIC_RAG_MAX_MERGED_CHUNKS,
    AGENTIC_RAG_MAX_RETRIES,
    MEMORY_STORE_PATH,
    MEMORY_RETRIEVAL_MIN_RELEVANCE,
    MEMORY_RETRIEVAL_TOP_K,
    SILICONFLOW_MODEL,
    STRUCTURED_OUTPUT_MAX_REPAIR_ATTEMPTS,
)
from conversation_memory import ConversationMemory
from memory_store import MemoryStore
from memory_retriever import retrieve_memories
from memory_write_policy import (
    MAX_PENDING_SUGGESTIONS,
    evaluate_memory_write,
)
from prompts import build_prompt_bundle
from output_contracts import (
    StructuredOutputError,
    StructuredOutputResult,
    build_output_schema_context,
    build_structured_output_repair,
    format_paper_analysis,
    validate_paper_analysis_output,
)
from planner import create_plan
from retrieval_decider import decide_retrieval
from context_evaluator import evaluate_retrieval_context
from retrieval_loop import (
    build_refined_query,
    build_retry_step,
    merge_retrieval_observations,
    should_retry,
)
from tool_registry import build_tool_call, run_tool
from permission_manager import check_tool_permission
from approval_manager import ApprovalManager
from tool_contracts import (
    ToolError,
    ToolInputValidationError,
    ToolObservation,
    error_from_tool_result,
)
from tool_loop import decide_next_tool_action


def aggregate_model_usage(model_results: list[dict]) -> dict:
    """Sum one turn's model calls without pretending unknown tokens are zero."""
    def sum_if_known(key: str) -> int | None:
        values = [
            result.get("usage", {}).get(key)
            for result in model_results
        ]
        if not values or any(not isinstance(value, int) for value in values):
            return None
        return sum(values)

    return {
        "latency": sum(
            float(result.get("latency", 0.0))
            for result in model_results
        ),
        "prompt_tokens": sum_if_known("prompt_tokens"),
        "completion_tokens": sum_if_known("completion_tokens"),
        "total_tokens": sum_if_known("total_tokens"),
    }

class FYPResearchAgent:
    def __init__(self, memory_store_path: str | None = None):
        self.memory = get_FYP_memory()
        self.llm = LLMClient()
        self.conversation_memory = ConversationMemory(max_turns=5)
        self.memory_store = MemoryStore(
            memory_store_path or MEMORY_STORE_PATH
        )
        # Persistent memory may contain sensitive data. It stays local until
        # the user explicitly opts in for the current Agent process.
        self.memory_context_enabled = False
        # Suggestions are deliberately process-local until the user approves.
        self.pending_memory_suggestions = {}
        # Search selections are short-term process state, not persistent user
        # memory. They only support follow-ups such as "下载第1篇".
        self.last_paper_search_results = []
        # Tool approvals are process-local, single-use, and bound to validated
        # arguments. No approval state is sent to the external LLM.
        self.approval_manager = ApprovalManager()


    def run(
        self,
        user_input: str,
        output_mode: str = "text",
        *,
        _resume_execution_plan: list[dict] | None = None,
        _resume_tool_observations: list[dict] | None = None,
    ) -> dict:
        planning_input, paper_selection = self.resolve_paper_selection(
            user_input
        )
        skill = select_skill(planning_input)
        structured_output_schema = build_output_schema_context(
            output_mode=output_mode,
            skill=skill,
        )
        retrieval_decision = decide_retrieval(
            user_input=planning_input,
            skill=skill,
        )

        if (
            (_resume_execution_plan is None)
            != (_resume_tool_observations is None)
        ):
            raise ValueError(
                "A resumed run requires both its plan and observations."
            )
        if _resume_execution_plan is not None:
            execution_plan = _resume_execution_plan
            tool_observations = _resume_tool_observations
        else:
            execution_plan = create_plan(
                skill=skill,
                user_input=planning_input,
                retrieval_decision=retrieval_decision,
            )
            tool_observations = self.execute_plan(
                execution_plan,
                user_input=user_input,
                output_mode=output_mode,
            )
        (
            execution_plan,
            tool_observations,
            retrieval_decision,
            agent_tool_loop,
        ) = self.run_observation_tool_loop(
            user_input=user_input,
            skill=skill,
            retrieval_decision=retrieval_decision,
            execution_plan=execution_plan,
            tool_observations=tool_observations,
            paper_selection=paper_selection,
        )
        (
            execution_plan,
            tool_observations,
            context_evaluation,
            retrieval_loop,
        ) = self.run_retrieval_loop(
            retrieval_decision=retrieval_decision,
            execution_plan=execution_plan,
            tool_observations=tool_observations,
        )

        conversation_summary = self.conversation_memory.get_summary()
        conversation_history = self.conversation_memory.format_recent_history(
            max_turns=2
        )
        memory_retrieval = self.retrieve_persistent_memories(user_input)
        prompt_memories = (
            memory_retrieval["memories"]
            if self.memory_context_enabled
            else []
        )
        if prompt_memories:
            self.memory_store.mark_many_used([
                memory["memory_id"]
                for memory in prompt_memories
            ])

        context = build_context(
            user_input=user_input,
            memory=self.memory,
            skill=skill,
            retrieval_decision=retrieval_decision,
            context_evaluation=context_evaluation,
            conversation_summary=conversation_summary,
            conversation_history=conversation_history,
            execution_plan=execution_plan,
            tool_observations=tool_observations,
            persistent_memories=prompt_memories,
            agent_tool_loop=agent_tool_loop,
            structured_output_schema=structured_output_schema,
        )

        llm_result = self.call_llm(context)
        model_results = [llm_result]

        response_content = llm_result["content"]
        structured_output = StructuredOutputResult(
            status="not_requested",
        )
        structured_output_repair = {
            "attempted": False,
            "max_attempts": STRUCTURED_OUTPUT_MAX_REPAIR_ATTEMPTS,
            "initial_error": None,
            "repair_model_status": None,
            "final_status": structured_output.status,
            "context_stats": None,
        }
        if structured_output_schema:
            if llm_result.get("status", "success") == "success":
                structured_output = validate_paper_analysis_output(
                    raw_content=llm_result["content"],
                    evidence=context["evidence"],
                ).model_copy(update={"attempt_count": 1})
            else:
                structured_output = StructuredOutputResult(
                    status="fallback",
                    schema_name="PaperAnalysisOutput",
                    error=StructuredOutputError(
                        code="model_call_failed",
                        message=(
                            "Structured output could not be produced because "
                            "the model call failed."
                        ),
                        details={
                            "model_error": llm_result.get("error"),
                        },
                    ),
                    attempt_count=1,
                )

            if (
                structured_output.status == "fallback"
                and llm_result.get("status", "success") == "success"
                and STRUCTURED_OUTPUT_MAX_REPAIR_ATTEMPTS > 0
            ):
                initial_error = structured_output.error.model_dump()
                repair_context = {
                    **context,
                    "structured_output_repair": build_structured_output_repair(
                        raw_content=llm_result["content"],
                        validation_result=structured_output,
                    ),
                }
                repair_result = self.call_llm(repair_context)
                model_results.append(repair_result)
                structured_output_repair = {
                    "attempted": True,
                    "max_attempts": STRUCTURED_OUTPUT_MAX_REPAIR_ATTEMPTS,
                    "initial_error": initial_error,
                    "repair_model_status": repair_result.get(
                        "status",
                        "success",
                    ),
                    "final_status": "fallback",
                    "context_stats": repair_result.get("context_stats"),
                }
                if repair_result.get("status", "success") == "success":
                    structured_output = validate_paper_analysis_output(
                        raw_content=repair_result["content"],
                        evidence=context["evidence"],
                    ).model_copy(update={"attempt_count": 2})
                else:
                    structured_output = StructuredOutputResult(
                        status="fallback",
                        schema_name="PaperAnalysisOutput",
                        error=StructuredOutputError(
                            code="repair_model_call_failed",
                            message=(
                                "The one allowed structured-output repair "
                                "model call failed."
                            ),
                            details={
                                "initial_validation_error": initial_error,
                                "model_error": repair_result.get("error"),
                            },
                        ),
                        attempt_count=2,
                    )

            if structured_output.status == "valid":
                response_content = format_paper_analysis(
                    structured_output.data
                )
            elif llm_result.get("status", "success") == "success":
                response_content = (
                    "结构化输出验证失败，已安全回退为模型原始文本。\n\n"
                    f"{llm_result['content']}"
                )
            structured_output_repair["final_status"] = (
                structured_output.status
            )

        usage = aggregate_model_usage(model_results)

        if llm_result.get("status", "success") == "success":
            self.conversation_memory.add_turn(
                user_input=user_input,
                assitant_response=response_content,
            )
        memory_write_decision = self.propose_memory_write(user_input)

        retrieval_debug = []
        paper_search_debug = []
        paper_import_debug = []
        mcp_debug = []
        for observation in tool_observations:
            tool_result = observation.get("tool_result")
            if not isinstance(tool_result, dict):
                continue

            if observation.get("tool_used") == "mcp_search_fyp_library":
                mcp_metadata = tool_result.get("mcp") or {}
                mcp_data = tool_result.get("data") or {}
                mcp_debug.append({
                    "server": mcp_metadata.get("server"),
                    "server_version": mcp_metadata.get("server_version"),
                    "protocol_version": mcp_metadata.get("protocol_version"),
                    "transport": mcp_metadata.get("transport"),
                    "tool": mcp_metadata.get("tool"),
                    "discovered_tools": mcp_metadata.get("discovered_tools", []),
                    "result_count": mcp_data.get("result_count", 0),
                    "error": tool_result.get("error"),
                })
            elif observation.get("tool_used") == "search_papers":
                paper_search_debug.append({
                    "provider": tool_result.get("provider"),
                    "query": tool_result.get("query"),
                    "year_filter": tool_result.get("year_filter"),
                    "result_count": tool_result.get("result_count", 0),
                    "total_estimate": tool_result.get("total_estimate"),
                    "error": tool_result.get("error"),
                })
            elif observation.get("tool_used") in {
                "get_paper_details",
                "import_paper_pdf",
            }:
                paper_import_debug.append({
                    "tool": observation.get("tool_used"),
                    "provider": tool_result.get("provider"),
                    "paper_id": tool_result.get("paper_id"),
                    "imported": tool_result.get("imported"),
                    "already_existed": tool_result.get("already_existed"),
                    "file_path": tool_result.get("file_path"),
                    "bytes_written": tool_result.get("bytes_written"),
                    "error": tool_result.get("error"),
                })

            retrieval = tool_result.get("retrieval")
            if not isinstance(retrieval, dict):
                continue

            retrieval_debug.append({
                "query": retrieval.get("query"),
                "expanded_query": retrieval.get("expanded_query"),
                "retrieval_method": retrieval.get("retrieval_method"),
                "fusion_method": retrieval.get("fusion_method"),
                "embedding_model": retrieval.get("embedding_model"),
                "fallback_reason": retrieval.get("fallback_reason"),
                "chunks_returned": len(retrieval.get("retrieved_chunks", [])),
                "candidate_k": retrieval.get("candidate_k"),
                "rank_constant": retrieval.get("rank_constant"),
                "bm25_candidates": retrieval.get("bm25_candidates"),
                "embedding_candidates": retrieval.get("embedding_candidates"),
                "index_cache": retrieval.get("index_cache"),
                "index_build_seconds": retrieval.get("index_build_seconds"),
            })

        debug_info = {
            "selected_skill": skill["name"],
            "skill_description": skill["description"],
            "allowed_tools":skill["allowed_tools"],
            "output_style":skill["output_style"],
            "model": SILICONFLOW_MODEL,
            "model_call": {
                "status": llm_result.get("status", "success"),
                "model": llm_result.get("model", SILICONFLOW_MODEL),
                "call_count": len(model_results),
                "error": llm_result.get("error"),
                "timeout_seconds": llm_result.get("timeout_seconds"),
                "max_retries_configured": llm_result.get(
                    "max_retries_configured"
                ),
                "max_output_tokens_configured": llm_result.get(
                    "max_output_tokens_configured"
                ),
            },
            "structured_output": structured_output.model_dump(),
            "structured_output_repair": structured_output_repair,
            "output_mode": output_mode,
            "context_keys": list(context.keys()),
            "memory_keys": list(self.memory.keys()),
            "relevant_memory_keys": list(context["relevant_memory"].keys()),
            "conversation_turns": len(self.conversation_memory.get_recent_history()),
            "persistent_memory_counts": self.memory_store.count_by_type(),
            "memory_context_enabled": self.memory_context_enabled,
            "memory_retrieval": {
                key: value
                for key, value in memory_retrieval.items()
                if key != "memories"
            } | {
                "selected": [
                    {
                        "memory_id": memory["memory_id"],
                        "memory_type": memory["memory_type"],
                        "retrieval_score": memory["retrieval_score"],
                        "signals": memory["signals"],
                        "included_in_prompt": self.memory_context_enabled,
                    }
                    for memory in memory_retrieval["memories"]
                ],
            },
            "memory_write_decision": {
                "status": memory_write_decision["status"],
                "reason_code": memory_write_decision["reason_code"],
                "reason": memory_write_decision["reason"],
                "suggestion_id": (
                    memory_write_decision["suggestion"]["suggestion_id"]
                    if memory_write_decision.get("suggestion")
                    else None
                ),
                "pending_count": len(self.pending_memory_suggestions),
            },
            "pending_tool_approvals": self.list_tool_approval_requests(),
            "plan_steps": [
                {

                    "step_id": step["step_id"],
                    "description": step["description"],
                    "tool_name": step["tool_name"],
                    "tool_args": step["tool_args"],
                }
                for step in execution_plan
            ],
            "tools_used": [
                observation["tool_used"]
                for observation in tool_observations
                if observation["tool_used"] is not None
            ],
            "tool_executions": [
                {
                    "step_id": observation.get("step_id"),
                    "tool_name": observation.get("tool_name"),
                    "status": observation.get("status"),
                    "error": observation.get("error"),
                    "permission": observation.get("permission"),
                    "latency_seconds": observation.get(
                        "latency_seconds",
                        0.0,
                    ),
                }
                for observation in tool_observations
            ],
            "retrievals": retrieval_debug,
            "paper_searches": paper_search_debug,
            "paper_imports": paper_import_debug,
            "mcp_calls": mcp_debug,
            "paper_selection": paper_selection,
            "agent_tool_loop": agent_tool_loop,
            "retrieval_decision": retrieval_decision,
            "context_evaluation": context_evaluation,
            "retrieval_loop": retrieval_loop,
            "context_stats": llm_result["context_stats"],
        }

        return {
            "response": response_content,
            "debug": debug_info,
            "usage": {
                "latency": usage["latency"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
            "execution_plan": execution_plan,
            "tool_observations": tool_observations,
            "retrieval_decision": retrieval_decision,
            "context_evaluation": context_evaluation,
            "retrieval_loop": retrieval_loop,
            "agent_tool_loop": agent_tool_loop,
            "model_call": debug_info["model_call"],
            "structured_output": structured_output.model_dump(),
            "structured_output_repair": structured_output_repair,
            "memory_write_decision": memory_write_decision,
            "memory_suggestion": memory_write_decision.get("suggestion"),
            "approval_requests": [
                observation["permission"]["approval_request"]
                for observation in tool_observations
                if isinstance(observation.get("permission"), dict)
                and observation["permission"].get("approval_request")
            ],
        }

    def resolve_paper_selection(self, user_input: str) -> tuple[str, dict]:
        """Resolve '第 N 篇' against the most recent process-local search."""
        match = re.search(r"第\s*(\d+)\s*篇", user_input)
        if match is None:
            match = re.search(
                r"\b(?:paper|result)\s*#?\s*(\d+)\b",
                user_input,
                flags=re.IGNORECASE,
            )
        if match is None:
            return user_input, {
                "status": "not_requested",
                "selected_index": None,
                "paper_id": None,
            }

        selected_index = int(match.group(1))
        if not self.last_paper_search_results:
            return user_input, {
                "status": "missing_search_results",
                "selected_index": selected_index,
                "paper_id": None,
                "reason": "No paper-search results exist in this Agent process.",
            }
        if not 1 <= selected_index <= len(self.last_paper_search_results):
            return user_input, {
                "status": "index_out_of_range",
                "selected_index": selected_index,
                "available_count": len(self.last_paper_search_results),
                "paper_id": None,
            }

        selected = self.last_paper_search_results[selected_index - 1]
        paper_id = selected.get("paper_id")
        if not isinstance(paper_id, str) or not re.fullmatch(
            r"[0-9a-f]{40}",
            paper_id,
            flags=re.IGNORECASE,
        ):
            return user_input, {
                "status": "invalid_cached_paper_id",
                "selected_index": selected_index,
                "paper_id": None,
            }

        planning_input = (
            f"{user_input}\nSemantic Scholar paper_id: {paper_id}"
        )
        return planning_input, {
            "status": "resolved",
            "selected_index": selected_index,
            "paper_id": paper_id,
            "title": selected.get("title"),
        }

    def _remember_paper_search_results(
        self,
        tool_observations: list[dict],
    ) -> None:
        """Cache only compact selection fields from the latest search."""
        for observation in tool_observations:
            if observation.get("tool_used") != "search_papers":
                continue
            result = observation.get("tool_result")
            if not isinstance(result, dict) or result.get("error"):
                continue
            self.last_paper_search_results = [
                {
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "url": paper.get("url"),
                }
                for paper in result.get("papers", [])
                if isinstance(paper, dict)
            ]

    def run_observation_tool_loop(
        self,
        user_input: str,
        skill: dict,
        retrieval_decision: dict,
        execution_plan: list[dict],
        tool_observations: list[dict],
        paper_selection: dict | None = None,
    ) -> tuple[list[dict], list[dict], dict, dict]:
        """Execute a small number of tool actions derived from observations."""
        retrieval_decision = dict(retrieval_decision)
        trace = {
            "enabled": True,
            "max_dynamic_steps": AGENT_TOOL_LOOP_MAX_STEPS,
            "dynamic_steps_used": 0,
            "status": None,
            "stopped_reason": None,
            "paper_selection": paper_selection or {},
            "iterations": [],
        }
        dynamic_steps_used = 0

        while True:
            self._remember_paper_search_results(tool_observations)
            decision = decide_next_tool_action(
                user_input=user_input,
                skill=skill,
                observations=tool_observations,
                dynamic_steps_used=dynamic_steps_used,
                max_steps=AGENT_TOOL_LOOP_MAX_STEPS,
            )
            trace["iterations"].append({
                "iteration": len(trace["iterations"]) + 1,
                "observed_tool": (
                    tool_observations[-1].get("tool_used")
                    if tool_observations
                    else None
                ),
                "status": decision.get("status"),
                "reason_code": decision.get("reason_code"),
                "reason": decision.get("reason"),
                "user_input_request": decision.get("user_input_request"),
                "choices": decision.get("choices", []),
                "proposed_tool": (
                    decision.get("next_step") or {}
                ).get("tool_name"),
            })

            if decision.get("status") != "continue":
                trace["status"] = decision.get("status")
                trace["stopped_reason"] = decision.get("reason_code")
                break

            next_step = decision.get("next_step") or {}
            tool_name = next_step.get("tool_name")
            if dynamic_steps_used >= AGENT_TOOL_LOOP_MAX_STEPS:
                trace["iterations"][-1]["status"] = "stop"
                trace["iterations"][-1]["reason_code"] = (
                    "max_dynamic_steps_reached"
                )
                trace["iterations"][-1]["reason"] = (
                    "The Agent tool-loop budget has been exhausted."
                )
                trace["status"] = "stop"
                trace["stopped_reason"] = "max_dynamic_steps_reached"
                break
            if tool_name not in skill.get("allowed_tools", []):
                trace["iterations"][-1]["status"] = "stop"
                trace["iterations"][-1]["reason_code"] = "tool_not_allowed"
                trace["iterations"][-1]["reason"] = (
                    f"Tool '{tool_name}' is not allowed for skill "
                    f"'{skill.get('name')}'."
                )
                trace["status"] = "stop"
                trace["stopped_reason"] = "tool_not_allowed"
                break

            next_step = dict(next_step)
            next_step["step_id"] = max(
                [
                    step.get("step_id", 0)
                    for step in execution_plan
                    if isinstance(step.get("step_id"), int)
                ],
                default=0,
            ) + 1
            execution_plan.append(next_step)
            retrieval_update = decision.get("retrieval_decision_update")
            if retrieval_update:
                retrieval_decision.update(retrieval_update)
            new_observation = self.execute_plan([next_step])[0]
            tool_observations.append(new_observation)
            dynamic_steps_used += 1
            trace["dynamic_steps_used"] = dynamic_steps_used

        return (
            execution_plan,
            tool_observations,
            retrieval_decision,
            trace,
        )

    @staticmethod
    def _evaluation_trace(evaluation: dict) -> dict:
        signals = evaluation.get("signals", {})
        return {
            "status": evaluation.get("status"),
            "is_sufficient": evaluation.get("is_sufficient"),
            "reason_code": evaluation.get("reason_code"),
            "recommended_action": evaluation.get("recommended_action"),
            "query_term_coverage": signals.get("query_term_coverage"),
            "maximum_embedding_score": signals.get("maximum_embedding_score"),
            "dual_source_chunks": signals.get("dual_source_chunks"),
            "missing_query_terms": signals.get("missing_query_terms", []),
        }

    @staticmethod
    def _index_cache_trace(observation: dict | None) -> dict | None:
        if not isinstance(observation, dict):
            return None
        tool_result = observation.get("tool_result")
        if not isinstance(tool_result, dict):
            return None
        retrieval = tool_result.get("retrieval")
        if not isinstance(retrieval, dict):
            return None
        return retrieval.get("index_cache")

    def run_retrieval_loop(
        self,
        retrieval_decision: dict,
        execution_plan: list[dict],
        tool_observations: list[dict],
    ) -> tuple[list[dict], list[dict], dict, dict]:
        """Evaluate evidence and run a bounded retrieval retry when needed."""
        context_evaluation = evaluate_retrieval_context(
            retrieval_decision=retrieval_decision,
            tool_observations=tool_observations,
        )
        loop_trace = {
            "enabled": retrieval_decision.get("needs_retrieval", False),
            "max_retries": AGENTIC_RAG_MAX_RETRIES,
            "retries_used": 0,
            "attempts": [],
            "final_evidence": "initial",
            "stopped_reason": None,
        }

        if not retrieval_decision.get("needs_retrieval"):
            loop_trace["stopped_reason"] = "retrieval_not_required"
            return (
                execution_plan,
                tool_observations,
                context_evaluation,
                loop_trace,
            )

        retrieval_observations = [
            observation
            for observation in tool_observations
            if observation.get("tool_used") == "retrieve_from_file"
        ]
        loop_trace["attempts"].append({
            "attempt": 1,
            "query": retrieval_decision.get("query"),
            "query_refinement": None,
            "evaluation": self._evaluation_trace(context_evaluation),
            "index_cache": self._index_cache_trace(
                retrieval_observations[0]
                if retrieval_observations
                else None
            ),
        })

        retries_used = 0
        while should_retry(
            evaluation=context_evaluation,
            retries_used=retries_used,
            max_retries=AGENTIC_RAG_MAX_RETRIES,
        ):
            refinement = build_refined_query(
                original_query=retrieval_decision.get("query") or "",
                evaluation=context_evaluation,
            )
            next_step_id = max(
                [
                    step.get("step_id", 0)
                    for step in execution_plan
                    if isinstance(step.get("step_id"), int)
                ],
                default=0,
            ) + 1
            retry_step = build_retry_step(
                source=retrieval_decision["source"],
                refined_query=refinement["refined_query"],
                step_id=next_step_id,
            )
            execution_plan.append(retry_step)
            retry_observation = self.execute_plan([retry_step])[0]
            retrieval_observations.append(retry_observation)
            retries_used += 1

            retry_evaluation = evaluate_retrieval_context(
                retrieval_decision=retrieval_decision,
                tool_observations=[retry_observation],
            )
            loop_trace["attempts"].append({
                "attempt": retries_used + 1,
                "query": refinement["refined_query"],
                "query_refinement": refinement,
                "evaluation": self._evaluation_trace(retry_evaluation),
                "index_cache": self._index_cache_trace(retry_observation),
            })

            merged_observation = merge_retrieval_observations(
                observations=retrieval_observations,
                original_query=retrieval_decision.get("query") or "",
                max_chunks=AGENTIC_RAG_MAX_MERGED_CHUNKS,
            )
            if merged_observation is not None:
                context_evaluation = evaluate_retrieval_context(
                    retrieval_decision=retrieval_decision,
                    tool_observations=[merged_observation],
                )
                non_retrieval_observations = [
                    observation
                    for observation in tool_observations
                    if observation.get("tool_used") != "retrieve_from_file"
                ]
                tool_observations = [
                    *non_retrieval_observations,
                    merged_observation,
                ]
                loop_trace["final_evidence"] = "merged_attempts"
            else:
                context_evaluation = retry_evaluation

        loop_trace["retries_used"] = retries_used
        if context_evaluation.get("is_sufficient"):
            loop_trace["stopped_reason"] = (
                "sufficient_after_retry"
                if retries_used
                else "initial_context_sufficient"
            )
        elif retries_used >= AGENTIC_RAG_MAX_RETRIES:
            loop_trace["stopped_reason"] = "max_retries_reached"
        else:
            loop_trace["stopped_reason"] = (
                "action_not_retryable:"
                f"{context_evaluation.get('recommended_action')}"
            )

        loop_trace["final_evaluation"] = self._evaluation_trace(
            context_evaluation
        )
        return (
            execution_plan,
            tool_observations,
            context_evaluation,
            loop_trace,
        )

    def execute_plan(
        self,
        execution_plan: list[dict],
        user_input: str | None = None,
        output_mode: str = "text",
    ) -> list[dict]:
        observations = []
        for step in execution_plan:
            tool_name = step["tool_name"]
            if tool_name is None:
                observations.append(ToolObservation(
                    step_id=step["step_id"],
                    description=step["description"],
                    tool_name=None,
                    status="skipped",
                ).to_runtime_dict())
                continue

            try:
                tool_call = build_tool_call(
                    tool_name=tool_name,
                    args=step["tool_args"],
                )
            except ToolInputValidationError as error:
                observations.append(ToolObservation(
                    step_id=step["step_id"],
                    description=step["description"],
                    tool_name=tool_name,
                    status="error",
                    arguments=step.get("tool_args", {}),
                    error=ToolError(
                        code="invalid_tool_arguments",
                        message=str(error),
                        retryable=False,
                        details={
                            "validation_errors": error.validation_errors,
                        },
                    ),
                ).to_runtime_dict())
                continue
            except ValueError as error:
                observations.append(ToolObservation(
                    step_id=step["step_id"],
                    description=step["description"],
                    tool_name=tool_name,
                    status="error",
                    arguments=step.get("tool_args", {}),
                    error=ToolError(
                        code="invalid_tool_call",
                        message=str(error),
                        retryable=False,
                    ),
                ).to_runtime_dict())
                continue

            permission = check_tool_permission(
                tool_spec=tool_call.get("spec"),
                user_input=user_input,
            )
            if not permission.allowed:
                permission_payload = permission.model_dump()
                if (
                    permission.requires_confirmation
                    and permission.reason_code == "confirmation_required"
                ):
                    approval_request = self._get_approval_manager().create_request(
                        tool_name=tool_name,
                        arguments=tool_call["args"],
                        risk_level=permission.risk_level,
                        reason=permission.reason,
                        resume_user_input=user_input,
                        resume_output_mode=output_mode,
                        resume_step_id=step["step_id"],
                        resume_description=step["description"],
                    )
                    permission_payload["approval_request"] = (
                        approval_request.model_dump()
                    )
                observations.append(ToolObservation(
                    step_id=step["step_id"],
                    description=step["description"],
                    tool_name=tool_name,
                    status="error",
                    arguments=tool_call["args"],
                    error=ToolError(
                        code="permission_denied",
                        message=permission.reason,
                        retryable=False,
                        details={
                            "reason_code": permission.reason_code,
                        },
                    ),
                    permission=permission_payload,
                ).to_runtime_dict())
                continue

            runtime_result = run_tool(tool_call)
            raw_error = runtime_result.get("error")
            if raw_error:
                tool_error = ToolError.model_validate(raw_error)
            else:
                tool_error = error_from_tool_result(
                    runtime_result.get("tool_result")
                )
            status = runtime_result.get("status") or (
                "error" if tool_error else "success"
            )
            observations.append(ToolObservation(
                step_id=step["step_id"],
                description=step["description"],
                tool_name=runtime_result.get("tool_used", tool_name),
                status=status,
                arguments=runtime_result.get(
                    "tool_args",
                    tool_call["args"],
                ),
                result=runtime_result.get("tool_result"),
                error=tool_error,
                permission=permission.model_dump(),
                latency_seconds=runtime_result.get(
                    "latency_seconds",
                    0.0,
                ),
            ).to_runtime_dict())

        return observations

    def _get_approval_manager(self) -> ApprovalManager:
        """Support lightweight test instances constructed with ``__new__``."""
        manager = getattr(self, "approval_manager", None)
        if manager is None:
            manager = ApprovalManager()
            self.approval_manager = manager
        return manager

    def list_tool_approval_requests(self) -> list[dict]:
        return [
            request.model_dump()
            for request in self._get_approval_manager().list_requests(
                status="pending"
            )
        ]

    def approve_tool_request(self, request_id: str) -> dict:
        """Execute the bound call, then resume from its Observation."""
        manager = self._get_approval_manager()
        request = manager.authorize(request_id)
        try:
            tool_call = build_tool_call(
                request.tool_name,
                request.arguments,
            )
            spec = tool_call["spec"]
            if (
                not spec.requires_confirmation
                or spec.risk_level != request.risk_level
            ):
                raise ValueError(
                    "Tool risk metadata changed after approval was requested."
                )
            runtime_result = run_tool(tool_call)
        except Exception as error:
            completed = manager.complete(
                request_id,
                execution_status="error",
            )
            return {
                "approval_request": completed.model_dump(),
                "execution": {
                    "tool_used": request.tool_name,
                    "tool_args": request.arguments,
                    "tool_result": None,
                    "status": "error",
                    "error": {
                        "code": "approved_tool_execution_error",
                        "message": f"{type(error).__name__}: {error}",
                        "retryable": False,
                    },
                    "latency_seconds": 0.0,
                },
            }

        execution_status = (
            "success"
            if runtime_result.get("status") == "success"
            else "error"
        )
        completed = manager.complete(
            request_id,
            execution_status=execution_status,
        )
        raw_error = runtime_result.get("error")
        tool_error = (
            ToolError.model_validate(raw_error)
            if raw_error
            else error_from_tool_result(runtime_result.get("tool_result"))
        )
        resumed_plan = [{
            "step_id": request.resume_step_id or 1,
            "description": (
                request.resume_description
                or f"Execute approved tool {request.tool_name}."
            ),
            "tool_name": request.tool_name,
            "tool_args": request.arguments,
        }]
        resumed_observation = ToolObservation(
            step_id=resumed_plan[0]["step_id"],
            description=resumed_plan[0]["description"],
            tool_name=request.tool_name,
            status=execution_status,
            arguments=request.arguments,
            result=runtime_result.get("tool_result"),
            error=tool_error,
            permission={
                "allowed": True,
                "reason_code": "approved_request",
                "reason": "The user approved the exact stored tool call.",
                "risk_level": request.risk_level,
                "requires_confirmation": True,
                "confirmation_source": request.request_id,
            },
            latency_seconds=runtime_result.get("latency_seconds", 0.0),
        ).to_runtime_dict()

        if request.resume_user_input is None:
            return {
                "approval_request": completed.model_dump(),
                "execution": runtime_result,
                "resumed": False,
                "agent_result": None,
            }

        agent_result = self.run(
            request.resume_user_input,
            output_mode=request.resume_output_mode,
            _resume_execution_plan=resumed_plan,
            _resume_tool_observations=[resumed_observation],
        )
        return {
            "approval_request": completed.model_dump(),
            "execution": runtime_result,
            "resumed": True,
            "agent_result": agent_result,
        }

    def reject_tool_request(self, request_id: str) -> dict:
        return self._get_approval_manager().reject(request_id).model_dump()

    def call_llm(self, context: dict) -> dict:
        prompt_bundle = build_prompt_bundle(context)

        result = self.llm.generate(
            system_prompt=prompt_bundle["system_prompt"],
            user_prompt=prompt_bundle["user_prompt"]
        )
        result["context_stats"] = prompt_bundle["stats"]

        return result


    def clear_conversation(self):
        self.conversation_memory.clear_history()

    def remember_long_term(
        self,
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.7,
    ) -> dict:
        """Store only information the user explicitly asks to remember."""
        return self.memory_store.add_memory(
            memory_type="long_term",
            content=content,
            source="explicit_user_statement",
            confidence=1.0,
            importance=importance,
            tags=tags,
        )

    def record_episode(
        self,
        task: str,
        outcome: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Manual v1.4.1 API; automatic episode writing comes later."""
        episode_metadata = {
            "task": task.strip(),
            "outcome": outcome.strip(),
            **(metadata or {}),
        }
        return self.memory_store.add_memory(
            memory_type="episodic",
            content=f"Task: {task.strip()}\nOutcome: {outcome.strip()}",
            source="task_result",
            confidence=1.0,
            importance=0.6,
            tags=tags,
            metadata=episode_metadata,
        )

    def list_persistent_memories(
        self,
        memory_type: str | None = None,
    ) -> list[dict]:
        return self.memory_store.list_memories(memory_type=memory_type)

    def retrieve_persistent_memories(
        self,
        query: str,
        *,
        top_k: int = MEMORY_RETRIEVAL_TOP_K,
    ) -> dict:
        """Retrieve locally; this method never calls the LLM."""
        return retrieve_memories(
            query=query,
            records=self.memory_store.list_memories(),
            top_k=top_k,
            min_relevance=MEMORY_RETRIEVAL_MIN_RELEVANCE,
        )

    def set_memory_context_enabled(self, enabled: bool) -> bool:
        """Explicit per-process consent for adding memories to LLM prompts."""
        self.memory_context_enabled = bool(enabled)
        return self.memory_context_enabled

    def propose_memory_write(self, user_input: str) -> dict:
        """Create a process-local candidate without writing to disk."""
        existing = self.memory_store.list_memories()
        pending = list(self.pending_memory_suggestions.values())
        decision = evaluate_memory_write(
            user_input,
            existing_records=[*existing, *pending],
        )
        suggestion = decision.get("suggestion")
        if suggestion:
            while len(self.pending_memory_suggestions) >= MAX_PENDING_SUGGESTIONS:
                oldest_id = next(iter(self.pending_memory_suggestions))
                del self.pending_memory_suggestions[oldest_id]
            self.pending_memory_suggestions[
                suggestion["suggestion_id"]
            ] = suggestion
        return decision

    def list_memory_suggestions(self) -> list[dict]:
        return [
            dict(suggestion)
            for suggestion in self.pending_memory_suggestions.values()
        ]

    def approve_memory_suggestion(self, suggestion_id: str) -> dict:
        """Persist a candidate only after explicit user approval."""
        suggestion = self.pending_memory_suggestions.get(suggestion_id)
        if suggestion is None:
            raise KeyError(f"Unknown memory suggestion: {suggestion_id}")

        duplicate_check = evaluate_memory_write(
            suggestion["content"],
            existing_records=self.memory_store.list_memories(),
        )
        if duplicate_check["status"] == "duplicate":
            del self.pending_memory_suggestions[suggestion_id]
            raise ValueError("An equivalent persistent memory already exists.")

        record = self.memory_store.add_memory(
            memory_type=suggestion["memory_type"],
            content=suggestion["content"],
            source="user_approved_policy_suggestion",
            confidence=suggestion["confidence"],
            importance=suggestion["importance"],
            tags=suggestion["tags"],
            metadata={
                "suggestion_id": suggestion_id,
                "category": suggestion["category"],
                "policy_reason": suggestion["reason"],
            },
        )
        del self.pending_memory_suggestions[suggestion_id]
        return record

    def dismiss_memory_suggestion(self, suggestion_id: str) -> bool:
        return self.pending_memory_suggestions.pop(
            suggestion_id,
            None,
        ) is not None

    def update_persistent_memory(self, memory_id: str, **changes) -> dict:
        return self.memory_store.update_memory(memory_id, **changes)

    def forget_memory(self, memory_id: str) -> bool:
        return self.memory_store.delete_memory(memory_id)
