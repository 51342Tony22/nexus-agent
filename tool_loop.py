"""Pure decisions for the bounded Observation -> Action tool loop.

This module deliberately does not execute tools.  It only interprets the
latest observation and proposes a next step, which keeps the Agent loop
deterministic, inspectable, and easy to test.
"""

import re


DEFAULT_IMPORTED_PAPER_QUERY = (
    "main ideas methods datasets results relevance to FYP"
)


def has_analysis_intent(user_input: str) -> bool:
    """Return whether the same request asks to analyze the imported paper."""
    return bool(re.search(
        r"(?:分析|总结|解释|解读|方法|实验|结果|贡献|局限)"
        r"|\b(?:analy[sz]e|summari[sz]e|explain|method|results?|limitations?)\b",
        user_input,
        flags=re.IGNORECASE,
    ))


def build_imported_paper_query(
    user_input: str,
    paper_id: str | None = None,
) -> str:
    """Keep the analysis need while removing selection/import syntax."""
    query = user_input
    if paper_id:
        query = re.sub(re.escape(paper_id), " ", query, flags=re.IGNORECASE)
    query = re.sub(r"第\s*\d+\s*篇", " ", query)
    query = re.sub(r"\bpaper\s*#?\s*\d+\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(
        r"(?:请帮我|请|帮我|下载|保存|导入|获取|论文|这篇|该篇)"
        r"|\b(?:download|save|import|paper|the)\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(
        r"^\s*(?:并且?|然后|再|and|then)\s*",
        "",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(
        r"^\s*(?:分析|总结|解释|解读|analy[sz]e|summari[sz]e|explain)\s*",
        "",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(r"\s+", " ", query).strip(" ，,。.:：;；-_")
    return query or DEFAULT_IMPORTED_PAPER_QUERY


def _paper_choices(papers: list[dict]) -> list[dict]:
    choices = []
    for index, paper in enumerate(papers, start=1):
        if not isinstance(paper, dict):
            continue
        choices.append({
            "index": index,
            "paper_id": paper.get("paper_id"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "url": paper.get("url"),
        })
    return choices


def decide_next_tool_action(
    user_input: str,
    skill: dict,
    observations: list[dict],
    dynamic_steps_used: int,
    max_steps: int,
) -> dict:
    """Evaluate observations and propose one bounded next action."""
    if not observations:
        return {
            "status": "stop",
            "reason_code": "no_observation",
            "reason": "There is no tool observation to evaluate.",
            "next_step": None,
        }

    latest = observations[-1]
    tool_used = latest.get("tool_used")
    result = latest.get("tool_result")
    if tool_used is None or not isinstance(result, dict):
        return {
            "status": "stop",
            "reason_code": "no_tool_result",
            "reason": "The latest plan step produced no structured tool result.",
            "next_step": None,
        }

    error = result.get("error")
    if error:
        return {
            "status": "stop",
            "reason_code": f"{tool_used}_failed",
            "reason": "The latest tool failed, so no dependent tool is safe to run.",
            "error": error,
            "next_step": None,
        }

    if tool_used == "search_papers":
        choices = _paper_choices(result.get("papers", []))
        if not choices:
            return {
                "status": "stop",
                "reason_code": "no_papers_found",
                "reason": "The search returned no selectable papers.",
                "next_step": None,
            }
        return {
            "status": "needs_user_input",
            "reason_code": "paper_selection_required",
            "reason": (
                "Search produced multiple candidates; the Agent must not choose "
                "or download one without the user's selection."
            ),
            "user_input_request": "请选择结果，例如：下载第1篇并分析方法。",
            "choices": choices,
            "next_step": None,
        }

    if tool_used == "import_paper_pdf":
        if not result.get("imported") or not result.get("file_path"):
            return {
                "status": "stop",
                "reason_code": "paper_not_imported",
                "reason": "No verified local PDF path was produced.",
                "next_step": None,
            }
        if not has_analysis_intent(user_input):
            return {
                "status": "stop",
                "reason_code": "paper_imported",
                "reason": "The requested import is complete; analysis was not requested.",
                "next_step": None,
            }
        if dynamic_steps_used >= max_steps:
            return {
                "status": "stop",
                "reason_code": "max_dynamic_steps_reached",
                "reason": "The Agent tool-loop budget has been exhausted.",
                "next_step": None,
            }

        paper_id = result.get("paper_id")
        query = build_imported_paper_query(user_input, paper_id)
        file_path = result["file_path"]
        return {
            "status": "continue",
            "reason_code": "analyze_imported_paper",
            "reason": (
                "Import produced a verified local file and the user also requested "
                "analysis, so local retrieval is the next grounded action."
            ),
            "next_step": {
                "description": "Retrieve evidence from the newly imported paper.",
                "tool_name": "retrieve_from_file",
                "tool_args": {
                    "file_path": file_path,
                    "query": query,
                    "top_k": 3,
                },
            },
            "retrieval_decision_update": {
                "needs_retrieval": True,
                "reason_code": "imported_paper_follow_up",
                "reason": (
                    "The imported PDF must be retrieved before its content can be analyzed."
                ),
                "strategy": "hybrid_file_retrieval",
                "source_type": "imported_local_file",
                "source": file_path,
                "source_was_explicit": False,
                "query": query,
            },
        }

    if tool_used == "retrieve_from_file":
        return {
            "status": "stop",
            "reason_code": "dynamic_retrieval_complete",
            "reason": "The dynamic retrieval observation is ready for evaluation.",
            "next_step": None,
        }

    return {
        "status": "stop",
        "reason_code": "no_follow_up_action",
        "reason": f"Tool '{tool_used}' does not require a dynamic follow-up.",
        "next_step": None,
    }
