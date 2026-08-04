#planner.py
#根据当前skill生成执行计划

from tool_registry import extract_dataset_names
from retrieval_decider import decide_retrieval, extract_file_reference
from datetime import datetime
import re

def build_plan_with_allowed_tools(
    requested_steps: list[dict],
    allowed_tools: list[str]
) -> list[dict]:
    plan = []
    for step in requested_steps:
        tool_name = step["tool_name"]
        if tool_name in allowed_tools:
            plan.append(step)

        else:
            plan.append({
                "step_id": step["step_id"],
                "description": (
                    f"Skipped tool {tool_name} because it is not allowed "
                    f"for the selected skill."
                ),
                "tool_name": None,
                "tool_args": {}
            })

    return plan

def extract_file_path(user_input: str) -> str:
    # Backward-compatible helper for callers outside the Agent.
    return extract_file_reference(user_input) or "fyp_description.md"


def extract_paper_search_request(user_input: str) -> dict:
    """Remove search commands while preserving the scholarly information need."""
    text = user_input.strip()
    lower_text = text.lower()
    explicit_year = re.search(
        r"(?<!\d)((?:19|20)\d{2}(?:-(?:(?:19|20)?\d{2})?)?)(?!\d)",
        text,
    )
    year = explicit_year.group(1) if explicit_year else None
    if year and re.fullmatch(r"\d{4}-\d{2}", year):
        year = f"{year[:5]}20{year[5:]}"

    if year is None and any(
        marker in lower_text
        for marker in ["最新", "近期", "latest", "recent"]
    ):
        year = f"{datetime.now().year - 2}-"

    query = re.sub(
        r"^\s*(?:请帮我|请|帮我)?\s*"
        r"(?:搜索|查找|检索|找|search(?:\s+for)?|find|look\s+up)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if explicit_year:
        query = query.replace(explicit_year.group(1), " ")
    query = re.sub(
        r"^\s*(?:最新|近期|latest|recent)?\s*"
        r"(?:的|关于|papers?\s+(?:about|on)|literature\s+(?:about|on))?\s*",
        "",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(
        r"\s*(?:的)?\s*(?:最新|近期|相关)?\s*(?:论文|文献|papers?)\s*$",
        "",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(r"\s+", " ", query).strip(" ，,。.:：;-")

    return {
        "query": query or "reasoning distillation",
        "year": year,
        "limit": 5,
    }


def extract_semantic_scholar_paper_id(user_input: str) -> str | None:
    match = re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", user_input, re.I)
    return match.group(0) if match else None


def extract_mcp_library_query(user_input: str) -> str:
    """Remove the protocol/UI wording and preserve the research need."""
    query = user_input
    query = re.sub(
        r"(?:请帮我|请|帮我|使用|通过|用)?\s*mcp",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(
        r"(?:在)?\s*(?:本地\s*)?(?:fyp\s*)?(?:论文库|paper\s+library|library)",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(
        r"(?:搜索|查找|检索|找|列出|search|find|list)",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(r"\s+", " ", query).strip(" ，,。.:：;-_")
    return query or "all"

def create_plan(
    skill: dict,
    user_input: str,
    retrieval_decision: dict | None = None,
) -> list[dict]:
    skill_name = skill["name"]
    allowed_tools = skill.get("allowed_tools", [])
    text = user_input.lower()
    if skill_name == "mcp_library_skill":
        return build_plan_with_allowed_tools(
            requested_steps=[{
                "step_id": 1,
                "description": (
                    "Discover and call the FYP Library MCP catalog search tool."
                ),
                "tool_name": "mcp_search_fyp_library",
                "tool_args": {
                    "query": extract_mcp_library_query(user_input),
                    "limit": 6,
                },
            }],
            allowed_tools=allowed_tools,
        )
    if skill_name == "paper_import_skill":
        if re.search(
            r"(?:不要|别|无需|不需要)\s*(?:下载|保存|导入)"
            r"|\b(?:do\s+not|don't)\s+(?:download|import|save)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return [{
                "step_id": 1,
                "description": "Paper import was explicitly negated by the user.",
                "tool_name": None,
                "tool_args": {},
            }]
        paper_id = extract_semantic_scholar_paper_id(user_input)
        if paper_id is None:
            return [{
                "step_id": 1,
                "description": (
                    "A 40-character Semantic Scholar paper_id is required. "
                    "Search first, then select one result by paper_id."
                ),
                "tool_name": None,
                "tool_args": {},
            }]

        details_only = any(
            marker in text
            for marker in ["详情", "查看", "details", "inspect"]
        ) and not any(
            marker in text
            for marker in ["下载", "保存", "导入", "download", "import"]
        )
        tool_name = "get_paper_details" if details_only else "import_paper_pdf"
        return build_plan_with_allowed_tools(
            requested_steps=[{
                "step_id": 1,
                "description": (
                    "Fetch selected paper details."
                    if details_only
                    else "Import the selected open-access PDF into project files."
                ),
                "tool_name": tool_name,
                "tool_args": {"paper_id": paper_id},
            }],
            allowed_tools=allowed_tools,
        )
    if skill_name == "paper_search_skill":
        request = extract_paper_search_request(user_input)
        return build_plan_with_allowed_tools(
            requested_steps=[{
                "step_id": 1,
                "description": "Search Semantic Scholar for relevant paper metadata.",
                "tool_name": "search_papers",
                "tool_args": request,
            }],
            allowed_tools=allowed_tools,
        )
    if skill_name == "dataset_selection_skill":
        dataset_names = extract_dataset_names(user_input)
        return build_plan_with_allowed_tools(
            requested_steps=[
                {
                    "step_id": 1,
                    "description": "Compare the datasets mentioned by the user.",
                    "tool_name": "compare_datasets",
                    "tool_args": {
                        "dataset_names": dataset_names
                    }
                }
            ],
            allowed_tools=allowed_tools
        )
    if skill_name == "teacher_student_model_skill":
        task_type = "classification" if any(
            word in text for word in ["classification", "分类", "distilbert"]
        ) else "generation"
        return build_plan_with_allowed_tools(
            requested_steps=[
                {
                    "step_id": 1,
                    "description": "Suggest suitable teacher models.",
                    "tool_name": "suggest_teacher_models",
                    "tool_args": {}
                },
                {
                    "step_id": 2,
                    "description": "Suggest suitable student models.",
                    "tool_name": "suggest_student_models",
                    "tool_args": {
                        "task_type": task_type
                    }
                }
            ],
            allowed_tools=allowed_tools
        )
    if skill_name == "experiment_design_skill":
        dataset_names = extract_dataset_names(user_input)
        return build_plan_with_allowed_tools(
            requested_steps=[
                {
                    "step_id": 1,
                    "description": "Generate core experiment groups.",
                    "tool_name": "generate_experiment_groups",
                    "tool_args": {}
                },
                {
                    "step_id": 2,
                    "description": "Compare candidate reasoning datasets.",
                    "tool_name": "compare_datasets",
                    "tool_args": {
                        "dataset_names": dataset_names
                    }
                },
                {
                    "step_id": 3,
                    "description": "Suggest suitable student models for the experiment.",
                    "tool_name": "suggest_student_models",
                    "tool_args": {
                        "task_type": "generation"
                    }
                }
            ],
            allowed_tools=allowed_tools
        )
    if skill_name == "file_analysis_skill":
        retrieval_decision = retrieval_decision or decide_retrieval(
            user_input=user_input,
            skill=skill,
        )

        if not retrieval_decision["needs_retrieval"]:
            return [{
                "step_id": 1,
                "description": retrieval_decision["reason"],
                "tool_name": None,
                "tool_args": {},
            }]

        return build_plan_with_allowed_tools(
            requested_steps=[
                {
                    "step_id": 1,
                    "description": "Retrieve the most relevant chunks from the local file mentioned by the user.",
                    "tool_name": "retrieve_from_file",
                    "tool_args": {
                        "file_path": retrieval_decision["source"],
                        "query": retrieval_decision["query"],
                        "top_k": 3
                    }
                }
            ],
            allowed_tools=allowed_tools
        )
    return [
        {
            "step_id": 1,
            "description": "No external tool is required. Answer based on context and conversation history.",
            "tool_name": None,
            "tool_args": {}
        }

    ]

def select_file_reader_tool(file_path: str) -> str:
    lower_path = file_path.lower()

    if lower_path.endswith(".pdf"):
        return "read_pdf_file"

    return "read_text_file"
