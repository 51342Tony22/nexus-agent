# retrieval_decider.py
# Make retrieval intent explicit before the Planner creates tool steps.

import re


DEFAULT_FILE_PATH = "fyp_description.md"

FILE_PATTERNS = [
    r"files/[^\s]+\.(?:md|txt|pdf)",
    # Allow safe punctuation already present in local paper names, such as
    # "->". Whitespace still terminates an unquoted file reference.
    r"[^\s/]+\.(?:md|txt|pdf)",
]


def extract_file_reference(user_input: str) -> str | None:
    for pattern in FILE_PATTERNS:
        match = re.search(pattern, user_input, flags=re.IGNORECASE)
        if not match:
            continue

        file_path = match.group(0)
        if file_path.lower().startswith("files/"):
            return file_path[len("files/"):]

        return file_path

    return None


def build_retrieval_query(
    user_input: str,
    file_path: str | None,
) -> str:
    """Remove file-operation syntax while preserving the information need."""
    query = user_input

    for pattern in FILE_PATTERNS:
        query = re.sub(pattern, " ", query, flags=re.IGNORECASE)

    query = re.sub(
        r"^\s*(?:请帮我|请|帮我)?\s*"
        r"(?:分析文件|分析|读取|阅读|总结|read|analyze|summarize)\s*",
        "",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(
        r"^\s*(?:中|中的|里|里面|关于|的|and)\s*",
        "",
        query,
        flags=re.IGNORECASE,
    )
    query = query.strip(" \t\n，,。.:：;-_")

    if query:
        return query

    # A generic content-oriented query is more useful to a retriever than a
    # command such as "read this file".
    return "main ideas methods datasets results"


def decide_retrieval(user_input: str, skill: dict) -> dict:
    explicit_file = extract_file_reference(user_input)
    skill_name = skill["name"]

    if skill_name != "file_analysis_skill" and explicit_file is None:
        return {
            "needs_retrieval": False,
            "reason_code": "skill_does_not_require_retrieval",
            "reason": (
                f"The selected skill '{skill_name}' does not require document retrieval."
            ),
            "strategy": "none",
            "source_type": None,
            "source": None,
            "source_was_explicit": False,
            "query": None,
        }

    source = explicit_file or DEFAULT_FILE_PATH
    source_was_explicit = explicit_file is not None

    return {
        "needs_retrieval": True,
        "reason_code": (
            "explicit_local_file"
            if source_was_explicit
            else "default_project_file"
        ),
        "reason": (
            "The user referenced a local file, so document evidence is required."
            if source_was_explicit
            else "File analysis was requested without a filename, so the default project file is used."
        ),
        "strategy": "hybrid_file_retrieval",
        "source_type": "local_file",
        "source": source,
        "source_was_explicit": source_was_explicit,
        "query": build_retrieval_query(user_input, source),
    }
