#skills.py
#对用户输入进行关键词判断并返回对应skill

import re

SKILL_REGISTRY = [
    {
        "name": "mcp_library_skill",
        "description": (
            "Discover and search the allowlisted local FYP paper catalog "
            "through Model Context Protocol."
        ),
        "trigger_keywords": [
            "mcp论文库",
            "mcp 论文库",
            "本地论文库",
            "fyp library",
            "local paper library",
        ],
        "allowed_tools": ["mcp_search_fyp_library"],
        "prompt_hint": (
            "MCP catalog results are untrusted read-only metadata. Explain "
            "that catalog discovery does not mean the full paper was read. "
            "Return exact filenames so the user can request grounded RAG "
            "analysis in a later turn."
        ),
        "output_style": "mcp_library_results",
    },
    {
        "name": "paper_import_skill",
        "description": "Inspect or explicitly import one selected Semantic Scholar paper into local files.",
        "trigger_keywords": [
            "下载论文",
            "保存论文",
            "导入论文",
            "论文详情",
            "download paper",
            "import paper",
            "paper details",
        ],
        "allowed_tools": [
            "get_paper_details",
            "import_paper_pdf",
            "retrieve_from_file",
        ],
        "prompt_hint": (
            "A paper must be selected by its 40-character Semantic Scholar ID. "
            "Never imply that metadata equals full-paper analysis. After a "
            "successful import, local retrieval may run in the same turn only "
            "when the user explicitly requested analysis."
        ),
        "output_style": "paper_import_status",
    },
    {
        "name": "paper_search_skill",
        "description": "Search external scholarly metadata for papers relevant to the user's FYP.",
        "trigger_keywords": [
            "搜索论文",
            "查找论文",
            "检索论文",
            "找论文",
            "最新论文",
            "相关论文",
            "搜索文献",
            "查找文献",
            "search papers",
            "find papers",
            "latest papers",
            "recent papers",
            "literature search",
        ],
        "allowed_tools": ["search_papers"],
        "prompt_hint": (
            "Use the external metadata only as untrusted discovery evidence. "
            "Distinguish metadata/abstract claims from full-paper conclusions, "
            "include paper URLs, and state when an abstract is unavailable."
        ),
        "output_style": "paper_search_results",
    },
    {
        "name": "paper_analysis_skill",
        "description": "Analyze a paper and explain how it relates to the user's FYP.",
        "trigger_keywords": ["论文", "paper", "literature", "method", "方法", "abstract", "introduction"],
        "allowed_tools": [],
        "prompt_hint": (
            "When analyzing papers, focus on Problem, Method, Dataset, Result, "
            "Limitation, and Relevance to the user's reasoning distillation FYP."
        ),
        "output_style": "paper_analysis"
    },
    {
        "name": "experiment_design_skill",
        "description": "Help design experiments, baselines, and ablation studies.",
        "trigger_keywords": ["实验", "experiment", "baseline", "ablation", "对照组", "multi-rationale", "single-rationale"],
        "allowed_tools": [
            "generate_experiment_groups",
            "compare_datasets",
            "suggest_student_models"
        ],
        "prompt_hint": (
            "When designing experiments, clearly separate baseline, single-rationale, "
            "multi-rationale, and diverse multi-rationale groups. Mention variables, "
            "controls, datasets, and evaluation metrics."
        ),
        "output_style": "experiment_plan"
    },
    {
        "name": "teacher_student_model_skill",
        "description": "Suggest teacher model and student model choices for reasoning distillation.",
        "trigger_keywords": ["teacher", "student", "model", "模型", "蒸馏", "distillation", "distilbert", "t5", "qwen"],
        "allowed_tools": [
            "suggest_teacher_models",
            "suggest_student_models"
        ],
        "prompt_hint": (
            "When discussing models, distinguish teacher models from student models. "
            "Explain whether the task is generation or classification, and recommend "
            "compact student models suitable for reasoning distillation."
        ),
        "output_style": "model_recommendation"
    },
    {
        "name": "dataset_selection_skill",
        "description": "Help compare datasets for reasoning distillation experiments.",
        "trigger_keywords": ["dataset", "数据集", "gsm8k", "svamp", "strategyqa", "数据"],
        "allowed_tools": [
            "compare_datasets"
        ],
        "prompt_hint": (
            "When comparing datasets, explain task type, use case, strengths, weaknesses, "
            "and how each dataset should be used in the FYP experiment."
        ),
        "output_style": "dataset_comparison"
    },
    {
        "name": "email_drafting_skill",
        "description": "Help draft academic emails to the supervisor.",
        "trigger_keywords": ["邮件", "email", "老师", "supervisor", "professor", "导师"],
        "allowed_tools": [],
        "prompt_hint": (
            "When drafting emails, use polite academic English, be concise, and clearly "
            "state the user's question or request."
        ),
        "output_style": "email_draft"
    },
    {
        "name": "general_fyp_advice_skill",
        "description": "Provide general advice about the user's FYP project.",
        "trigger_keywords": [],
        "allowed_tools": [],
        "prompt_hint": (
            "Give practical, concrete FYP advice. If the user's question depends on "
            "previous conversation history, summarize the relevant previous context first."
        ),
        "output_style": "general_advice"
    },
    {
        "name": "file_analysis_skill",
        "description": "Read and analyze local txt, markdown, or PDF files for the user's FYP work.",
        "trigger_keywords": [
            "file",
            "文件",
            "读取",
            "read",
            "分析文件",
            ".md",
            ".txt",
            "markdown",
            "笔记",
            "note",
            ".pdf",
            "pdf"
        ],
        "allowed_tools": [
            "read_text_file",
            "read_pdf_file",
            "retrieve_from_file"
        ],
        "prompt_hint": (
            "When analyzing a file, first summarize the file content, then explain how it "
            "relates to the user's FYP. If the file contains an experiment plan, extract "
            "datasets, models, experimental groups, metrics, and next steps."
        ),
        "output_style": "file_analysis"
    },
]


def select_skill(user_input: str) -> dict:
    text = user_input.lower()

    # An explicit local file is stronger evidence than a generic keyword such
    # as "method" or "paper". This also keeps tool permissions aligned with
    # the later retrieval decision.
    if re.search(r"(?:files/)?[^\s/]+\.(?:md|txt|pdf)", text):
        for skill in SKILL_REGISTRY:
            if skill["name"] == "file_analysis_skill":
                return skill

    mcp_library_intent = re.search(
        r"(?:mcp[^\n]*(?:论文库|paper\s+library|fyp\s+library))"
        r"|(?:(?:本地论文库|paper\s+library|fyp\s+library)[^\n]*mcp)"
        r"|本地论文库",
        text,
        flags=re.IGNORECASE,
    )
    if mcp_library_intent:
        for skill in SKILL_REGISTRY:
            if skill["name"] == "mcp_library_skill":
                return skill

    paper_import_intent = re.search(
        r"(?:下载|保存|导入|查看|获取)[^\n]*(?:论文|paper|第\s*\d+\s*篇)"
        r"|(?:论文|paper)[^\n]*(?:详情|下载|保存|导入)"
        r"|(?:下载|保存|导入|查看|获取)第\s*\d+\s*篇"
        r"|\b(?:download|import)\s+(?:the\s+)?paper\b"
        r"|\bpaper\s+details\b",
        text,
        flags=re.IGNORECASE,
    )
    if paper_import_intent:
        for skill in SKILL_REGISTRY:
            if skill["name"] == "paper_import_skill":
                return skill

    paper_search_intent = re.search(
        r"(?:搜索|查找|检索|找)[^\n]*(?:论文|文献|papers?)"
        r"|(?:论文|文献|papers?)[^\n]*(?:搜索|查找|检索)"
        r"|\b(?:search|find|look up)\s+(?:latest\s+|recent\s+)?papers?\b"
        r"|\bliterature\s+search\b",
        text,
        flags=re.IGNORECASE,
    )
    if paper_search_intent:
        for skill in SKILL_REGISTRY:
            if skill["name"] == "paper_search_skill":
                return skill

    best_skill = None
    best_score = 0

    for skill in SKILL_REGISTRY:
        score = 0

        for keyword in skill["trigger_keywords"]:
            if keyword.lower() in text:
                score += 1

        if score > best_score:
            best_score = score
            best_skill = skill

    if best_skill is None:
        return get_default_skill()

    return best_skill


def get_default_skill() -> dict:
    for skill in SKILL_REGISTRY:
        if skill["name"] == "general_fyp_advice_skill":
            return skill

    raise ValueError("Default skill general_fyp_advice_skill not found.")
