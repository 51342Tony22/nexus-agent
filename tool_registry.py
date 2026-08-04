# tool_registry.py
#根据skill选择可以调用哪个tools，根据输入决定调用哪个tool
#接下来执行并返回result

from time import perf_counter

from tool_contracts import (
    CompareDatasetsInput,
    MCPLibrarySearchInput,
    NoArgumentsInput,
    PaperSearchInput,
    ReadPdfFileInput,
    ReadTextFileInput,
    RetrieveFromFileInput,
    SemanticScholarPaperInput,
    StudentModelInput,
    ToolError,
    ToolSpec,
    error_from_tool_result,
)
from mcp_client_adapter import search_fyp_library_via_mcp
from tools import (
    compare_datasets,
    suggest_teacher_models,
    suggest_student_models,
    generate_experiment_groups,
    read_text_file,
    read_pdf_file,
    retrieve_from_file,
    search_papers,
    get_paper_details,
    import_paper_pdf,
)

# ToolSpec is now the single source of truth for every executable tool.
TOOL_SPECS = {
    "compare_datasets": ToolSpec(
        name="compare_datasets",
        description="Compare supported reasoning datasets for FYP experiments.",
        function=compare_datasets,
        input_schema=CompareDatasetsInput,
        risk_level="read_only",
        requires_confirmation=False,
    ),
    "suggest_teacher_models": ToolSpec(
        name="suggest_teacher_models",
        description="Return deterministic teacher-model suggestions.",
        function=suggest_teacher_models,
        input_schema=NoArgumentsInput,
        risk_level="read_only",
        requires_confirmation=False,
    ),
    "suggest_student_models": ToolSpec(
        name="suggest_student_models",
        description="Suggest student models for generation or classification.",
        function=suggest_student_models,
        input_schema=StudentModelInput,
        risk_level="read_only",
        requires_confirmation=False,
    ),
    "generate_experiment_groups": ToolSpec(
        name="generate_experiment_groups",
        description="Generate deterministic FYP experiment groups.",
        function=generate_experiment_groups,
        input_schema=NoArgumentsInput,
        risk_level="read_only",
        requires_confirmation=False,
    ),
    "read_text_file": ToolSpec(
        name="read_text_file",
        description="Read one local TXT or Markdown file inside project files.",
        function=read_text_file,
        input_schema=ReadTextFileInput,
        risk_level="read_only",
        requires_confirmation=False,
    ),
    "read_pdf_file": ToolSpec(
        name="read_pdf_file",
        description="Read bounded text from one local PDF inside project files.",
        function=read_pdf_file,
        input_schema=ReadPdfFileInput,
        risk_level="read_only",
        requires_confirmation=False,
    ),
    "retrieve_from_file": ToolSpec(
        name="retrieve_from_file",
        description="Retrieve relevant chunks from one local TXT, MD, or PDF file.",
        function=retrieve_from_file,
        input_schema=RetrieveFromFileInput,
        risk_level="read_only",
        requires_confirmation=False,
    ),
    "search_papers": ToolSpec(
        name="search_papers",
        description="Search Semantic Scholar paper metadata and abstracts.",
        function=search_papers,
        input_schema=PaperSearchInput,
        risk_level="read_only",
        requires_confirmation=False,
    ),
    "mcp_search_fyp_library": ToolSpec(
        name="mcp_search_fyp_library",
        description=(
            "Discover and call the read-only FYP Library MCP server to "
            "search its allowlisted local paper catalog."
        ),
        function=search_fyp_library_via_mcp,
        input_schema=MCPLibrarySearchInput,
        risk_level="read_only",
        requires_confirmation=False,
    ),
    "get_paper_details": ToolSpec(
        name="get_paper_details",
        description="Fetch metadata for one selected Semantic Scholar paper.",
        function=get_paper_details,
        input_schema=SemanticScholarPaperInput,
        risk_level="read_only",
        requires_confirmation=False,
    ),
    "import_paper_pdf": ToolSpec(
        name="import_paper_pdf",
        description="Save one explicitly selected open-access PDF to project files.",
        function=import_paper_pdf,
        input_schema=SemanticScholarPaperInput,
        risk_level="local_write",
        requires_confirmation=True,
    ),
}

# Compatibility view for existing imports. It is derived, never maintained as
# a second registry, so names and functions cannot silently drift apart.
TOOL_REGISTRY = {
    name: spec.function
    for name, spec in TOOL_SPECS.items()
}


def get_tool_spec(tool_name: str) -> ToolSpec | None:
    return TOOL_SPECS.get(tool_name)

def build_tool_call(tool_name: str, args: dict | None = None) -> dict:

    if args is None:
        args = {}
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool_spec = TOOL_SPECS[tool_name]
    validated_args = tool_spec.validate_arguments(args)

    return {
        "name": tool_name,
        "function": tool_spec.function,
        "args": validated_args,
        "spec": tool_spec,
    }

def run_tool(tool_call: dict | None) -> dict:
    if tool_call is None:
        return {
            "tool_used": None,
            "tool_args": None,
            "tool_result": None,
            "status": "skipped",
            "error": None,
            "latency_seconds": 0.0,
        }

    tool_function = tool_call["function"]
    tool_args = tool_call["args"]

    started_at = perf_counter()
    try:
        result = tool_function(**tool_args)
    except Exception as error:
        latency = perf_counter() - started_at
        tool_error = ToolError(
            code="tool_execution_error",
            message=f"{type(error).__name__}: {error}",
            retryable=False,
        )
        return {
            "tool_used": tool_call["name"],
            "tool_args": tool_args,
            "tool_result": None,
            "status": "error",
            "error": tool_error.model_dump(),
            "latency_seconds": latency,
        }

    latency = perf_counter() - started_at
    tool_error = error_from_tool_result(result)

    return {
        "tool_used": tool_call["name"],
        "tool_args": tool_args,
        "tool_result": result,
        "status": "error" if tool_error else "success",
        "error": tool_error.model_dump() if tool_error else None,
        "latency_seconds": latency,
    }


def extract_dataset_names(user_input: str) -> list[str]:
    text = user_input.lower()
    dataset_names = []

    if "gsm8k" in text:
        dataset_names.append("GSM8K")

    if "svamp" in text:
        dataset_names.append("SVAMP")

    if "strategyqa" in text or "strategy qa" in text:
        dataset_names.append("StrategyQA")

    if not dataset_names:
        dataset_names = ["GSM8K", "SVAMP", "StrategyQA"]

    return dataset_names
