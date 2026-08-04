#tools.py
#这里放可以被agent调用的python函数
from pathlib import Path
from pypdf import PdfReader
from config import (
    DOCUMENT_CHUNK_OVERLAP,
    DOCUMENT_CHUNK_SIZE,
    DOCUMENT_INDEX_CACHE_MAX_ENTRIES,
    EMBEDDING_DEVICE,
    EMBEDDING_LOCAL_FILES_ONLY,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
)
from document_index import DocumentIndex
from document_index_cache import DocumentIndexCache
from embedding_store import EmbeddingStore
from paper_search import SemanticScholarClient


DOCUMENT_INDEX_CACHE = DocumentIndexCache(
    max_entries=DOCUMENT_INDEX_CACHE_MAX_ENTRIES
)
PAPER_SEARCH_CLIENT = SemanticScholarClient(
    api_key=SEMANTIC_SCHOLAR_API_KEY,
    timeout_seconds=SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
)


def search_papers(
    query: str,
    limit: int = 5,
    year: str | None = None,
) -> dict:
    """Search external paper metadata through the provider adapter."""
    return PAPER_SEARCH_CLIENT.search(
        query=query,
        limit=limit,
        year=year,
    )


def get_paper_details(paper_id: str) -> dict:
    return PAPER_SEARCH_CLIENT.get_paper(paper_id)


def import_paper_pdf(paper_id: str) -> dict:
    return PAPER_SEARCH_CLIENT.import_open_access_pdf(paper_id)


def _resolve_project_file(file_path: str) -> tuple[Path | None, str | None]:
    base_dir = (Path(__file__).resolve().parent / "files").resolve()
    target_path = (base_dir / file_path).resolve()

    try:
        target_path.relative_to(base_dir)
    except ValueError:
        return None, "Access denied. Only files inside the files/ directory can be read."

    if not target_path.exists() or not target_path.is_file():
        return None, f"File not found: {file_path}"

    return target_path, None

def compare_datasets(dataset_names: list[str]) -> dict:
    dataset_info = {
        "GSM8K": {
            "task_type": "Mathematical reasoning",
            "format": "Grade-school math word problems",
            "strength": "Good for testing chain-of-thought and numerical reasoning.",
            "weakness": "Mostly focused on math; may not test general commonsense reasoning.",
            "recommended_use": "Main dataset for reasoning distillation experiments."
        },
        "SVAMP": {
            "task_type": "Mathematical reasoning robustness",
            "format": "Variations of arithmetic word problems",
            "strength": "Good for testing robustness and generalization.",
            "weakness": "Smaller and often used more as an evaluation dataset.",
            "recommended_use": "Good as a generalization or transfer test dataset."
        },
        "StrategyQA": {
            "task_type": "Commonsense and multi-hop reasoning",
            "format": "Yes/No questions requiring implicit reasoning",
            "strength": "Good for testing non-mathematical reasoning.",
            "weakness": "Evaluation can be more ambiguous than math datasets.",
            "recommended_use": "Useful if the project wants to include commonsense reasoning."
        }
    }

    result = {}

    for name in dataset_names:
        normalized_name = name.upper()

        if normalized_name == "GSM8K":
            result["GSM8K"] = dataset_info["GSM8K"]
        elif normalized_name == "SVAMP":
            result["SVAMP"] = dataset_info["SVAMP"]
        elif normalized_name == "STRATEGYQA":
            result["StrategyQA"] = dataset_info["StrategyQA"]

    if not result:
        return {
            "error": "No supported dataset found.",
            "supported_datasets": ["GSM8K", "SVAMP", "StrategyQA"]
        }

    return result

def suggest_teacher_models() -> dict:

    return {
        "recommended_teacher_models": [
            {
                "name": "GPT-4o or GPT-4.1",
                "type": "API-based large model",
                "reason": "Strong reasoning ability and convenient for generating high-quality rationales."

            },
            {
                "name": "DeepSeek-R1 / DeepSeek reasoning model",
                "type": "reasoning-focused large model",
                "reason": "Good for math and chain-of-thought style reasoning tasks."
            },
            {
                "name": "Qwen large instruct model",
                "type": "open-source or API model",
                "reason": "Good multilingual ability and useful if the user wants Chinese/English support."
            }

        ],

        "recommendation": "For the first version, use an API-based large model as the teacher to avoid deployment complexity."
    }



def suggest_student_models(task_type: str = "generation") -> dict:
    if task_type == "classification":
        return {
            "recommended_models": [
                {
                    "name": "DistilBERT",
                    "type": "encoder-only",
                    "reason": "Good for classification tasks and the user already has experience with DistilBERT fine-tuning."
                },
                {
                    "name": "MobileBERT",
                    "type": "encoder-only",
                    "reason": "Very compact and suitable for lightweight classification experiments."
                }
            ],
            "warning": "Encoder-only models are not ideal for generating rationales."
        }

    return {
        "recommended_models": [
            {
                "name": "Flan-T5-small",
                "type": "encoder-decoder",
                "reason": "Small, easy to fine-tune, and suitable for text-to-text reasoning distillation."
            },
            {
                "name": "T5-small",
                "type": "encoder-decoder",
                "reason": "Classic compact seq2seq model; good for controlled experiments."
            },
            {
                "name": "Qwen2.5-0.5B/1.5B-Instruct",
                "type": "decoder-only",
                "reason": "Compact instruction-following model with stronger generation ability."
            }
        ],
        "warning": "For GSM8K and SVAMP, generation-capable student models are usually more suitable than encoder-only models."
    }


def generate_experiment_groups() -> dict:
    return {
        "experiment_groups": [
            {
                "name": "Baseline",
                "training_data": "Question-answer pairs without rationales.",
                "purpose": "Measure the student model's base performance without reasoning supervision."
            },
            {
                "name": "Single-rationale",
                "training_data": "Each sample has one teacher-generated rationale.",
                "purpose": "Test whether one reasoning trace improves student performance."
            },
            {
                "name": "Multi-rationale",
                "training_data": "Each sample has multiple teacher-generated rationales.",
                "purpose": "Test whether multiple reasoning traces improve robustness."
            },
            {
                "name": "Diverse multi-rationale",
                "training_data": "Each sample has multiple rationales generated with different prompts or strategies.",
                "purpose": "Test whether rationale diversity improves generalization."
            }
        ],
        "recommended_first_step": "Start with Baseline, Single-rationale, and Multi-rationale. Add Diverse multi-rationale after the pipeline is stable."
    }

def read_text_file(file_path: str) -> dict:
    target_path, path_error = _resolve_project_file(file_path)
    if path_error:
        return {"error": path_error}

    if target_path.suffix.lower() not in [".txt", ".md"]:
        return {
            "error": "Unsupported file type. Only .txt and .md files are supported in v1.0."
        }

    content = target_path.read_text(encoding="utf-8")

    return {
        "file_path": str(target_path),
        "file_type": target_path.suffix.lower(),
        "content": content,
        "character_count": len(content)
    }

def read_pdf_file(
    file_path: str,
    max_pages: int | None = 5,
    max_characters: int | None = 8000,
) -> dict:
    target_path, path_error = _resolve_project_file(file_path)
    if path_error:
        return {"error": path_error}

    if target_path.suffix.lower() != ".pdf":
        return {
            "error": "Unsupported file type. This tool only supports .pdf files."
        }

    try:
        reader = PdfReader(str(target_path))
        total_pages = len(reader.pages)

        pages_to_read = (
            total_pages
            if max_pages is None
            else min(total_pages, max_pages)
        )
        extracted_pages = []

        for page_index in range(pages_to_read):
            page = reader.pages[page_index]
            text = page.extract_text() or ""

            extracted_pages.append({
                "page_number": page_index + 1,
                "text": text
            })

        full_text = "\n\n".join(
            f"[Page {page['page_number']}]\n{page['text']}"
            for page in extracted_pages
        )

        if max_characters is not None and len(full_text) > max_characters:
            full_text = full_text[:max_characters] + "\n\n[Content truncated because the PDF text is too long.]"

        return {
            "file_path": str(target_path),
            "file_type": ".pdf",
            "total_pages": total_pages,
            "pages_read": pages_to_read,
            "content": full_text,
            "character_count": len(full_text)
        }

    except Exception as error:
        return {
            "error": f"Failed to read PDF file: {str(error)}"
        }

def retrieve_from_file(file_path: str, query: str, top_k: int = 3) -> dict:
    target_path, path_error = _resolve_project_file(file_path)
    if path_error:
        return {
            "error": path_error,
            "file_path": file_path,
        }

    lower_path = file_path.lower()
    if not lower_path.endswith((".pdf", ".txt", ".md")):
        return {
            "error": "Unsupported file type. Retrieval supports .pdf, .txt, and .md files.",
            "file_path": file_path,
        }

    index_signature = (
        DOCUMENT_CHUNK_SIZE,
        DOCUMENT_CHUNK_OVERLAP,
        EmbeddingStore.configured_model_source(),
        EMBEDDING_LOCAL_FILES_ONLY,
        EMBEDDING_DEVICE,
    )

    def build_indexed_document() -> dict:
        if lower_path.endswith(".pdf"):
            file_result = read_pdf_file(
                file_path=file_path,
                max_pages=None,
                max_characters=120000,
            )
        else:
            file_result = read_text_file(file_path=file_path)

        if "error" in file_result:
            raise RuntimeError(file_result["error"])

        return {
            "file_type": file_result.get("file_type"),
            "character_count": file_result.get("character_count"),
            "index": DocumentIndex(
                text=file_result.get("content", ""),
                chunk_size=DOCUMENT_CHUNK_SIZE,
                overlap=DOCUMENT_CHUNK_OVERLAP,
            ),
        }

    try:
        indexed_document, cache_debug = DOCUMENT_INDEX_CACHE.get_or_build(
            path=target_path,
            index_signature=index_signature,
            builder=build_indexed_document,
        )
    except Exception as error:
        return {
            "error": str(error),
            "file_path": file_path,
        }

    retrieval_result = indexed_document["index"].search(
        query=query,
        top_k=top_k,
    )
    retrieval_result["index_cache"] = cache_debug
    retrieval_result["index_build_seconds"] = indexed_document[
        "index"
    ].build_seconds

    return {
        "file_path": file_path,
        "file_type": indexed_document.get("file_type"),
        "character_count": indexed_document.get("character_count"),
        "retrieval": retrieval_result,
    }


def clear_document_index_cache():
    """Explicit reset for tests and long-running interactive sessions."""
    DOCUMENT_INDEX_CACHE.clear()
