import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from agent import FYPResearchAgent
from context_builder import build_context
from memory import get_FYP_memory
from paper_search import SemanticScholarClient
from planner import create_plan, extract_semantic_scholar_paper_id
from prompts import build_prompt_bundle
from skills import select_skill
from tool_registry import build_tool_call, run_tool


PAPER_ID = "a" * 40
PDF_BYTES = b"%PDF-1.4\n% minimal test PDF\n%%EOF\n"


def detail_payload(
    *,
    pdf_url: str | None = "https://arxiv.org/pdf/1234.5678",
    title: str = "A / Dangerous: Paper Title",
) -> bytes:
    return json.dumps({
        "paperId": PAPER_ID,
        "title": title,
        "abstract": "Paper abstract.",
        "year": 2024,
        "publicationDate": "2024-01-01",
        "venue": "ACL",
        "citationCount": 12,
        "referenceCount": 20,
        "url": f"https://www.semanticscholar.org/paper/{PAPER_ID}",
        "openAccessPdf": ({"url": pdf_url} if pdf_url else None),
        "authors": [{"name": "Researcher"}],
        "externalIds": {"ArXiv": "1234.5678"},
    }).encode("utf-8")


class PaperDetailTests(unittest.TestCase):
    def test_get_paper_validates_id_before_network(self):
        calls = []
        client = SemanticScholarClient(
            transport=lambda request, timeout: calls.append(request)
        )

        result = client.get_paper("not-a-semantic-scholar-id")

        self.assertEqual(result["error"]["code"], "invalid_paper_id")
        self.assertEqual(calls, [])

    def test_get_paper_requests_selected_fields_and_normalizes_record(self):
        captured = {}

        def transport(request, timeout):
            captured["request"] = request
            return 200, detail_payload()

        client = SemanticScholarClient(
            api_key="provider-secret",
            transport=transport,
        )
        result = client.get_paper(PAPER_ID)

        request = captured["request"]
        parameters = parse_qs(urlparse(request.full_url).query)
        self.assertIn(PAPER_ID, request.full_url)
        self.assertIn("externalIds", parameters["fields"][0])
        self.assertEqual(result["paper"]["paper_id"], PAPER_ID)
        self.assertEqual(result["paper"]["external_ids"]["ArXiv"], "1234.5678")
        self.assertIsNone(result["error"])

    def test_get_paper_handles_not_found(self):
        client = SemanticScholarClient(
            transport=lambda request, timeout: (404, b"not found")
        )

        result = client.get_paper(PAPER_ID)

        self.assertEqual(result["error"]["code"], "paper_not_found")


class PaperImportTests(unittest.TestCase):
    def test_import_saves_verified_pdf_without_forwarding_provider_key(self):
        captured = {}

        def download(request, timeout):
            captured["request"] = request
            return 200, PDF_BYTES

        with tempfile.TemporaryDirectory() as directory:
            client = SemanticScholarClient(
                api_key="provider-secret",
                transport=lambda request, timeout: (200, detail_payload()),
                download_transport=download,
                files_directory=directory,
            )
            result = client.import_open_access_pdf(PAPER_ID)
            target = Path(directory) / result["file_path"]

            pdf_headers = dict(captured["request"].header_items())
            self.assertTrue(result["imported"])
            self.assertFalse(result["already_existed"])
            self.assertEqual(result["bytes_written"], len(PDF_BYTES))
            self.assertTrue(target.exists())
            self.assertEqual(target.read_bytes(), PDF_BYTES)
            self.assertNotIn("/", result["file_path"])
            self.assertNotIn("X-api-key", pdf_headers)

    def test_import_is_idempotent_and_does_not_download_twice(self):
        downloads = []

        def download(request, timeout):
            downloads.append(request.full_url)
            return 200, PDF_BYTES

        with tempfile.TemporaryDirectory() as directory:
            client = SemanticScholarClient(
                transport=lambda request, timeout: (200, detail_payload()),
                download_transport=download,
                files_directory=directory,
            )

            first = client.import_open_access_pdf(PAPER_ID)
            second = client.import_open_access_pdf(PAPER_ID)

            self.assertFalse(first["already_existed"])
            self.assertTrue(second["already_existed"])
            self.assertEqual(first["file_path"], second["file_path"])
            self.assertEqual(len(downloads), 1)

    def test_import_rejects_missing_unsafe_or_invalid_pdf(self):
        cases = [
            (
                detail_payload(pdf_url=None),
                lambda request, timeout: (200, PDF_BYTES),
                "no_open_access_pdf",
            ),
            (
                detail_payload(pdf_url="https://127.0.0.1/private.pdf"),
                lambda request, timeout: (200, PDF_BYTES),
                "unsafe_pdf_url",
            ),
            (
                detail_payload(),
                lambda request, timeout: (200, b"<html>not a pdf</html>"),
                "invalid_pdf",
            ),
            (
                detail_payload(),
                lambda request, timeout: (
                    200,
                    PDF_BYTES,
                    "https://127.0.0.1/redirected.pdf",
                ),
                "unsafe_pdf_redirect",
            ),
        ]

        for payload, downloader, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with tempfile.TemporaryDirectory() as directory:
                    client = SemanticScholarClient(
                        transport=lambda request, timeout, p=payload: (200, p),
                        download_transport=downloader,
                        files_directory=directory,
                    )
                    result = client.import_open_access_pdf(PAPER_ID)

                    self.assertFalse(result["imported"])
                    self.assertEqual(result["error"]["code"], expected_code)
                    self.assertEqual(list(Path(directory).glob("*.pdf")), [])

    def test_import_does_not_trust_or_overwrite_invalid_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            client = SemanticScholarClient(
                transport=lambda request, timeout: (200, detail_payload()),
                download_transport=lambda request, timeout: (200, PDF_BYTES),
                files_directory=directory,
            )
            filename = client._build_pdf_filename(
                "A / Dangerous: Paper Title",
                PAPER_ID,
            )
            target = Path(directory) / filename
            target.write_bytes(b"not a pdf")

            result = client.import_open_access_pdf(PAPER_ID)

            self.assertFalse(result["imported"])
            self.assertEqual(result["error"]["code"], "invalid_existing_file")
            self.assertEqual(target.read_bytes(), b"not a pdf")


class PaperImportPlanningTests(unittest.TestCase):
    def test_download_and_details_intents_choose_different_tools(self):
        download_input = f"下载论文 {PAPER_ID}"
        detail_input = f"查看论文详情 {PAPER_ID}"

        download_skill = select_skill(download_input)
        detail_skill = select_skill(detail_input)
        download_plan = create_plan(download_skill, download_input)
        detail_plan = create_plan(detail_skill, detail_input)

        self.assertEqual(download_skill["name"], "paper_import_skill")
        self.assertEqual(download_plan[0]["tool_name"], "import_paper_pdf")
        self.assertEqual(detail_plan[0]["tool_name"], "get_paper_details")
        self.assertEqual(download_plan[0]["tool_args"]["paper_id"], PAPER_ID)

    def test_missing_paper_id_produces_no_tool_call(self):
        user_input = "下载这篇论文"
        skill = select_skill(user_input)
        plan = create_plan(skill, user_input)

        self.assertEqual(skill["name"], "paper_import_skill")
        self.assertIsNone(plan[0]["tool_name"])
        self.assertIn("paper_id", plan[0]["description"])

    def test_negated_download_request_never_writes(self):
        user_input = f"不要下载论文 {PAPER_ID}，只告诉我你是否支持"
        skill = select_skill(user_input)
        plan = create_plan(skill, user_input)

        self.assertEqual(skill["name"], "paper_import_skill")
        self.assertIsNone(plan[0]["tool_name"])
        self.assertIn("negated", plan[0]["description"])

    def test_paper_id_extractor_rejects_partial_hex_strings(self):
        self.assertEqual(
            extract_semantic_scholar_paper_id(f"paper id: {PAPER_ID}"),
            PAPER_ID,
        )
        self.assertIsNone(extract_semantic_scholar_paper_id("a" * 39))
        self.assertIsNone(extract_semantic_scholar_paper_id("a" * 41))

    def test_registry_exposes_import_tool(self):
        expected = {
            "provider": "Semantic Scholar",
            "paper_id": PAPER_ID,
            "imported": True,
            "file_path": "paper.pdf",
            "error": None,
        }
        with patch("tools.PAPER_SEARCH_CLIENT") as client:
            client.import_open_access_pdf.return_value = expected
            result = run_tool(build_tool_call(
                "import_paper_pdf",
                {"paper_id": PAPER_ID},
            ))

        self.assertEqual(result["tool_used"], "import_paper_pdf")
        self.assertEqual(result["tool_result"], expected)


class PaperImportContextTests(unittest.TestCase):
    def test_import_result_is_external_data_and_reports_local_filename(self):
        user_input = f"下载论文 {PAPER_ID}"
        skill = select_skill(user_input)
        plan = create_plan(skill, user_input)
        observation = {
            "step_id": 1,
            "description": "Import paper.",
            "tool_used": "import_paper_pdf",
            "tool_args": {"paper_id": PAPER_ID},
            "tool_result": {
                "provider": "Semantic Scholar",
                "paper_id": PAPER_ID,
                "paper": {
                    "title": "Imported Paper",
                    "url": "https://example.org/paper",
                },
                "imported": True,
                "already_existed": False,
                "file_path": "Imported_Paper_aaaaaaaa.pdf",
                "bytes_written": 1234,
                "error": None,
                "attribution": "Paper metadata provided by Semantic Scholar.",
            },
        }
        context = build_context(
            user_input=user_input,
            memory=get_FYP_memory(),
            skill=skill,
            execution_plan=plan,
            tool_observations=[observation],
        )
        bundle = build_prompt_bundle(context)

        self.assertEqual(context["evidence"], [])
        self.assertEqual(
            context["external_tool_data"][0]["tool"],
            "import_paper_pdf",
        )
        self.assertIn("Imported_Paper_aaaaaaaa.pdf", bundle["user_prompt"])

    def test_agent_executes_import_tool_with_mocked_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = FYPResearchAgent(
                memory_store_path=str(Path(directory) / "memories.json")
            )
            captured = {}
            def fake_llm(context):
                captured["context"] = context
                return {
                    "content": "论文已经导入。",
                    "latency": 0.0,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "context_stats": {},
                }

            agent.call_llm = fake_llm
            with patch("tools.PAPER_SEARCH_CLIENT") as client:
                client.import_open_access_pdf.return_value = {
                    "provider": "Semantic Scholar",
                    "paper_id": PAPER_ID,
                    "paper": {"title": "Imported Paper"},
                    "imported": True,
                    "already_existed": False,
                    "file_path": "Imported_Paper_aaaaaaaa.pdf",
                    "bytes_written": 1234,
                    "error": None,
                }
                result = agent.run(f"下载论文 {PAPER_ID}")

            self.assertEqual(result["debug"]["tools_used"], ["import_paper_pdf"])
            self.assertEqual(
                result["debug"]["paper_imports"][0]["file_path"],
                "Imported_Paper_aaaaaaaa.pdf",
            )
            self.assertEqual(
                captured["context"]["external_tool_data"][0]["imported"],
                True,
            )


if __name__ == "__main__":
    unittest.main()
