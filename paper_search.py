"""Small Semantic Scholar Academic Graph client for paper discovery."""

import json
import ipaddress
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


SEARCH_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
SEARCH_FIELDS = (
    "paperId,title,abstract,year,authors,venue,citationCount,"
    "referenceCount,url,openAccessPdf,publicationDate"
)
DETAIL_FIELDS = SEARCH_FIELDS + ",externalIds"
MAX_RESPONSE_BYTES = 2_000_000
MAX_ABSTRACT_CHARACTERS = 1500
MAX_PDF_BYTES = 25_000_000
SEMANTIC_SCHOLAR_PAPER_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


class SemanticScholarClient:
    """Provider adapter whose transport can be replaced in unit tests."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
        transport=None,
        download_transport=None,
        files_directory: str | Path | None = None,
    ):
        self.api_key = api_key.strip() if api_key else None
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.transport = transport or self._default_transport
        self.download_transport = (
            download_transport or self._default_download_transport
        )
        self.files_directory = Path(
            files_directory
            or Path(__file__).resolve().parent / "files"
        ).expanduser().resolve()

    @staticmethod
    def _default_transport(request: Request, timeout: float) -> tuple[int, bytes]:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            return response.status, body

    @staticmethod
    def _default_download_transport(
        request: Request,
        timeout: float,
    ) -> tuple[int, bytes, str]:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_PDF_BYTES + 1)
            return response.status, body, response.geturl()

    @staticmethod
    def _error_result(query: str, code: str, message: str) -> dict:
        return {
            "provider": "Semantic Scholar",
            "query": query,
            "result_count": 0,
            "papers": [],
            "error": {
                "code": code,
                "message": message,
            },
        }

    @staticmethod
    def _paper_error_result(
        paper_id: str,
        code: str,
        message: str,
    ) -> dict:
        return {
            "provider": "Semantic Scholar",
            "paper_id": paper_id,
            "paper": None,
            "error": {"code": code, "message": message},
        }

    @staticmethod
    def _provider_error(status_code: int) -> tuple[str, str]:
        code = {
            400: "bad_request",
            401: "authentication_failed",
            403: "access_forbidden",
            404: "paper_not_found",
            429: "rate_limited",
        }.get(status_code, "provider_http_error")
        message = (
            "Semantic Scholar rate limit reached. Retry later or set "
            "SEMANTIC_SCHOLAR_API_KEY in .env."
            if status_code == 429
            else f"Semantic Scholar returned HTTP {status_code}."
        )
        return code, message

    def _request_headers(self, accept: str = "application/json") -> dict:
        return {
            "Accept": accept,
            "User-Agent": "FYP-Research-Mini-Agent/1.5.2",
            **({"x-api-key": self.api_key} if self.api_key else {}),
        }

    @staticmethod
    def _validate_year(year: str | None) -> str | None:
        if year is None:
            return None
        clean_year = str(year).strip()
        if not re.fullmatch(r"\d{4}(?:-\d{4}|-)?", clean_year):
            raise ValueError(
                "year must be YYYY, YYYY-YYYY, or YYYY-."
            )
        return clean_year

    @staticmethod
    def _normalize_paper(paper: dict) -> dict:
        open_access_pdf = paper.get("openAccessPdf")
        pdf_url = (
            open_access_pdf.get("url")
            if isinstance(open_access_pdf, dict)
            else None
        )
        abstract = paper.get("abstract")
        if isinstance(abstract, str) and len(abstract) > MAX_ABSTRACT_CHARACTERS:
            abstract = abstract[:MAX_ABSTRACT_CHARACTERS] + "…"

        return {
            "paper_id": paper.get("paperId"),
            "title": paper.get("title"),
            "authors": [
                author.get("name")
                for author in paper.get("authors", [])
                if isinstance(author, dict) and author.get("name")
            ],
            "year": paper.get("year"),
            "publication_date": paper.get("publicationDate"),
            "venue": paper.get("venue"),
            "citation_count": paper.get("citationCount"),
            "reference_count": paper.get("referenceCount"),
            "abstract": abstract,
            "url": paper.get("url"),
            "open_access_pdf_url": pdf_url,
        }

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        year: str | None = None,
    ) -> dict:
        clean_query = query.strip()
        if len(clean_query) < 2:
            return self._error_result(
                clean_query,
                "invalid_query",
                "Paper search query must contain at least two characters.",
            )
        if len(clean_query) > 300:
            return self._error_result(
                clean_query,
                "invalid_query",
                "Paper search query cannot exceed 300 characters.",
            )
        if not 1 <= limit <= 10:
            return self._error_result(
                clean_query,
                "invalid_limit",
                "Paper search limit must be between 1 and 10.",
            )

        try:
            clean_year = self._validate_year(year)
        except ValueError as error:
            return self._error_result(clean_query, "invalid_year", str(error))

        parameters = {
            "query": clean_query,
            "limit": str(limit),
            "fields": SEARCH_FIELDS,
        }
        if clean_year:
            parameters["year"] = clean_year
        request = Request(
            f"{SEARCH_ENDPOINT}?{urlencode(parameters)}",
            headers=self._request_headers(),
            method="GET",
        )

        try:
            status, body = self.transport(request, self.timeout_seconds)
        except HTTPError as error:
            status_code = error.code
            error.close()
            code = {
                400: "bad_request",
                401: "authentication_failed",
                403: "access_forbidden",
                429: "rate_limited",
            }.get(status_code, "provider_http_error")
            message = (
                "Semantic Scholar rate limit reached. Retry later or set "
                "SEMANTIC_SCHOLAR_API_KEY in .env."
                if status_code == 429
                else f"Semantic Scholar returned HTTP {status_code}."
            )
            return self._error_result(
                clean_query,
                code,
                message,
            )
        except (TimeoutError, URLError) as error:
            return self._error_result(
                clean_query,
                "network_error",
                f"Semantic Scholar could not be reached: {error}",
            )
        except OSError as error:
            return self._error_result(
                clean_query,
                "network_error",
                f"Paper search failed: {error}",
            )

        if status != 200:
            code = "rate_limited" if status == 429 else "provider_http_error"
            message = (
                "Semantic Scholar rate limit reached. Retry later or set "
                "SEMANTIC_SCHOLAR_API_KEY in .env."
                if status == 429
                else f"Semantic Scholar returned HTTP {status}."
            )
            return self._error_result(
                clean_query,
                code,
                message,
            )
        if len(body) > MAX_RESPONSE_BYTES:
            return self._error_result(
                clean_query,
                "response_too_large",
                "Semantic Scholar response exceeded the local safety limit.",
            )

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._error_result(
                clean_query,
                "invalid_response",
                "Semantic Scholar returned invalid JSON.",
            )

        raw_papers = payload.get("data", []) if isinstance(payload, dict) else []
        if not isinstance(raw_papers, list):
            return self._error_result(
                clean_query,
                "invalid_response",
                "Semantic Scholar response did not contain a paper list.",
            )

        papers = [
            self._normalize_paper(paper)
            for paper in raw_papers
            if isinstance(paper, dict)
        ]
        return {
            "provider": "Semantic Scholar",
            "query": clean_query,
            "year_filter": clean_year,
            "total_estimate": payload.get("total"),
            "result_count": len(papers),
            "papers": papers,
            "error": None,
            "attribution": "Paper metadata provided by Semantic Scholar.",
        }

    def get_paper(self, paper_id: str) -> dict:
        """Fetch details for one Semantic Scholar paper ID."""
        clean_id = paper_id.strip()
        if not SEMANTIC_SCHOLAR_PAPER_ID_PATTERN.fullmatch(clean_id):
            return self._paper_error_result(
                clean_id,
                "invalid_paper_id",
                "paper_id must be the 40-character hexadecimal Semantic Scholar ID.",
            )

        request = Request(
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"{quote(clean_id, safe='')}?{urlencode({'fields': DETAIL_FIELDS})}",
            headers=self._request_headers(),
            method="GET",
        )
        try:
            status, body = self.transport(request, self.timeout_seconds)
        except HTTPError as error:
            status_code = error.code
            error.close()
            code, message = self._provider_error(status_code)
            return self._paper_error_result(clean_id, code, message)
        except (TimeoutError, URLError, OSError) as error:
            return self._paper_error_result(
                clean_id,
                "network_error",
                f"Semantic Scholar could not be reached: {error}",
            )

        if status != 200:
            code, message = self._provider_error(status)
            return self._paper_error_result(clean_id, code, message)
        if len(body) > MAX_RESPONSE_BYTES:
            return self._paper_error_result(
                clean_id,
                "response_too_large",
                "Semantic Scholar response exceeded the local safety limit.",
            )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._paper_error_result(
                clean_id,
                "invalid_response",
                "Semantic Scholar returned invalid JSON.",
            )
        if not isinstance(payload, dict) or not payload.get("paperId"):
            return self._paper_error_result(
                clean_id,
                "invalid_response",
                "Semantic Scholar returned an invalid paper record.",
            )

        paper = self._normalize_paper(payload)
        paper["external_ids"] = payload.get("externalIds", {})
        return {
            "provider": "Semantic Scholar",
            "paper_id": clean_id,
            "paper": paper,
            "error": None,
            "attribution": "Paper metadata provided by Semantic Scholar.",
        }

    @staticmethod
    def _safe_pdf_url(url: str) -> tuple[bool, str | None]:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            return False, "Open-access PDF URL must use public HTTPS."
        hostname = parsed.hostname.lower()
        if (
            hostname == "localhost"
            or hostname.endswith(".localhost")
            or hostname.endswith(".local")
        ):
            return False, "Private or local PDF hosts are not allowed."
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            return False, "Private or local PDF hosts are not allowed."
        return True, None

    @staticmethod
    def _build_pdf_filename(title: str | None, paper_id: str) -> str:
        base = re.sub(
            r"[^\w.-]+",
            "_",
            (title or "paper").strip(),
            flags=re.UNICODE,
        ).strip("._")
        base = base[:70] or "paper"
        return f"{base}_{paper_id[:8]}.pdf"

    def import_open_access_pdf(self, paper_id: str) -> dict:
        """Download an explicitly selected open-access PDF into files/."""
        details = self.get_paper(paper_id)
        if details.get("error"):
            return {
                **details,
                "imported": False,
                "file_path": None,
            }

        paper = details["paper"]
        pdf_url = paper.get("open_access_pdf_url")
        if not pdf_url:
            return {
                **details,
                "imported": False,
                "file_path": None,
                "error": {
                    "code": "no_open_access_pdf",
                    "message": "Semantic Scholar has no open-access PDF URL for this paper.",
                },
            }

        safe_url, url_error = self._safe_pdf_url(pdf_url)
        if not safe_url:
            return {
                **details,
                "imported": False,
                "file_path": None,
                "error": {"code": "unsafe_pdf_url", "message": url_error},
            }

        filename = self._build_pdf_filename(paper.get("title"), paper_id)
        self.files_directory.mkdir(parents=True, exist_ok=True)
        target = (self.files_directory / filename).resolve()
        try:
            target.relative_to(self.files_directory)
        except ValueError:
            return {
                **details,
                "imported": False,
                "file_path": None,
                "error": {
                    "code": "unsafe_file_path",
                    "message": "Generated PDF path escaped the project files directory.",
                },
            }

        # The deterministic filename makes import idempotent: a repeated
        # request does not re-download or overwrite a previously imported PDF.
        if target.exists():
            try:
                existing_is_pdf = target.read_bytes()[:5] == b"%PDF-"
            except OSError as error:
                existing_is_pdf = False
                existing_error = str(error)
            else:
                existing_error = None
            if not existing_is_pdf:
                return {
                    **details,
                    "imported": False,
                    "already_existed": True,
                    "file_path": filename,
                    "error": {
                        "code": "invalid_existing_file",
                        "message": (
                            "Existing destination is not a valid PDF and was not overwritten."
                            + (f" Read error: {existing_error}" if existing_error else "")
                        ),
                    },
                }
            return {
                **details,
                "imported": True,
                "already_existed": True,
                "file_path": filename,
                "bytes_written": target.stat().st_size,
            }

        request = Request(
            pdf_url,
            # Never forward the Semantic Scholar API key to a third-party PDF
            # host. Provider credentials are scoped to provider requests only.
            headers={
                "Accept": "application/pdf",
                "User-Agent": "FYP-Research-Mini-Agent/1.5.2",
            },
            method="GET",
        )
        try:
            download_result = self.download_transport(
                request,
                self.timeout_seconds,
            )
        except HTTPError as error:
            status_code = error.code
            error.close()
            return {
                **details,
                "imported": False,
                "file_path": None,
                "error": {
                    "code": "pdf_http_error",
                    "message": f"PDF host returned HTTP {status_code}.",
                },
            }
        except (TimeoutError, URLError, OSError) as error:
            return {
                **details,
                "imported": False,
                "file_path": None,
                "error": {
                    "code": "pdf_network_error",
                    "message": f"Open-access PDF could not be downloaded: {error}",
                },
            }

        if len(download_result) == 3:
            status, body, final_url = download_result
        else:
            status, body = download_result
            final_url = pdf_url
        safe_final_url, final_url_error = self._safe_pdf_url(final_url)
        if not safe_final_url:
            return {
                **details,
                "imported": False,
                "file_path": None,
                "error": {
                    "code": "unsafe_pdf_redirect",
                    "message": final_url_error,
                },
            }

        if status != 200:
            return {
                **details,
                "imported": False,
                "file_path": None,
                "error": {
                    "code": "pdf_http_error",
                    "message": f"PDF host returned HTTP {status}.",
                },
            }
        if len(body) > MAX_PDF_BYTES:
            return {
                **details,
                "imported": False,
                "file_path": None,
                "error": {
                    "code": "pdf_too_large",
                    "message": "PDF exceeded the 25 MB local safety limit.",
                },
            }
        if not body.startswith(b"%PDF-"):
            return {
                **details,
                "imported": False,
                "file_path": None,
                "error": {
                    "code": "invalid_pdf",
                    "message": "Downloaded content is not a valid PDF file.",
                },
            }

        temporary = target.with_suffix(".pdf.part")
        try:
            temporary.write_bytes(body)
            temporary.replace(target)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            return {
                **details,
                "imported": False,
                "file_path": None,
                "error": {
                    "code": "file_write_error",
                    "message": f"PDF could not be saved: {error}",
                },
            }

        return {
            **details,
            "imported": True,
            "already_existed": False,
            "file_path": filename,
            "bytes_written": len(body),
        }
