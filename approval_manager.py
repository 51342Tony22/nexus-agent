"""Process-local, single-use approvals bound to exact validated tool calls."""

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


ApprovalStatus = Literal[
    "pending",
    "approved",
    "rejected",
    "executed",
    "failed",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tool_call_digest(tool_name: str, arguments: dict) -> str:
    """Create a stable audit fingerprint for one validated tool call."""
    canonical = json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(pattern=r"^approval-[0-9a-f]{12}$")
    tool_name: str = Field(min_length=1)
    arguments: dict
    argument_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk_level: Literal["local_write", "external_write"]
    reason: str = Field(min_length=1)
    status: ApprovalStatus = "pending"
    created_at: str
    decided_at: str | None = None
    completed_at: str | None = None
    execution_status: Literal["success", "error"] | None = None
    resume_user_input: str | None = None
    resume_output_mode: Literal["text", "paper_analysis"] = "text"
    resume_step_id: int | str | None = None
    resume_description: str | None = None


class ApprovalManager:
    """Keep bounded approval state for one Agent process."""

    def __init__(self, max_pending: int = 20):
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1.")
        self.max_pending = max_pending
        self._requests: dict[str, ApprovalRequest] = {}

    def create_request(
        self,
        *,
        tool_name: str,
        arguments: dict,
        risk_level: str,
        reason: str,
        resume_user_input: str | None = None,
        resume_output_mode: Literal["text", "paper_analysis"] = "text",
        resume_step_id: int | str | None = None,
        resume_description: str | None = None,
    ) -> ApprovalRequest:
        digest = tool_call_digest(tool_name, arguments)
        for request in self._requests.values():
            if (
                request.status == "pending"
                and request.tool_name == tool_name
                and request.argument_digest == digest
            ):
                return request

        pending_count = sum(
            request.status == "pending"
            for request in self._requests.values()
        )
        if pending_count >= self.max_pending:
            raise ValueError("The pending approval limit has been reached.")

        request = ApprovalRequest(
            request_id=f"approval-{uuid4().hex[:12]}",
            tool_name=tool_name,
            arguments=dict(arguments),
            argument_digest=digest,
            risk_level=risk_level,
            reason=reason,
            created_at=_utc_now(),
            resume_user_input=resume_user_input,
            resume_output_mode=resume_output_mode,
            resume_step_id=resume_step_id,
            resume_description=resume_description,
        )
        self._requests[request.request_id] = request
        return request

    def get(self, request_id: str) -> ApprovalRequest:
        request = self._requests.get(request_id)
        if request is None:
            raise KeyError(f"Unknown approval request: {request_id}")
        return request

    def list_requests(
        self,
        *,
        status: ApprovalStatus | None = None,
    ) -> list[ApprovalRequest]:
        return [
            request
            for request in self._requests.values()
            if status is None or request.status == status
        ]

    def authorize(self, request_id: str) -> ApprovalRequest:
        request = self.get(request_id)
        if request.status != "pending":
            raise ValueError(
                f"Approval request is already {request.status}: {request_id}"
            )
        approved = request.model_copy(update={
            "status": "approved",
            "decided_at": _utc_now(),
        })
        self._requests[request_id] = approved
        return approved

    def reject(self, request_id: str) -> ApprovalRequest:
        request = self.get(request_id)
        if request.status != "pending":
            raise ValueError(
                f"Approval request is already {request.status}: {request_id}"
            )
        rejected = request.model_copy(update={
            "status": "rejected",
            "decided_at": _utc_now(),
        })
        self._requests[request_id] = rejected
        return rejected

    def complete(
        self,
        request_id: str,
        *,
        execution_status: Literal["success", "error"],
    ) -> ApprovalRequest:
        request = self.get(request_id)
        if request.status != "approved":
            raise ValueError(
                "Only an approved, unused request can record execution."
            )
        completed = request.model_copy(update={
            "status": (
                "executed" if execution_status == "success" else "failed"
            ),
            "completed_at": _utc_now(),
            "execution_status": execution_status,
        })
        self._requests[request_id] = completed
        return completed
