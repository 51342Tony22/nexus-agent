"""Central, deterministic permission decisions for tool execution."""

import re

from pydantic import BaseModel, ConfigDict

from tool_contracts import ToolSpec


class PermissionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    reason_code: str
    reason: str
    risk_level: str
    requires_confirmation: bool
    confirmation_source: str | None = None


_IMPORT_INTENT = re.compile(
    r"(?:下载|保存|导入|获取)[^\n]*(?:论文|paper|第\s*\d+\s*篇)"
    r"|(?:论文|paper)[^\n]*(?:下载|保存|导入)"
    r"|(?:下载|保存|导入|获取)第\s*\d+\s*篇"
    r"|\b(?:download|import|save)\s+(?:the\s+)?paper\b",
    flags=re.IGNORECASE,
)
_NEGATED_IMPORT = re.compile(
    r"(?:不要|别|无需|不需要)\s*(?:下载|保存|导入|获取)"
    r"|\b(?:do\s+not|don't)\s+(?:download|import|save)\b",
    flags=re.IGNORECASE,
)


def check_tool_permission(
    tool_spec: ToolSpec | None,
    user_input: str | None,
) -> PermissionDecision:
    """Authorize one call independently from Planner-produced fields."""
    if tool_spec is None:
        return PermissionDecision(
            allowed=True,
            reason_code="legacy_tool_not_yet_migrated",
            reason="This legacy tool has not yet been assigned a ToolSpec.",
            risk_level="legacy_unknown",
            requires_confirmation=False,
        )

    if not tool_spec.requires_confirmation:
        return PermissionDecision(
            allowed=True,
            reason_code="confirmation_not_required",
            reason="The ToolSpec does not require explicit confirmation.",
            risk_level=tool_spec.risk_level,
            requires_confirmation=False,
        )

    text = user_input or ""
    if tool_spec.name == "import_paper_pdf":
        if _NEGATED_IMPORT.search(text):
            return PermissionDecision(
                allowed=False,
                reason_code="write_explicitly_negated",
                reason="The user explicitly said not to import or save the paper.",
                risk_level=tool_spec.risk_level,
                requires_confirmation=True,
            )
        if _IMPORT_INTENT.search(text):
            return PermissionDecision(
                allowed=True,
                reason_code="explicit_import_request",
                reason="The current user message explicitly requests a paper import.",
                risk_level=tool_spec.risk_level,
                requires_confirmation=True,
                confirmation_source="current_user_message",
            )

    return PermissionDecision(
        allowed=False,
        reason_code="confirmation_required",
        reason=(
            f"Tool '{tool_spec.name}' requires explicit confirmation in the "
            "current user message."
        ),
        risk_level=tool_spec.risk_level,
        requires_confirmation=True,
    )
