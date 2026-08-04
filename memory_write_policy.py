"""Explainable, consent-first write policy for persistent memory v1.4.3."""

import re
from difflib import SequenceMatcher
from uuid import uuid4


MAX_PENDING_SUGGESTIONS = 10

SENSITIVE_PATTERNS = [
    r"\bapi[ _-]?key\b",
    r"\baccess[ _-]?token\b",
    r"\brefresh[ _-]?token\b",
    r"\bpassword\b",
    r"\bsecret\b",
    r"\bsk-[a-z0-9_-]{8,}\b",
    r"密码",
    r"密钥",
    r"令牌",
    r"身份证",
    r"银行卡",
    r"信用卡",
    r"家庭住址",
    r"电话号码",
]

MEMORY_RULES = [
    {
        "category": "explicit_memory_request",
        "patterns": [
            r"(?:请|帮我)?记住",
            r"(?:please\s+)?remember\b",
        ],
        "tags": ["explicit"],
        "importance": 0.8,
        "reason": "The user explicitly asked the Agent to remember this.",
    },
    {
        "category": "response_preference",
        "patterns": [
            r"我(?:希望|喜欢|偏好|更喜欢|不喜欢).{0,30}(?:回答|解释|分析|代码|教学)",
            r"(?:回答|解释|分析)(?:时|的时候).{1,40}",
            r"请(?:以后|总是|一直).{0,40}",
            r"\bi\s+(?:prefer|like|want).{0,40}(?:answer|explain|analysis|code)",
            r"\bplease\s+(?:always|from now on)\b",
        ],
        "tags": ["preference"],
        "importance": 0.8,
        "reason": "The input describes a reusable response preference.",
    },
    {
        "category": "research_profile",
        "patterns": [
            r"我的?(?:研究方向|研究主题|fyp方向|fyp主题|课题|专业)(?:是|为|：|:)",
            r"\bmy\s+(?:research|fyp)\s+(?:topic|direction|field)\s+is\b",
        ],
        "tags": ["profile", "research"],
        "importance": 0.85,
        "reason": "The input describes a stable research profile.",
    },
    {
        "category": "long_term_goal",
        "patterns": [
            r"我的?(?:长期目标|学习目标|fyp目标)(?:是|为|：|:)",
            r"\bmy\s+(?:long[- ]term|learning|fyp)\s+goal\s+is\b",
        ],
        "tags": ["profile", "goal"],
        "importance": 0.75,
        "reason": "The input describes a reusable long-term goal.",
    },
    {
        "category": "development_environment",
        "patterns": [
            r"我的?(?:开发环境|操作系统|python版本)(?:是|为|：|:)",
            r"我(?:正在)?使用.{0,20}(?:python|macos|windows|linux)",
            r"\bmy\s+(?:development\s+)?environment\s+is\b",
        ],
        "tags": ["profile", "environment"],
        "importance": 0.65,
        "reason": "The input describes a stable development environment.",
    },
]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", value.lower())


def _is_duplicate(content: str, records: list[dict]) -> tuple[bool, str | None]:
    normalized = _normalize(content)
    if not normalized:
        return False, None

    for record in records:
        existing = _normalize(str(record.get("content", "")))
        if not existing:
            continue
        similarity = SequenceMatcher(None, normalized, existing).ratio()
        if normalized == existing or similarity >= 0.9:
            return True, record.get("memory_id") or record.get("suggestion_id")

    return False, None


def evaluate_memory_write(
    user_input: str,
    existing_records: list[dict] | None = None,
) -> dict:
    """Return a local suggestion decision; never writes persistent state."""
    content = user_input.strip()
    if not content:
        return {
            "status": "not_memorable",
            "reason_code": "empty_input",
            "reason": "Empty input cannot become memory.",
            "suggestion": None,
        }

    if len(content) > 500:
        return {
            "status": "not_memorable",
            "reason_code": "too_long",
            "reason": "Long task content should not be stored as one memory.",
            "suggestion": None,
        }

    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, content, flags=re.IGNORECASE):
            return {
                "status": "rejected",
                "reason_code": "sensitive_information",
                "reason": "Potentially sensitive information is never suggested for storage.",
                "suggestion": None,
            }

    matched_rule = None
    for rule in MEMORY_RULES:
        if any(
            re.search(pattern, content, flags=re.IGNORECASE)
            for pattern in rule["patterns"]
        ):
            matched_rule = rule
            break

    if matched_rule is None:
        return {
            "status": "not_memorable",
            "reason_code": "no_stable_memory_signal",
            "reason": "No stable profile, preference, goal, or explicit memory request was detected.",
            "suggestion": None,
        }

    duplicate, duplicate_of = _is_duplicate(
        content,
        existing_records or [],
    )
    if duplicate:
        return {
            "status": "duplicate",
            "reason_code": "duplicate_memory",
            "reason": "A substantially identical memory already exists.",
            "duplicate_of": duplicate_of,
            "suggestion": None,
        }

    suggestion = {
        "suggestion_id": f"sug_{uuid4().hex[:10]}",
        "memory_type": "long_term",
        "content": content,
        "source": "memory_write_policy",
        "confidence": 1.0,
        "importance": matched_rule["importance"],
        "tags": list(matched_rule["tags"]),
        "category": matched_rule["category"],
        "reason": matched_rule["reason"],
    }
    return {
        "status": "suggested",
        "reason_code": matched_rule["category"],
        "reason": matched_rule["reason"],
        "suggestion": suggestion,
    }
