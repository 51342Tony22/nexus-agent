# Persistent typed memory records for v1.4.1.

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


SCHEMA_VERSION = 1
PERSISTENT_MEMORY_TYPES = {"long_term", "episodic"}


class MemoryStoreError(RuntimeError):
    """Raised when persistent memory cannot be safely loaded or saved."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """Small JSON-backed store with explicit CRUD and no automatic writes."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._records = {}
        self._load()

    @staticmethod
    def _validate_memory_type(memory_type: str):
        if memory_type not in PERSISTENT_MEMORY_TYPES:
            allowed = ", ".join(sorted(PERSISTENT_MEMORY_TYPES))
            raise ValueError(
                f"Unsupported persistent memory type '{memory_type}'. "
                f"Allowed types: {allowed}."
            )

    @staticmethod
    def _validate_score(name: str, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0.")

    @staticmethod
    def _normalize_tags(tags: list[str] | None) -> list[str]:
        normalized = []
        seen = set()

        for tag in tags or []:
            clean_tag = str(tag).strip().lower()
            if not clean_tag or clean_tag in seen:
                continue
            seen.add(clean_tag)
            normalized.append(clean_tag)

        return normalized

    def _load(self):
        if not self.path.exists():
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MemoryStoreError(
                f"Failed to load memory store '{self.path}': {error}"
            ) from error

        if not isinstance(payload, dict):
            raise MemoryStoreError("Memory store root must be a JSON object.")

        if payload.get("schema_version") != SCHEMA_VERSION:
            raise MemoryStoreError(
                "Unsupported memory schema version: "
                f"{payload.get('schema_version')}"
            )

        memories = payload.get("memories")
        if not isinstance(memories, list):
            raise MemoryStoreError("Memory store 'memories' must be a list.")

        loaded_records = {}
        for record in memories:
            if not isinstance(record, dict) or not record.get("memory_id"):
                raise MemoryStoreError("Memory store contains an invalid record.")
            self._validate_memory_type(record.get("memory_type"))
            loaded_records[record["memory_id"]] = record

        self._records = loaded_records

    def _save(self):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "memories": list(self._records.values()),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")

        try:
            temporary_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(self.path)
        except OSError as error:
            raise MemoryStoreError(
                f"Failed to save memory store '{self.path}': {error}"
            ) from error

    def add_memory(
        self,
        memory_type: str,
        content: str,
        *,
        source: str,
        confidence: float = 1.0,
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        self._validate_memory_type(memory_type)
        self._validate_score("confidence", confidence)
        self._validate_score("importance", importance)

        clean_content = content.strip()
        clean_source = source.strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty.")
        if not clean_source:
            raise ValueError("Memory source cannot be empty.")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Memory metadata must be a dictionary.")

        timestamp = _utc_now()
        record = {
            "memory_id": f"mem_{uuid4().hex[:12]}",
            "memory_type": memory_type,
            "content": clean_content,
            "source": clean_source,
            "confidence": float(confidence),
            "importance": float(importance),
            "tags": self._normalize_tags(tags),
            "metadata": deepcopy(metadata or {}),
            "created_at": timestamp,
            "updated_at": timestamp,
            "last_used_at": None,
        }
        self._records[record["memory_id"]] = record
        self._save()
        return deepcopy(record)

    def get_memory(self, memory_id: str) -> dict | None:
        record = self._records.get(memory_id)
        return deepcopy(record) if record else None

    def list_memories(
        self,
        memory_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict]:
        if memory_type is not None:
            self._validate_memory_type(memory_type)

        required_tags = set(self._normalize_tags(tags))
        records = []
        for record in self._records.values():
            if memory_type and record["memory_type"] != memory_type:
                continue
            if required_tags and not required_tags.issubset(record["tags"]):
                continue
            records.append(deepcopy(record))

        return sorted(
            records,
            key=lambda record: record["updated_at"],
            reverse=True,
        )

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        if memory_id not in self._records:
            raise KeyError(f"Unknown memory id: {memory_id}")

        record = self._records[memory_id]
        if content is not None:
            clean_content = content.strip()
            if not clean_content:
                raise ValueError("Memory content cannot be empty.")
            record["content"] = clean_content
        if confidence is not None:
            self._validate_score("confidence", confidence)
            record["confidence"] = float(confidence)
        if importance is not None:
            self._validate_score("importance", importance)
            record["importance"] = float(importance)
        if tags is not None:
            record["tags"] = self._normalize_tags(tags)
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("Memory metadata must be a dictionary.")
            record["metadata"] = deepcopy(metadata)

        record["updated_at"] = _utc_now()
        self._save()
        return deepcopy(record)

    def mark_used(self, memory_id: str) -> dict:
        if memory_id not in self._records:
            raise KeyError(f"Unknown memory id: {memory_id}")
        self._records[memory_id]["last_used_at"] = _utc_now()
        self._save()
        return deepcopy(self._records[memory_id])

    def mark_many_used(self, memory_ids: list[str]) -> list[dict]:
        """Update selected memories with one atomic store write."""
        unique_ids = list(dict.fromkeys(memory_ids))
        unknown_ids = [
            memory_id
            for memory_id in unique_ids
            if memory_id not in self._records
        ]
        if unknown_ids:
            raise KeyError(f"Unknown memory ids: {', '.join(unknown_ids)}")
        if not unique_ids:
            return []

        timestamp = _utc_now()
        for memory_id in unique_ids:
            self._records[memory_id]["last_used_at"] = timestamp
        self._save()
        return [deepcopy(self._records[memory_id]) for memory_id in unique_ids]

    def delete_memory(self, memory_id: str) -> bool:
        if memory_id not in self._records:
            return False
        del self._records[memory_id]
        self._save()
        return True

    def count_by_type(self) -> dict:
        counts = {
            memory_type: 0
            for memory_type in sorted(PERSISTENT_MEMORY_TYPES)
        }
        for record in self._records.values():
            counts[record["memory_type"]] += 1
        counts["total"] = len(self._records)
        return counts
