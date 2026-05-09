import json
from datetime import datetime
from typing import Any


def normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_optional_json_object(value: Any, field_name: str) -> dict[str, Any] | None:
    text = normalize_optional_str(value)
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON: {exc}") from exc

    if payload is not None and not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return payload


def parse_optional_json_string_list(value: Any, field_name: str) -> list[str] | None:
    text = normalize_optional_str(value)
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError(f"{field_name} must be a JSON array")

    items: list[str] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{idx}] must be string")
        text_item = item.strip()
        if not text_item:
            raise ValueError(f"{field_name}[{idx}] must not be empty")
        items.append(text_item)
    return items


def parse_optional_int(value: Any, field_name: str, *, min_value: int | None = None, max_value: int | None = None) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc

    if min_value is not None and parsed < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}")
    return parsed


def parse_optional_float(value: Any, field_name: str, *, min_value: float | None = None, max_value: float | None = None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc

    if min_value is not None and parsed < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}")
    return parsed


def parse_optional_bool(value: Any, field_name: str) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{field_name} must be a boolean")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def validate_graph_or_user_id(graph_id: str | None, user_id: str | None, operation_name: str) -> None:
    if not graph_id and not user_id:
        raise ValueError(f"{operation_name} requires either graph_id or user_id")


def validate_rfc3339(value: str | None, field_name: str) -> None:
    if not value:
        return
    candidate = value.strip()
    if not candidate:
        return
    normalized = candidate.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be RFC3339 datetime string") from exc


def validate_episode_metadata(metadata: dict[str, Any] | None) -> None:
    if metadata is None:
        return
    if len(metadata) > 10:
        raise ValueError("metadata_json supports at most 10 keys")
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata_json keys must be non-empty strings")
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, str)):
            continue
        raise ValueError("metadata_json values must be string/number/boolean/null")
