from collections.abc import Generator
from typing import Any
import json

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from zep_cloud.client import Zep


def _parse_optional_json_object(value: str | None, field_name: str) -> dict[str, Any] | None:
    if value is None or not value.strip():
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON: {exc}") from exc

    if payload is not None and not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return payload

class UserAddTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            api_key = self.runtime.credentials["zep_api_key"]
            base_url = self.runtime.credentials.get("zep_base_url") or None
            client = Zep(api_key=api_key, base_url=base_url)

            result = client.user.add(
                user_id=tool_parameters["user_id"],
                email=tool_parameters.get("email"),
                first_name=tool_parameters.get("first_name"),
                last_name=tool_parameters.get("last_name"),
                metadata=_parse_optional_json_object(tool_parameters.get("metadata_json"), "metadata_json"),
            )

            result_payload = result.model_dump(mode="json", exclude_none=True) if hasattr(result, "model_dump") else result
            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)
