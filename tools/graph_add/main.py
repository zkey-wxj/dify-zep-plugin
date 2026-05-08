from collections.abc import Generator
from typing import Any
import json

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.zep_entity_reader import ZepEntityReader


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


class GraphAddTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            api_key = self.runtime.credentials["zep_api_key"]
            base_url = self.runtime.credentials.get("zep_base_url") or None
            reader = ZepEntityReader(api_key=api_key, api_base_url=base_url)

            result_payload = reader.add_graph_data(
                data=tool_parameters["data"],
                data_type=tool_parameters["data_type"],
                user_id=tool_parameters.get("user_id"),
                graph_id=tool_parameters.get("graph_id"),
                source_description=tool_parameters.get("source_description"),
                metadata=_parse_optional_json_object(tool_parameters.get("metadata_json"), "metadata_json"),
            )

            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)
