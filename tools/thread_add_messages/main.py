from collections.abc import Generator
from typing import Any
import json

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from zep_cloud.client import Zep

from zep_cloud import Message


def _parse_messages(messages_json: str) -> list[Message]:
    try:
        raw = json.loads(messages_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"messages_json must be valid JSON: {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError("messages_json must be a JSON array")

    messages: list[Message] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"messages_json[{index}] must be an object")
        content = item.get("content")
        role = item.get("role")
        if not content or not role:
            raise ValueError(f"messages_json[{index}] requires content and role")
        messages.append(
            Message(
                content=content,
                role=role,
                metadata=item.get("metadata"),
                name=item.get("name"),
            )
        )
    return messages

class ThreadAddMessagesTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            api_key = self.runtime.credentials["zep_api_key"]
            base_url = self.runtime.credentials.get("zep_base_url") or None
            client = Zep(api_key=api_key, base_url=base_url)

            result = client.thread.add_messages(
                tool_parameters["thread_id"],
                messages=_parse_messages(tool_parameters["messages_json"]),
                return_context=tool_parameters.get("return_context"),
            )

            result_payload = result.model_dump(mode="json", exclude_none=True) if hasattr(result, "model_dump") else result
            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)
