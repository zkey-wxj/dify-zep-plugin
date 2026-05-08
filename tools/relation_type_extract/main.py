from __future__ import annotations

from collections.abc import Generator
from typing import Any
import json
import os
import sys

from dify_plugin import Tool
from dify_plugin.entities.model.message import SystemPromptMessage, UserPromptMessage
from dify_plugin.entities.tool import ToolInvokeMessage

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.entity_extractor import discover_relation_types_from_documents  # noqa: E402


class RelationTypeExtractTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        try:
            text = str(tool_parameters.get("text") or "").strip()
            if not text:
                raise ValueError("text 参数不能为空")

            model_config = tool_parameters.get("model")
            if not isinstance(model_config, dict):
                raise ValueError("model 参数缺失或格式不正确，请在工具配置中选择模型")

            llm_invoke = self._build_llm_invoke(model_config)
            discovered = discover_relation_types_from_documents(
                documents=[text],
                llm_invoke=llm_invoke,
            )
            relation_types = [item.get("label", "").strip() for item in discovered if item.get("label")]
            relation_types = self._deduplicate_keep_order(relation_types)

            result_payload = {
                "relation_types": relation_types,
                "count": len(relation_types),
            }
            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_variable_message("relation_types", result_payload)
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)

    def _build_llm_invoke(self, model_config: dict[str, Any]):
        def _invoke(messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
            prompt_messages = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "system":
                    prompt_messages.append(SystemPromptMessage(content=content))
                else:
                    prompt_messages.append(UserPromptMessage(content=content))

            result = self.session.model.llm.invoke(
                model_config=model_config,
                prompt_messages=prompt_messages,
                stream=False,
            )
            content = result.message.content
            return content if isinstance(content, str) else ""

        return _invoke

    @staticmethod
    def _deduplicate_keep_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = value.strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(normalized)
        return result
