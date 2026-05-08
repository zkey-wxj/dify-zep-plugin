from __future__ import annotations

from collections.abc import Generator
from typing import Any
import json
import os
import re
import sys

from dify_plugin import Tool
from dify_plugin.entities.model.message import SystemPromptMessage, UserPromptMessage
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.logger import get_logger

logger = get_logger('text_to_knowledge_graph')
from utils.zep_entity_reader import ZepEntityReader

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.entity_extractor import (  # noqa: E402
    EntityExtractor,
    discover_relation_types_from_documents,
)


class TextToKnowledgeGraphTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        try:
            logger.info('invoke params=%s', json.dumps(tool_parameters, ensure_ascii=False, default=str))
            text = str(tool_parameters.get("text") or "").strip()
            if not text:
                raise ValueError("text 参数不能为空")

            model_config = tool_parameters.get("model")
            if not isinstance(model_config, dict):
                raise ValueError("model 参数缺失或格式不正确，请在工具配置中选择模型")

            llm_invoke = self._build_llm_invoke(model_config)

            entity_types = self._parse_list_param(tool_parameters.get("entity_types"))
            relation_types = self._parse_list_param(tool_parameters.get("relation_types"))

            if not relation_types:
                discovered_relations = discover_relation_types_from_documents(
                    documents=[text],
                    llm_invoke=llm_invoke,
                )
                relation_types = [item.get("label", "").strip() for item in discovered_relations if item.get("label")]
                relation_types = self._deduplicate_keep_order(relation_types)

            if not entity_types:
                entity_types = self._discover_entity_types(text=text, llm_invoke=llm_invoke)

            extractor = EntityExtractor(
                llm_invoke=llm_invoke,
                max_gleanings=2,
                entity_types=entity_types or None,
            )

            ontology = self._build_ontology(entity_types, relation_types)
            extraction = extractor.extract(text=text, ontology=ontology)

            triples = self._build_triples_from_extraction(extraction)
            if not triples:
                result_payload = {
                    "status": "success",
                    "result": {
                        "graph_id": tool_parameters.get("graph_id"),
                        "user_id": tool_parameters.get("user_id"),
                        "triples_count": 0,
                        "triples": [],
                        "entity_types": entity_types,
                        "relation_types": relation_types,
                    },
                    "message": "未抽取到可写入的三元组",
                }
                yield self.create_json_message(result_payload)
                yield self.create_variable_message("knowledge_graph", result_payload["result"])
                yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
                return

            zep_api_key = self.runtime.credentials["zep_api_key"]
            zep_base_url = self.runtime.credentials.get("zep_base_url") or None
            reader = ZepEntityReader(api_key=zep_api_key, api_base_url=zep_base_url)

            zep_data = {
                "triples": triples,
                "source_text": text,
            }

            add_result_payload = reader.add_graph_data(
                data=json.dumps(zep_data, ensure_ascii=False),
                data_type="fact_triple",
                user_id=tool_parameters.get("user_id"),
                graph_id=tool_parameters.get("graph_id"),
                source_description="text_to_knowledge_graph",
            )

            result_payload = {
                "graph_id": tool_parameters.get("graph_id"),
                "user_id": tool_parameters.get("user_id"),
                "triples_count": len(triples),
                "triples": triples,
                "entity_types": entity_types,
                "relation_types": relation_types,
                "topics": extraction.topics,
                "tokens_used": extraction.tokens_used,
                "zep_add_result": add_result_payload,
            }

            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_variable_message("knowledge_graph", result_payload)
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))

        except Exception as e:
            logger.exception('invoke failed')
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

    def _discover_entity_types(self, text: str, llm_invoke) -> list[str]:
        prompt = f"""
从以下文本中识别可能的实体类型（例如人物、组织、地点、概念、事件、方法、物品、时间）。
请每行返回一个类型名称，不要返回其他内容。

文本：
{text[:2000]}
""".strip()

        output = llm_invoke(
            [
                {"role": "system", "content": "你是知识图谱实体类型识别助手。"},
                {"role": "user", "content": prompt},
            ],
            0.3,
            300,
        )

        lines = [line.strip("- •*\t ") for line in output.splitlines()]
        candidates = [line for line in lines if line and len(line) <= 30]
        return self._deduplicate_keep_order(candidates)

    @staticmethod
    def _build_ontology(entity_types: list[str], relation_types: list[str]) -> dict[str, Any]:
        return {
            "entity_types": [{"name": item, "description": item} for item in entity_types],
            "edge_types": [{"name": item, "description": item} for item in relation_types],
        }

    def _build_triples_from_extraction(self, extraction) -> list[dict[str, Any]]:
        entity_type_map: dict[str, str] = {}
        for entity in extraction.entities:
            key = entity.name.strip().casefold()
            if key and key not in entity_type_map:
                entity_type_map[key] = entity.entity_type

        triples = []
        seen: set[str] = set()
        for relation in extraction.relations:
            source = relation.source.strip()
            target = relation.target.strip()
            rel_type = relation.relation_type.strip() or "RELATED_TO"
            if not source or not target:
                continue

            key = f"{source.casefold()}|{rel_type.casefold()}|{target.casefold()}"
            if key in seen:
                continue
            seen.add(key)

            triple = {
                "source": source,
                "relation": rel_type,
                "target": target,
                "source_type": entity_type_map.get(source.casefold(), "Entity"),
                "target_type": entity_type_map.get(target.casefold(), "Entity"),
                "evidence": relation.description or "",
                "confidence": round(max(0.0, min(1.0, relation.strength / 10.0)), 3),
            }
            triples.append(triple)

        return triples

    @staticmethod
    def _parse_list_param(raw_value: Any) -> list[str]:
        if raw_value is None:
            return []

        if isinstance(raw_value, list):
            values = [str(item).strip() for item in raw_value]
            return [item for item in values if item]

        text = str(raw_value).strip()
        if not text:
            return []

        parts = re.split(r"[,，\n]+", text)
        return [part.strip() for part in parts if part.strip()]

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


