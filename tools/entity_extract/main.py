from __future__ import annotations

from collections.abc import Generator
from typing import Any
import json
import os
import sys

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.logger import get_logger

logger = get_logger('entity_extract')

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.entity_extractor import (  # noqa: E402
    EntityExtractor,
    EntityExtractionToolCore,
    discover_relation_types_from_documents,
)


class EntityExtractTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        try:
            logger.info('invoke params=%s', json.dumps(tool_parameters, ensure_ascii=False, default=str))
            text = str(tool_parameters.get("text") or "").strip()
            if not text:
                raise ValueError("text 参数不能为空")

            llm_invoke = EntityExtractionToolCore.build_llm_invoke(self.session, tool_parameters.get("model"))

            entity_type_defs = EntityExtractionToolCore.parse_type_definitions(
                tool_parameters.get("entity_types"),
                preferred_keys=("entity_types", "relation_types"),
            )
            relation_type_defs = EntityExtractionToolCore.parse_type_definitions(
                tool_parameters.get("relation_types"),
                preferred_keys=("relation_types", "entity_types"),
            )

            if not relation_type_defs:
                discovered_relations = discover_relation_types_from_documents(
                    documents=[text],
                    llm_invoke=llm_invoke,
                )
                relation_type_defs = EntityExtractionToolCore.build_relation_type_defs_from_discovered(discovered_relations)

            if not entity_type_defs:
                discovered_entity_types = EntityExtractionToolCore.discover_entity_type_names(text=text, llm_invoke=llm_invoke)
                entity_type_defs = EntityExtractionToolCore.build_type_defs_from_names(discovered_entity_types)

            entity_type_names = EntityExtractionToolCore.extract_type_names(entity_type_defs)
            relation_type_names = EntityExtractionToolCore.extract_type_names(relation_type_defs)
            entity_type_defs = EntityExtractionToolCore.deduplicate_type_definitions_keep_order(
                entity_type_defs
            )
            relation_type_defs = EntityExtractionToolCore.deduplicate_type_definitions_keep_order(
                relation_type_defs
            )
            entity_type_names = EntityExtractionToolCore.deduplicate_keep_order(entity_type_names)
            relation_type_names = EntityExtractionToolCore.deduplicate_keep_order(relation_type_names)

            extractor = EntityExtractor(
                llm_invoke=llm_invoke,
                max_gleanings=2,
                entity_types=entity_type_names or None,
            )

            ontology = EntityExtractionToolCore.build_ontology(entity_type_defs, relation_type_defs)
            entity_columns = [
                ("name", "Name"),
                ("type", "Type"),
                ("description", "Description"),
            ]
            header_sent = False
            stream_entity_seen: set[str] = set()
            stream_forbidden_names = {
                str(name).strip().casefold()
                for name in (entity_type_names + relation_type_names)
                if str(name).strip()
            }

            extraction = None
            for progress in extractor.extract_with_progress(text=text, ontology=ontology):
                event = progress.get("event")
                if event == "chunk":
                    chunk_entities_raw = progress.get("chunk_entities") or []
                    chunk_entities_filtered = []
                    for entity in chunk_entities_raw:
                        if not isinstance(entity, dict):
                            continue
                        name = str(entity.get("name") or "").strip()
                        entity_type = str(entity.get("type") or "").strip()
                        if not name:
                            continue
                        if name.casefold() in stream_forbidden_names:
                            continue
                        dedup_key = f"{name.casefold()}|{entity_type.casefold()}"
                        if dedup_key in stream_entity_seen:
                            continue
                        stream_entity_seen.add(dedup_key)
                        chunk_entities_filtered.append(entity)
                    rows = EntityExtractionToolCore.build_markdown_table_rows(
                        rows=chunk_entities_filtered,
                        columns=entity_columns,
                        include_empty_row=False,
                    )
                    if rows:
                        if not header_sent:
                            header = EntityExtractionToolCore.build_markdown_table_header(
                                columns=entity_columns,
                                title="Entities",
                            )
                            for stream_msg in EntityExtractionToolCore.stream_markdown_variable(
                                tool=self,
                                variable_name="entities_markdown",
                                markdown=header,
                            ):
                                yield stream_msg
                            header_sent = True
                        for stream_msg in EntityExtractionToolCore.stream_markdown_variable(
                            tool=self,
                            variable_name="entities_markdown",
                            markdown=rows,
                        ):
                            yield stream_msg
                elif event == "final":
                    extraction = progress.get("result")

            if extraction is None:
                extraction = extractor.extract(text=text, ontology=ontology)

            entities = [entity.to_dict() for entity in extraction.entities]
            entities = EntityExtractionToolCore.deduplicate_entity_dicts_keep_order(entities)
            entities = EntityExtractionToolCore.filter_entities_exclude_type_names(
                entities=entities,
                entity_type_names=entity_type_names,
                relation_type_names=relation_type_names,
            )
            entities = EntityExtractionToolCore.deduplicate_entity_dicts_keep_order(entities)

            result_payload = {
                "entities_count": len(entities),
                "entities": entities,
                "entity_types": entity_type_defs,
                "relation_types": relation_type_defs,
                "entity_type_names": entity_type_names,
                "relation_type_names": relation_type_names,
                "topics": extraction.topics,
                "tokens_used": extraction.tokens_used,
            }

            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_variable_message("entities", result_payload)
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))

        except Exception as e:
            logger.exception('invoke failed')
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)



