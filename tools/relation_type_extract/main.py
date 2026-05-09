from __future__ import annotations

from collections.abc import Generator
from typing import Any
import json
import os
import sys

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.logger import get_logger

logger = get_logger('relation_type_extract')

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.entity_extractor import discover_relation_types_from_documents  # noqa: E402
from utils.entity_extractor import EntityExtractionToolCore  # noqa: E402


class RelationTypeExtractTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        try:
            logger.info('invoke params=%s', json.dumps(tool_parameters, ensure_ascii=False, default=str))
            text = str(tool_parameters.get("text") or "").strip()
            if not text:
                raise ValueError("text 参数不能为空")

            model_config = tool_parameters.get("model")
            llm_invoke = EntityExtractionToolCore.build_llm_invoke(self.session, model_config)
            discovered = discover_relation_types_from_documents(
                documents=[text],
                llm_invoke=llm_invoke,
            )
            relation_types = EntityExtractionToolCore.build_relation_type_defs_from_discovered(discovered)

            result_payload = {
                "relation_types": relation_types,
                "count": len(relation_types),
            }
            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_variable_message("relation_types", result_payload)
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            logger.exception('invoke failed')
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)


