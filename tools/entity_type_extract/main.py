from __future__ import annotations

from collections.abc import Generator
from typing import Any
import json
import os
import sys

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.logger import get_logger

logger = get_logger('entity_type_extract')

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.entity_extractor import EntityExtractionToolCore  # noqa: E402


class EntityTypeExtractTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        try:
            logger.info('invoke params=%s', json.dumps(tool_parameters, ensure_ascii=False, default=str))
            text = str(tool_parameters.get("text") or "").strip()
            if not text:
                raise ValueError("text 参数不能为空")

            model_config = tool_parameters.get("model")
            llm_invoke = EntityExtractionToolCore.build_llm_invoke(self.session, model_config)
            entity_type_names = EntityExtractionToolCore.discover_entity_type_names(text=text, llm_invoke=llm_invoke)
            entity_types = EntityExtractionToolCore.build_type_defs_from_names(entity_type_names)

            result_payload = {
                "entity_types": entity_types,
                "count": len(entity_types),
            }
            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_variable_message("entity_types", result_payload)
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            logger.exception('invoke failed')
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)


