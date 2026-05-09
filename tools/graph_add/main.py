from collections.abc import Generator
from typing import Any
import json

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.logger import get_logger
from utils.tool_params import (
    normalize_optional_str,
    parse_optional_json_object,
    validate_episode_metadata,
    validate_graph_or_user_id,
    validate_rfc3339,
)

logger = get_logger('graph_add')
from utils.zep_entity_reader import ZepEntityReader

class GraphAddTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            logger.info('invoke params=%s', json.dumps(tool_parameters, ensure_ascii=False, default=str))
            api_key = self.runtime.credentials["zep_api_key"]
            base_url = self.runtime.credentials.get("zep_base_url") or None
            reader = ZepEntityReader(api_key=api_key, api_base_url=base_url)
            user_id = normalize_optional_str(tool_parameters.get("user_id"))
            graph_id = normalize_optional_str(tool_parameters.get("graph_id"))
            created_at = normalize_optional_str(tool_parameters.get("created_at"))
            metadata = parse_optional_json_object(tool_parameters.get("metadata_json"), "metadata_json")
            validate_graph_or_user_id(graph_id=graph_id, user_id=user_id, operation_name="graph.add")
            validate_rfc3339(created_at, "created_at")
            validate_episode_metadata(metadata)

            result_payload = reader.add_graph_data(
                data=tool_parameters["data"],
                data_type=tool_parameters["data_type"],
                user_id=user_id,
                graph_id=graph_id,
                created_at=created_at,
                source_description=normalize_optional_str(tool_parameters.get("source_description")),
                metadata=metadata,
            )

            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            logger.exception('invoke failed')
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)


