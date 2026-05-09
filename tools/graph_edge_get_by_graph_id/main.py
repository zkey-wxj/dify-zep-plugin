from collections.abc import Generator
from typing import Any
import json

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.logger import get_logger
from utils.tool_params import normalize_optional_str, parse_optional_int

logger = get_logger('graph_edge_get_by_graph_id')
from utils.zep_entity_reader import ZepEntityReader

class GraphEdgeGetByGraphIdTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            logger.info('invoke params=%s', json.dumps(tool_parameters, ensure_ascii=False, default=str))
            api_key = self.runtime.credentials["zep_api_key"]
            base_url = self.runtime.credentials.get("zep_base_url") or None
            reader = ZepEntityReader(api_key=api_key, api_base_url=base_url)
            limit = parse_optional_int(tool_parameters.get("limit"), "limit", min_value=1)
            uuid_cursor = normalize_optional_str(tool_parameters.get("uuid_cursor"))
            if limit is None and uuid_cursor is None:
                result = reader.get_all_edges(tool_parameters["graph_id"])
            else:
                result = reader.get_edges_page(
                    graph_id=tool_parameters["graph_id"],
                    limit=limit,
                    uuid_cursor=uuid_cursor,
                )
            result_payload = result.model_dump(mode="json", exclude_none=True) if hasattr(result, "model_dump") else result
            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            logger.exception('invoke failed')
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)
    


