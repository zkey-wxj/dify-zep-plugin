from collections.abc import Generator
from typing import Any
import json

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.logger import get_logger

logger = get_logger('graph_node_get')
from utils.zep_entity_reader import ZepEntityReader

class GraphNodeGetTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            logger.info('invoke params=%s', json.dumps(tool_parameters, ensure_ascii=False, default=str))
            api_key = self.runtime.credentials["zep_api_key"]
            base_url = self.runtime.credentials.get("zep_base_url") or None
            graph_id = str(tool_parameters.get("graph_id") or "").strip()
            if not graph_id:
                raise ValueError("graph_id 参数不能为空（使用 zep_entity_reader 读取节点详情需要 graph_id）")

            reader = ZepEntityReader(api_key=api_key, api_base_url=base_url)
            entity = reader.get_entity_with_context(graph_id=graph_id, entity_uuid=tool_parameters["uuid"])
            result_payload = entity.to_dict() if entity else None
            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            logger.exception('invoke failed')
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)


