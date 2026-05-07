from collections.abc import Generator
from typing import Any
import json

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from zep_cloud.client import Zep

class GraphCreateTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            api_key = self.runtime.credentials["zep_api_key"]
            base_url = self.runtime.credentials.get("zep_base_url") or None
            client = Zep(api_key=api_key, base_url=base_url)

            result = client.graph.create(
                graph_id=tool_parameters["graph_id"],
                name=tool_parameters.get("name"),
                description=tool_parameters.get("description"),
            )

            result_payload = result.model_dump(mode="json", exclude_none=True) if hasattr(result, "model_dump") else result
            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)
