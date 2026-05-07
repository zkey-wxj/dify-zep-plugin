from collections.abc import Generator
from typing import Any
import json

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from zep_cloud.client import Zep

class GraphObservationDeleteTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            api_key = self.runtime.credentials["zep_api_key"]
            base_url = self.runtime.credentials.get("zep_base_url") or None
            client = Zep(api_key=api_key, base_url=base_url)

            raise NotImplementedError("client.graph.observation.delete(uuid_) is not available in current zep-cloud SDK")

            result_payload = result.model_dump(mode="json", exclude_none=True) if hasattr(result, "model_dump") else result
            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)
