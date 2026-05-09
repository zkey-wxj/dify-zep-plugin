from collections.abc import Generator
from typing import Any
import json

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.logger import get_logger
from utils.tool_params import (
    normalize_optional_str,
    parse_optional_bool,
    parse_optional_float,
    parse_optional_int,
    parse_optional_json_object,
    parse_optional_json_string_list,
    validate_graph_or_user_id,
)

logger = get_logger('graph_search')
from utils.zep_entity_reader import ZepEntityReader


class GraphSearchTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            logger.info('invoke params=%s', json.dumps(tool_parameters, ensure_ascii=False, default=str))
            api_key = self.runtime.credentials["zep_api_key"]
            base_url = self.runtime.credentials.get("zep_base_url") or None
            reader = ZepEntityReader(api_key=api_key, api_base_url=base_url)
            user_id = normalize_optional_str(tool_parameters.get("user_id"))
            graph_id = normalize_optional_str(tool_parameters.get("graph_id"))
            reranker = normalize_optional_str(tool_parameters.get("reranker"))
            scope = normalize_optional_str(tool_parameters.get("scope"))
            limit = parse_optional_int(tool_parameters.get("limit"), "limit", min_value=1)
            max_characters = parse_optional_int(
                tool_parameters.get("max_characters"),
                "max_characters",
                min_value=1,
            )
            mmr_lambda = parse_optional_float(
                tool_parameters.get("mmr_lambda"),
                "mmr_lambda",
                min_value=0.0,
                max_value=1.0,
            )
            return_raw_results = parse_optional_bool(tool_parameters.get("return_raw_results"), "return_raw_results")
            center_node_uuid = normalize_optional_str(tool_parameters.get("center_node_uuid"))
            bfs_origin_node_uuids = parse_optional_json_string_list(
                tool_parameters.get("bfs_origin_node_uuids_json"),
                "bfs_origin_node_uuids_json",
            )
            search_filters = parse_optional_json_object(tool_parameters.get("search_filters_json"), "search_filters_json")
            validate_graph_or_user_id(graph_id=graph_id, user_id=user_id, operation_name="graph.search")
            if reranker == "mmr" and mmr_lambda is None:
                raise ValueError("mmr_lambda is required when reranker=mmr")
            if reranker == "node_distance" and not center_node_uuid:
                raise ValueError("center_node_uuid is required when reranker=node_distance")

            result_payload = reader.search_graph(
                query=tool_parameters["query"],
                user_id=user_id,
                graph_id=graph_id,
                scope=scope,
                limit=limit,
                max_characters=max_characters,
                mmr_lambda=mmr_lambda,
                reranker=reranker,
                return_raw_results=return_raw_results,
                center_node_uuid=center_node_uuid,
                bfs_origin_node_uuids=bfs_origin_node_uuids,
                search_filters=search_filters,
            )

            yield self.create_json_message({"status": "success", "result": result_payload})
            yield self.create_text_message(json.dumps(result_payload, ensure_ascii=False))
        except Exception as e:
            logger.exception('invoke failed')
            err = str(e)
            yield self.create_json_message({"status": "error", "error": err})
            yield self.create_text_message(err)


