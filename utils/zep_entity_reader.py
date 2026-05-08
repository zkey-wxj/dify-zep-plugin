"""
Zep实体读取与过滤服务
从Zep图谱中读取节点，筛选出符合预定义实体类型的节点

支持两种模式:
1. GRAPH_BACKEND=zep: 使用 Zep Cloud (zep_cloud.client.Zep)
2. GRAPH_BACKEND=zep_local: 使用本地 Neo4j + Qdrant (zep_adapter)
"""

import time
from typing import Dict, Any, List, Optional, Set, Callable, TypeVar
from dataclasses import dataclass, field

from utils.logger import get_logger
from zep_cloud.client import Zep

try:
    from zep_cloud import InternalServerError
except Exception:  # pragma: no cover - 兼容旧版 zep_cloud
    class InternalServerError(Exception):
        pass

logger = get_logger('zep_entity_reader')

# 用于泛型返回类型
T = TypeVar('T')
logger.info("使用 Zep Cloud for entity reader")

_DEFAULT_PAGE_SIZE = 100
_MAX_NODES = 2000
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 2.0


@dataclass
class EntityNode:
    """实体节点数据结构"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # 相关的边信息
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # 相关的其他节点信息
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }
    
    def get_entity_type(self) -> Optional[str]:
        """获取实体类型（排除默认的Entity标签）"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """过滤后的实体集合"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class ZepEntityReader:
    """
    Zep实体读取与过滤服务

    主要功能：
    1. 从Zep图谱读取所有节点
    2. 筛选出符合预定义实体类型的节点（Labels不只是Entity的节点）
    3. 获取每个实体的相关边和关联节点信息
    """

    def __init__(self, api_key: Optional[str] = None, api_base_url: Optional[str] = None):
        self.api_key = api_key
        self.api_base_url = api_base_url
        
        # 云端模式使用原始 Zep SDK
        if not self.api_key:
            raise ValueError("ZEP_API_KEY 未配置")
        self.client = Zep(api_key=self.api_key, base_url=self.api_base_url)
        self.default_page_size = _DEFAULT_PAGE_SIZE
        self.max_nodes = _MAX_NODES
        self.max_retries = _DEFAULT_MAX_RETRIES
        self.retry_delay = _DEFAULT_RETRY_DELAY

    def _call_with_retry(
        self, 
        func: Callable[[], T], 
        operation_name: str,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        initial_delay: float = _DEFAULT_RETRY_DELAY
    ) -> T:
        """
        带重试机制的Zep API调用 (云端模式)

        Args:
            func: 要执行的函数（无参数的lambda或callable）
            operation_name: 操作名称，用于日志
            max_retries: 最大重试次数（默认3次，即最多尝试3次）
            initial_delay: 初始延迟秒数

        Returns:
            API调用结果
        """
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} 第 {attempt + 1} 次尝试失败: {str(e)[:100]}, "
                        f"{delay:.1f}秒后重试..."
                    )
                    time.sleep(delay)
                    delay *= 2  # 指数退避
                else:
                    logger.error(f"Zep {operation_name} 在 {max_retries} 次尝试后仍失败: {str(e)}")

        raise last_exception

    def _fetch_page_with_retry(
        self,
        api_call: Callable[..., List[Any]],
        operation_name: str,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_delay: float = _DEFAULT_RETRY_DELAY,
        **kwargs: Any,
    ) -> List[Any]:
        last_exception: Exception | None = None
        delay = retry_delay

        for attempt in range(max_retries):
            try:
                return api_call(**kwargs)
            except (ConnectionError, TimeoutError, OSError, InternalServerError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} 第 {attempt + 1} 次尝试失败: {str(e)[:100]}, "
                        f"{delay:.1f}秒后重试..."
                    )
                    time.sleep(delay)
                    delay *= 2
            except Exception as e:
                # 非典型瞬态错误，交给统一重试逻辑兜底
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2

        if last_exception is None:
            return []
        raise last_exception

    def _fetch_all_by_cursor(
        self,
        api_call: Callable[..., List[Any]],
        graph_id: str,
        operation_name: str,
        *,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_items: Optional[int] = None,
    ) -> List[Any]:
        all_items: List[Any] = []
        cursor: Optional[str] = None
        page_num = 0

        while True:
            kwargs: Dict[str, Any] = {"graph_id": graph_id, "limit": page_size}
            if cursor is not None:
                kwargs["uuid_cursor"] = cursor

            page_num += 1
            batch = self._fetch_page_with_retry(
                api_call=api_call,
                operation_name=f"{operation_name} 第{page_num}页(graph={graph_id})",
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                **kwargs,
            )
            if not batch:
                break

            all_items.extend(batch)
            if max_items is not None and len(all_items) >= max_items:
                all_items = all_items[:max_items]
                break
            if len(batch) < page_size:
                break

            cursor = getattr(batch[-1], "uuid_", None) or getattr(batch[-1], "uuid", None)
            if cursor is None:
                break

        return all_items

    # ========== 本地模式代理方法 ==========

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """获取图谱的所有节点"""
       
        return self._get_all_nodes_cloud_impl(graph_id)

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
      
        return self._get_all_edges_cloud_impl(graph_id)

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """获取节点的边"""
       
        return self._get_node_edges_cloud_impl(node_uuid)

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """筛选实体"""
        
        return self._filter_defined_entities_cloud_impl(graph_id, defined_entity_types, enrich_with_edges)

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """获取实体及上下文"""
       
        return self._get_entity_with_context_cloud_impl(graph_id, entity_uuid)

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """按类型获取实体"""
        
        return self._get_entities_by_type_cloud_impl(graph_id, entity_type, enrich_with_edges)

    def create_graph(
        self,
        graph_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建图谱"""
        return self._create_graph_cloud_impl(graph_id, name, description)

    def add_graph_data(
        self,
        data: str,
        data_type: str,
        user_id: Optional[str] = None,
        graph_id: Optional[str] = None,
        source_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """向图谱添加数据"""
        return self._add_graph_data_cloud_impl(
            data=data,
            data_type=data_type,
            user_id=user_id,
            graph_id=graph_id,
            source_description=source_description,
            metadata=metadata,
        )

    def search_graph(
        self,
        query: str,
        user_id: Optional[str] = None,
        graph_id: Optional[str] = None,
        scope: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """图谱语义检索"""
        return self._search_graph_cloud_impl(query, user_id, graph_id, scope, limit)

    def delete_graph(self, graph_id: str) -> Dict[str, Any]:
        """删除图谱"""
        return self._delete_graph_cloud_impl(graph_id)

    def get_episode(self, uuid: str) -> Dict[str, Any]:
        """获取 episode"""
        return self._get_episode_cloud_impl(uuid)

    def get_observation(self, uuid: str) -> Dict[str, Any]:
        """获取 observation"""
        return self._get_observation_cloud_impl(uuid)

    def delete_observation(self, uuid: str) -> Dict[str, Any]:
        """删除 observation（当前 SDK 可能不支持）"""
        return self._delete_observation_cloud_impl(uuid)

    # ========== 云端模式实现 (_cloud_impl 后缀) ==========

    def _get_all_nodes_cloud_impl(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        获取图谱的所有节点（带重试机制）
        
        Args:
            graph_id: 图谱ID
            
        Returns:
            节点列表
        """
        logger.info(f"获取图谱 {graph_id} 的所有节点...")
        
        nodes = self._fetch_all_by_cursor(
            api_call=self.client.graph.node.get_by_graph_id,
            graph_id=graph_id,
            operation_name="获取节点",
            page_size=self.default_page_size,
            max_items=self.max_nodes,
        )
        
        nodes_data = []
        for node in nodes:
            nodes_data.append({
                "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                "name": node.name or "",
                "labels": node.labels or [],
                "summary": node.summary or "",
                "attributes": node.attributes or {},
            })
        
        logger.info(f"共获取 {len(nodes_data)} 个节点")
        return nodes_data

    def _model_to_dict(self, value: Any) -> Dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, dict):
            return value
        return {"value": value}
    
    def _get_all_edges_cloud_impl(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        获取图谱的所有边（带重试机制）
        
        Args:
            graph_id: 图谱ID
            
        Returns:
            边列表
        """
        logger.info(f"获取图谱 {graph_id} 的所有边...")
        
        edges = self._fetch_all_by_cursor(
            api_call=self.client.graph.edge.get_by_graph_id,
            graph_id=graph_id,
            operation_name="获取边",
            page_size=self.default_page_size,
            max_items=None,
        )
        
        edges_data = []
        for edge in edges:
            edges_data.append({
                "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                "name": edge.name or "",
                "fact": edge.fact or "",
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "attributes": edge.attributes or {},
            })
        
        logger.info(f"共获取 {len(edges_data)} 条边")
        return edges_data
    
    def _get_node_edges_cloud_impl(self, node_uuid: str) -> List[Dict[str, Any]]:
        """
        获取指定节点的所有相关边（带重试机制）
        
        Args:
            node_uuid: 节点UUID
            
        Returns:
            边列表
        """
        try:
            # 使用重试机制调用Zep API
            edges = self._call_with_retry(
                func=lambda: self.client.graph.node.get_entity_edges(node_uuid=node_uuid),
                operation_name=f"获取节点边(node={node_uuid[:8]}...)"
            )
            
            edges_data = []
            for edge in edges:
                edges_data.append({
                    "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                    "name": edge.name or "",
                    "fact": edge.fact or "",
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                    "attributes": edge.attributes or {},
                })
            
            return edges_data
        except Exception as e:
            logger.warning(f"获取节点 {node_uuid} 的边失败: {str(e)}")
            return []
    
    def _filter_defined_entities_cloud_impl(
        self, 
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """
        筛选出符合预定义实体类型的节点
        
        筛选逻辑：
        - 如果节点的Labels只有一个"Entity"，说明这个实体不符合我们预定义的类型，跳过
        - 如果节点的Labels包含除"Entity"和"Node"之外的标签，说明符合预定义类型，保留
        
        Args:
            graph_id: 图谱ID
            defined_entity_types: 预定义的实体类型列表（可选，如果提供则只保留这些类型）
            enrich_with_edges: 是否获取每个实体的相关边信息
            
        Returns:
            FilteredEntities: 过滤后的实体集合
        """
        logger.info(f"开始筛选图谱 {graph_id} 的实体...")
        
        # 获取所有节点
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)
        
        # 获取所有边（用于后续关联查找）
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        
        # 构建节点UUID到节点数据的映射
        node_map = {n["uuid"]: n for n in all_nodes}
        
        # 筛选符合条件的实体
        filtered_entities = []
        entity_types_found = set()
        
        for node in all_nodes:
            labels = node.get("labels", [])
            
            # 筛选逻辑：Labels必须包含除"Entity"和"Node"之外的标签
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]
            
            if not custom_labels:
                # 只有默认标签，跳过
                continue
            
            # 如果指定了预定义类型，检查是否匹配
            if defined_entity_types:
                matching_labels = [l for l in custom_labels if l in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]
            
            entity_types_found.add(entity_type)
            
            # 创建实体节点对象
            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )
            
            # 获取相关边和节点
            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()
                
                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])
                
                entity.related_edges = related_edges
                
                # 获取关联节点的基本信息
                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        related_node = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node["labels"],
                            "summary": related_node.get("summary", ""),
                        })
                
                entity.related_nodes = related_nodes
            
            filtered_entities.append(entity)
        
        logger.info(f"筛选完成: 总节点 {total_count}, 符合条件 {len(filtered_entities)}, "
                   f"实体类型: {entity_types_found}")
        
        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )
    
    def _get_entity_with_context_cloud_impl(
        self, 
        graph_id: str, 
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """
        获取单个实体及其完整上下文（边和关联节点，带重试机制）
        
        Args:
            graph_id: 图谱ID
            entity_uuid: 实体UUID
            
        Returns:
            EntityNode或None
        """
        try:
            # 使用重试机制获取节点
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=entity_uuid),
                operation_name=f"获取节点详情(uuid={entity_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            # 获取节点的边
            edges = self.get_node_edges(entity_uuid)
            
            # 获取所有节点用于关联查找
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}
            
            # 处理相关边和节点
            related_edges = []
            related_node_uuids = set()
            
            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])
            
            # 获取关联节点信息
            related_nodes = []
            for related_uuid in related_node_uuids:
                if related_uuid in node_map:
                    related_node = node_map[related_uuid]
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node["labels"],
                        "summary": related_node.get("summary", ""),
                    })
            
            return EntityNode(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {},
                related_edges=related_edges,
                related_nodes=related_nodes,
            )
            
        except Exception as e:
            logger.error(f"获取实体 {entity_uuid} 失败: {str(e)}")
            return None
    
    def _get_entities_by_type_cloud_impl(
        self, 
        graph_id: str, 
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """
        获取指定类型的所有实体
        
        Args:
            graph_id: 图谱ID
            entity_type: 实体类型（如 "Student", "PublicFigure" 等）
            enrich_with_edges: 是否获取相关边信息
            
        Returns:
            实体列表
        """
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities

    def _create_graph_cloud_impl(
        self,
        graph_id: str,
        name: Optional[str],
        description: Optional[str],
    ) -> Dict[str, Any]:
        result = self._call_with_retry(
            func=lambda: self.client.graph.create(
                graph_id=graph_id,
                name=name,
                description=description,
            ),
            operation_name=f"创建图谱(graph={graph_id})",
        )
        return self._model_to_dict(result)

    def _add_graph_data_cloud_impl(
        self,
        data: str,
        data_type: str,
        user_id: Optional[str],
        graph_id: Optional[str],
        source_description: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        result = self._call_with_retry(
            func=lambda: self.client.graph.add(
                data=data,
                type=data_type,
                user_id=user_id,
                graph_id=graph_id,
                source_description=source_description,
                metadata=metadata,
            ),
            operation_name=f"图谱添加(type={data_type}, graph={graph_id or '-'})",
        )
        return self._model_to_dict(result)

    def _search_graph_cloud_impl(
        self,
        query: str,
        user_id: Optional[str],
        graph_id: Optional[str],
        scope: Optional[str],
        limit: Optional[int],
    ) -> Dict[str, Any]:
        result = self._call_with_retry(
            func=lambda: self.client.graph.search(
                query=query,
                user_id=user_id,
                graph_id=graph_id,
                scope=scope,
                limit=limit,
            ),
            operation_name=f"图谱检索(graph={graph_id or '-'}, query={query[:20]}...)",
        )
        return self._model_to_dict(result)

    def _delete_graph_cloud_impl(self, graph_id: str) -> Dict[str, Any]:
        result = self._call_with_retry(
            func=lambda: self.client.graph.delete(graph_id),
            operation_name=f"删除图谱(graph={graph_id})",
        )
        return self._model_to_dict(result)

    def _get_episode_cloud_impl(self, uuid: str) -> Dict[str, Any]:
        result = self._call_with_retry(
            func=lambda: self.client.graph.episode.get(uuid),
            operation_name=f"获取episode(uuid={uuid[:8]}...)",
        )
        return self._model_to_dict(result)

    def _get_observation_cloud_impl(self, uuid: str) -> Dict[str, Any]:
        result = self._call_with_retry(
            func=lambda: self.client.graph.observation.get(uuid),
            operation_name=f"获取observation(uuid={uuid[:8]}...)",
        )
        return self._model_to_dict(result)

    def _delete_observation_cloud_impl(self, uuid: str) -> Dict[str, Any]:
        raise NotImplementedError("client.graph.observation.delete(uuid_) is not available in current zep-cloud SDK")


