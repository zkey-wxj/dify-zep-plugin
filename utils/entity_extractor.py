"""
LLM 实体抽取服务 - 基于 GraphRAG 方案

参考 RAGFlow/Microsoft GraphRAG 实现:
- 多轮抽取 (Gleaning) 最大化实体数量
- 关系强度评分
- 实体解析去重
- 根据文本长度采用不同策略
"""

import json
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

try:
    from utils.logger import get_logger
except Exception:
    from logger import get_logger

logger = get_logger('entity_extractor')

LlmInvokeFn = Callable[[List[Dict[str, str]], float, int], str]


def _sanitize_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    清洗属性值，确保所有值都是 Neo4j 支持的原始类型或数组

    Neo4j 不支持嵌套的 dict/map，需要转换为 JSON 字符串
    """
    sanitized = {}
    for key, value in attributes.items():
        if value is None:
            sanitized[key] = None
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            # 递归清洗列表中的每个元素
            sanitized[key] = [_sanitize_list_item(item) for item in value]
        elif isinstance(value, dict):
            # 将嵌套字典转换为 JSON 字符串
            sanitized[key] = json.dumps(value, ensure_ascii=False)
        else:
            # 其他类型转换为字符串（包括 tuple, set, 自定义对象等）
            sanitized[key] = str(value)
    return sanitized


def _sanitize_list_item(item: Any) -> Any:
    """递归清洗列表中的单个值"""
    if item is None:
        return None
    elif isinstance(item, (str, int, float, bool)):
        return item
    elif isinstance(item, list):
        return [_sanitize_list_item(i) for i in item]
    elif isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False)
    else:
        return str(item)


def _convert_neo4j_record(value: Any) -> Any:
    """
    将 Neo4j 返回的记录转换为纯 Python 类型

    Neo4j 驱动可能返回特殊的 Map 对象，需要转换为纯 Python dict
    递归处理嵌套的 dict 和 list
    """
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, dict):
        # 转换为纯 Python dict，并递归处理嵌套值
        return {k: _convert_neo4j_record(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_convert_neo4j_record(item) for item in value]
    else:
        # Neo4j Map 对象或其他类型，转换为字符串或尝试遍历
        try:
            # 尝试像 dict 一样遍历（处理 Neo4j Map 类型）
            if hasattr(value, 'items'):
                return {k: _convert_neo4j_record(v) for k, v in value.items()}
            elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                return [_convert_neo4j_record(item) for item in value]
            else:
                return str(value)
        except Exception:
            return str(value)


# ========== 分隔符定义 ==========
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


# ========== 关系类型标准化词库 ==========
RELATION_TYPE_NORMALIZATION = {
    # 归属/所属关系
    "属于": "BELONGS_TO",
    "隶属于": "BELONGS_TO",
    "位于": "LOCATED_AT",
    "在": "LOCATED_AT",
    "源于": "ORIGINATED_FROM",
    "来自": "FROM",

    # 身份/角色关系
    "是": "IS",
    "担任": "HOLDS_POSITION",
    "任职于": "EMPLOYED_BY",
    "工作于": "EMPLOYED_BY",
    "曾是": "WAS",

    # 社交/互动关系
    "合作": "COLLABORATES_WITH",
    "共同": "COLLABORATES_WITH",
    "师承": "MENTORED_BY",
    "指导": "MENTORS",
    "学生": "STUDENT_OF",
    "老师": "TEACHER_OF",
    "认识": "KNOWS",
    "朋友": "FRIEND_OF",

    # 创作/生产关系
    "创作": "CREATED",
    "发表": "PUBLISHED",
    "提出": "PROPOSED",
    "发明": "INVENTED",
    "编写": "WROTE",
    "著": "AUTHORED",
    "主演": "STARRED_IN",

    # 内容/组成关系
    "包含": "CONTAINS",
    "包括": "INCLUDES",
    "部分": "PART_OF",
    "阐述": "DESCRIBES",
    "解释": "EXPLAINS",
    "讨论": "DISCUSSES",

    # 因果/影响关系
    "导致": "CAUSES",
    "影响": "INFLUENCES",
    "基于": "BASED_ON",
    "依赖": "DEPENDS_ON",
    "促进": "PROMOTES",
    "阻碍": "HINDERS",

    # 时间/顺序关系
    "早于": "BEFORE",
    "晚于": "AFTER",
    "发生于": "OCCURRED_AT",
    "持续至": "LASTED_UNTIL",

    # 学术/研究关系
    "研究": "RESEARCHES",
    "领域": "FIELD_OF",
    "专长": "EXPERT_IN",
    "引用": "CITES",
    "参考": "REFERENCES",

    # 事件参与
    "参与": "PARTICIPATED_IN",
    "组织": "ORGANIZED",
    "主持": "HOSTED",
    "出席": "ATTENDED",

    # 通用关系（fallback）
    "相关": "RELATED_TO",
    "关联": "ASSOCIATED_WITH",
    "有关": "RELATED_TO",
}

# 反向关系映射（用于自动推断反向关系）
RELATION_INVERSE = {
    "BELONGS_TO": "CONTAINS",
    "EMPLOYED_BY": "EMPLOYS",
    "MENTORED_BY": "MENTORS",
    "STUDENT_OF": "TEACHER_OF",
    "CAUSES": "CAUSED_BY",
    "INFLUENCES": "INFLUENCED_BY",
    "BASED_ON": "BASIS_FOR",
    "PART_OF": "HAS_PART",
    "BEFORE": "AFTER",
    "MENTORS": "MENTORED_BY",
}


# ========== 关系类型双语映射 ==========
# 英文关系类型代码 → 中文显示标签
RELATION_TYPE_LABELS = {
    # 归属/来源类
    "ORIGINATED_FROM": "源于",
    "BELONGS_TO": "属于",
    "LOCATED_AT": "位于",
    "FROM": "来自",

    # 身份/角色类
    "IS": "是",
    "WAS": "曾是",
    "BECAME": "成为",
    "EMPLOYED_BY": "任职于",
    "HOLDS_POSITION": "担任",

    # 创作/生产类
    "CREATED": "创作",
    "AUTHORED": "著",
    "INVENTED": "发明",
    "PROPOSED": "提出",
    "PUBLISHED": "发表",
    "STARRED_IN": "主演",
    "WROTE": "编写",

    # 组成/包含类
    "CONTAINS": "包含",
    "INCLUDES": "包括",
    "PART_OF": "组成部分",
    "HAS_PART": "包含部分",
    "DESCRIBES": "阐述",
    "EXPLAINS": "解释",
    "DISCUSSES": "讨论",

    # 社交/互动类
    "COLLABORATES_WITH": "合作",
    "MENTORS": "指导",
    "MENTORED_BY": "师承",
    "STUDENT_OF": "学习",
    "TEACHER_OF": "教导",
    "KNOWS": "认识",
    "FRIEND_OF": "朋友",

    # 因果/影响类
    "CAUSES": "导致",
    "CAUSED_BY": "由...导致",
    "INFLUENCES": "影响",
    "INFLUENCED_BY": "受...影响",
    "BASED_ON": "基于",
    "BASIS_FOR": "是...基础",
    "DEPENDS_ON": "依赖",
    "PROMOTES": "促进",
    "HINDERS": "阻碍",

    # 学术/研究类
    "RESEARCHES": "研究",
    "FIELD_OF": "领域",
    "EXPERT_IN": "专长",
    "CITES": "引用",
    "REFERENCES": "参考",

    # 参与类
    "PARTICIPATED_IN": "参与",
    "ORGANIZED": "组织",
    "HOSTED": "主持",
    "ATTENDED": "出席",

    # 时间类
    "BEFORE": "早于",
    "AFTER": "晚于",
    "OCCURRED_AT": "发生于",

    # 通用关系
    "RELATED_TO": "相关",
    "ASSOCIATED_WITH": "关联",
}


def get_relation_label(relation_code: str, lang: str = "zh") -> str:
    """
    获取关系类型的本地化标签

    Args:
        relation_code: 关系类型英文代码（如 PART_OF）
        lang: 语言代码 ("zh" 中文, "en" 英文)

    Returns:
        对应语言的标签，如果未找到则返回原代码
    """
    if not relation_code:
        return ""

    if lang == "zh":
        # 返回中文标签
        return RELATION_TYPE_LABELS.get(relation_code.upper(), relation_code)
    else:
        # 英文直接返回代码
        return relation_code.upper()


def discover_relation_types_from_documents(
    documents: List[str],
    llm_invoke: Optional[LlmInvokeFn] = None,
    max_types: int = 30
) -> List[Dict[str, str]]:
    """
    从文档中自动发现关系类型

    分析文档内容，提取出现的关系表达，生成标准化的关系类型列表

    Args:
        documents: 文档文本列表
        llm_invoke: LLM 调用函数，签名为 (messages, temperature, max_tokens) -> str
        max_types: 最大返回类型数量

    Returns:
        发现的关系类型列表，格式：[{"code": "PART_OF", "label": "组成部分"}, ...]
    """
    if not documents:
        return []

    if llm_invoke is None:
        raise ValueError("llm_invoke 未配置")

    # 合并文档样本进行快速分析
    sample_text = "\n\n".join(documents[:5])  # 取前5个文档作为样本
    if len(sample_text) > 10000:
        sample_text = sample_text[:10000]  # 限制长度

    # 直接让 LLM 同时返回中文标签和英文代码
    discovery_prompt = f"""分析以下文本，提取其中描述实体之间关系的词语，并为每个关系生成对应的英文代码。

【分析要求】
1. 找出所有表示实体间关系的中文词语
2. 为每个关系生成一个简洁的英文代码（全大写，用下划线连接）
3. 英文代码应该语义清晰，例如：治疗->TREATS, 包含->CONTAINS, 基于->BASED_ON
4. 避免通用词（如"相关"、"有关"）

【文本样本】
{sample_text}

【输出格式】
每行格式：中文关系词 | 英文代码
示例：
治疗 | TREATS
包含 | CONTAINS
基于 | BASED_ON

请开始分析：
"""

    try:
        response_text = llm_invoke(
            [
                {
                    "role": "system",
                    "content": "你是一个专业的知识图谱分析师，擅长从文本中发现实体关系模式。"
                },
                {
                    "role": "user",
                    "content": discovery_prompt
                }
            ],
            0.3,
            800,
        )

        # 解析提取的关系词对
        discovered_types = []
        seen_codes = set()

        for line in response_text.strip().split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 2:
                    label = parts[0].strip('- •*').strip()
                    code = parts[1].strip().upper().replace(' ', '_')

                    if label and code and code not in seen_codes:
                        discovered_types.append({
                            "code": code,
                            "label": label,
                            "source": "discovered"
                        })
                        seen_codes.add(code)

            # 兼容无分隔符的格式（纯中文）
            elif line and len(line) > 1 and len(line) < 20:
                label = line.strip('- •*').strip()
                if label and label in RELATION_TYPE_NORMALIZATION:
                    code = RELATION_TYPE_NORMALIZATION[label]
                    if code not in seen_codes:
                        discovered_types.append({"code": code, "label": label, "source": "mapped"})
                        seen_codes.add(code)

        logger.info(f"从文档中发现 {len(discovered_types)} 种关系类型")
        return discovered_types[:max_types]

    except Exception as e:
        logger.error(f"关系类型发现失败: {e}")
        # 返回默认类型列表
        return [
            {"code": "RELATED_TO", "label": "相关", "source": "default"},
            {"code": "PART_OF", "label": "包含", "source": "default"},
            {"code": "IS", "label": "是", "source": "default"},
            {"code": "BASED_ON", "label": "基于", "source": "default"},
        ]

def infer_relation_from_fact(
    fact: str,
    source: str = "",
    target: str = "",
    llm_invoke: Optional[LlmInvokeFn] = None,
) -> str:
    """
    使用 LLM 从事实描述中推断关系类型

    基于语义理解，智能推断最合适的关系类型

    Args:
        fact: 关系事实描述
        source: 源实体名称（用于上下文）
        target: 目标实体名称（用于上下文）

    Returns:
        标准化的关系类型
    """
    if not fact:
        return "RELATED_TO"

    # 所有有效的关系类型
    VALID_RELATION_TYPES = {
        # 归属/来源类
        "ORIGINATED_FROM": "源于、来源于、起源于",
        "BELONGS_TO": "属于、隶属于",
        "LOCATED_AT": "位于、坐落在",
        "FROM": "来自",

        # 身份/角色类
        "IS": "是、即是",
        "WAS": "曾是、曾经是",
        "BECAME": "成为、变成了",
        "EMPLOYED_BY": "任职于、工作于",
        "HOLDS_POSITION": "担任",

        # 创作/生产类
        "CREATED": "创作、创造",
        "AUTHORED": "编写、著、撰写",
        "INVENTED": "发明、研发",
        "PROPOSED": "提出、首创",
        "PUBLISHED": "发表、发布",
        "STARRED_IN": "主演",

        # 组成/包含类
        "CONTAINS": "包含、包括",
        "INCLUDES": "包括、含有",
        "PART_OF": "是...的一部分、组成部分",
        "HAS_PART": "包含...作为部分",
        "DESCRIBES": "阐述、描述",
        "EXPLAINS": "解释、说明",
        "DISCUSSES": "讨论、探讨、论述",

        # 社交/互动类
        "COLLABORATES_WITH": "合作",
        "MENTORS": "指导、辅导",
        "MENTORED_BY": "师承、师从",
        "STUDENT_OF": "学习、就读",
        "TEACHER_OF": "教学、教导",
        "KNOWS": "认识、相识",
        "FRIEND_OF": "朋友",

        # 因果/影响类
        "CAUSES": "导致、致使、造成",
        "CAUSED_BY": "由...导致",
        "INFLUENCES": "影响",
        "INFLUENCED_BY": "受...影响",
        "BASED_ON": "基于、建立在",
        "BASIS_FOR": "是...的基础",
        "DEPENDS_ON": "依赖、依赖于",
        "PROMOTES": "促进、推动",
        "HINDERS": "阻碍、妨碍",

        # 学术/研究类
        "RESEARCHES": "研究",
        "FIELD_OF": "领域",
        "EXPERT_IN": "专长、擅长",
        "CITES": "引用",
        "REFERENCES": "参考",

        # 参与类
        "PARTICIPATED_IN": "参与、加入",
        "ORGANIZED": "组织、策划",
        "HOSTED": "主持",
        "ATTENDED": "出席、参加",

        # 时间类
        "BEFORE": "早于、先于",
        "AFTER": "晚于、后于",
        "OCCURRED_AT": "发生于、发生在",

        # 通用关系（fallback）
        "RELATED_TO": "相关、关联",
        "ASSOCIATED_WITH": "有关、有关联",
    }

    # 构建 LLM 提示词
    valid_types_list = "\n".join([
        f"  - {t}: {d}" for t, d in VALID_RELATION_TYPES.items()
    ])

    prompt = f"""分析以下关系描述，返回最精确的关系类型。

【关系描述】
{fact}

【实体信息】
源实体：{source if source else "(未提供)"}
目标实体：{target if target else "(未提供)"}

【可选的关系类型】
{valid_types_list}

【要求】
1. 根据关系描述的语义，选择最精确的关系类型
2. 只返回关系类型的英文名称，不要有其他内容
3. 如果无法确定，返回 RELATED_TO

关系类型："""

    try:
        if llm_invoke is None:
            return "RELATED_TO"
        response_text = llm_invoke(
            [
                {
                    "role": "system",
                    "content": "你是一个专业的知识图谱构建助手，擅长分析实体之间的语义关系。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            0.1,
            50,
        )

        result = response_text.strip().upper()

        # 清理可能的前缀/后缀
        result = result.strip().strip('"\'.`')

        # 验证返回的关系类型是否有效
        if result in VALID_RELATION_TYPES:
            logger.debug(f"LLM推断关系: '{fact[:30]}...' -> {result}")
            return result
        else:
            logger.warning(f"LLM返回无效关系类型: {result}，使用 RELATED_TO")
            return "RELATED_TO"

    except Exception as e:
        logger.error(f"LLM推断关系失败: {e}，使用 RELATED_TO")
        return "RELATED_TO"


def infer_relation_dynamic(
    fact: str,
    source: str = "",
    target: str = "",
    project_id: str = None,
    llm_invoke: Optional[LlmInvokeFn] = None,
) -> str:
    """
    使用项目上下文动态推断关系类型

    结合项目特定的关系类型和通用类型库，提供最适配的关系推断

    Args:
        fact: 关系事实描述
        source: 源实体名称
        target: 目标实体名称
        project_id: 项目ID，用于获取项目特定的关系类型

    Returns:
        标准化的关系类型代码
    """
    if not fact:
        return "RELATED_TO"

    # 获取项目特定的关系类型
    project_types = []
    if project_id:
        project_types = get_project_relation_types(project_id)

    # 合并通用类型作为基础
    base_types = [
        {"code": "RELATED_TO", "label": "相关"},
        {"code": "PART_OF", "label": "组成部分"},
        {"code": "IS", "label": "是"},
        {"code": "BASED_ON", "label": "基于"},
        {"code": "ORIGINATED_FROM", "label": "源于"},
        {"code": "BELONGS_TO", "label": "属于"},
        {"code": "CONTAINS", "label": "包含"},
        {"code": "INFLUENCES", "label": "影响"},
        {"code": "CAUSES", "label": "导致"},
        {"code": "CREATED", "label": "创作"},
    ]

    # 合并去重（项目类型优先）
    all_types = project_types + [
        t for t in base_types if t["code"] not in {pt["code"] for pt in project_types}
    ]

    # 构建 LLM 提示词
    valid_types_list = "\n".join([
        f"  - {t['code']}: {t['label']}" for t in all_types
    ])

    prompt = f"""分析以下关系描述，返回最精确的关系类型代码。

【关系描述】
{fact}

【实体信息】
源实体：{source if source else "(未提供)"}
目标实体：{target if target else "(未提供)"}

【可选的关系类型】
{valid_types_list}

【要求】
1. 根据关系描述的语义，从上述类型中选择最合适的
2. 只返回类型的英文名称（如 PART_OF），不要有其他内容
3. 如果现有类型都不合适，可以创建新的类型（格式：动词+_RELATION，如 TREATS_RELATION）
4. 如果完全无法确定，返回 RELATED_TO

关系类型代码："""

    try:
        if llm_invoke is None:
            return "RELATED_TO"
        response_text = llm_invoke(
            [
                {
                    "role": "system",
                    "content": "你是一个专业的知识图谱构建助手，擅长根据项目上下文分析实体关系。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            0.1,
            50,
        )

        result = response_text.strip().strip().strip('"\'.`')
        result = result.upper()

        # 验证是否为有效的类型
        valid_codes = {t["code"] for t in all_types}
        if result in valid_codes:
            logger.debug(f"动态推断关系: '{fact[:30]}...' -> {result} (project: {project_id})")
            return result
        else:
            # 处理新创建的类型（以 _RELATION 结尾）
            if result.endswith("_RELATION"):
                # 记录新类型到项目配置
                new_label = result.replace("_RELATION", "")
                if project_id:
                    # 将新类型添加到项目配置
                    current_types = get_project_relation_types(project_id)
                    new_type_entry = {"code": result, "label": new_label, "source": "discovered"}
                    if result not in {ct["code"] for ct in current_types}:
                        current_types.append(new_type_entry)
                        save_project_relation_types(project_id, current_types)
                        logger.info(f"项目 {project_id} 新增关系类型: {result} ({new_label})")
                return result
            else:
                logger.warning(f"LLM返回无效关系类型: {result}，使用 RELATED_TO")
                return "RELATED_TO"

    except Exception as e:
        logger.error(f"动态关系推断失败: {e}，使用 RELATED_TO")
        return "RELATED_TO"


def normalize_relation_type(raw_type: str) -> str:
    """
    标准化关系类型

    将中文或非标准关系类型映射到标准化的英文关系类型

    Args:
        raw_type: 原始关系类型

    Returns:
        标准化的关系类型
    """
    if not raw_type:
        return "RELATED_TO"

    raw_type = raw_type.strip()

    # 已经是标准化的英文类型
    if raw_type.upper() in RELATION_TYPE_NORMALIZATION.values():
        return raw_type.upper()

    # 查找映射
    for key, value in RELATION_TYPE_NORMALIZATION.items():
        if key in raw_type or raw_type in key:
            return value

    # 如果包含多个关键词，尝试分别匹配
    for key, value in RELATION_TYPE_NORMALIZATION.items():
        if key in raw_type:
            return value

    # 未找到映射，返回原值（转大写）
    return raw_type.upper() if len(raw_type) < 20 else "RELATED_TO"


@dataclass
class ExtractedEntity:
    """提取的实体"""
    name: str
    entity_type: str
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.entity_type,
            "description": self.description,
            "attributes": self.attributes
        }


@dataclass
class ExtractedRelation:
    """提取的关系"""
    source: str
    target: str
    relation_type: str
    description: str = ""
    strength: int = 5  # 关系强度 1-10
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type,
            "description": self.description,
            "strength": self.strength,
            "attributes": self.attributes
        }


@dataclass
class ExtractionResult:
    """抽取结果"""
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    tokens_used: int = 0


# ========== RAGFlow/LightRAG 风格实体抽取 Prompt ==========

GRAPH_EXTRACTION_PROMPT = """---Goal---
给定一个文本文档和一系列实体类型，请从文本中识别出所有这些类型的实体，以及这些实体之间的关系。
使用原文语言作为输出语言。

---Steps---
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体名称，使用原文语言。如果是英文，请首字母大写。
- entity_type: 以下类型之一: [{entity_types}]
- entity_description: *仅基于输入文本中明确存在的信息* 提供实体属性和活动的全面描述。**不要推断或虚构未明确说明的信息。** 如果文本提供的信息不足以创建全面描述，请说明"文本中无可用描述"。

格式：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1识别的实体中，识别所有相互之间有明显关系的(源实体, 目标实体)对。
对于每对相关实体，提取以下信息：
- source_entity: 源实体名称（来自步骤1）
- target_entity: 目标实体名称（来自步骤1）
- relationship_description: 解释为什么认为源实体和目标实体相关
- relationship_keywords: 一个或多个关键词，概括关系的具体类型。**请使用精确的动词或关系词**，例如：
  * 归属类：属于、位于、源于
  * 身份类：是、担任、任职于
  * 社交类：合作、指导、学生、朋友
  * 创作类：创作、发表、提出、发明
  * 内容类：包含、部分、阐述
  * 因果类：导致、影响、基于
  * 参与类：参与、组织、主持
  * 学术类：研究、领域、专长
- relationship_strength: 表示源实体和目标实体之间关系强度的数字评分 (1-10)

格式：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 识别概括整个文本的主要概念、主题或话题的高级关键词。这些应该捕获文档中存在的整体思想。
格式：("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 使用 **{record_delimiter}** 作为列表分隔符，返回步骤1和2中识别的所有实体和关系的单个列表。

5. 完成后，输出 {completion_delimiter}

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: [{entity_types}]
Text: {input_text}
######################
Output:"""

# Few-Shot 示例（根据领域动态选择）
ENTITY_EXTRACTION_EXAMPLES = {
    "default": [
        """Example 1:

Entity_types: [person, organization, location, product, technology, metric, time, event, concept]
Text:
```
Acme Cloud 在 2024 年发布了 AuroraDB 3.0。该版本基于 Raft 共识协议，并在多可用区部署中将平均故障切换时间降至 30 秒。官方技术报告《AuroraDB 3.0 Availability Benchmark》记录了这组数据。发布会在上海举办，平台团队和数据库团队共同参与。
```

Output:
("entity"{tuple_delimiter}"Acme Cloud"{tuple_delimiter}"organization"{tuple_delimiter}"Acme Cloud 发布了 AuroraDB 3.0"){record_delimiter}
("entity"{tuple_delimiter}"AuroraDB 3.0"{tuple_delimiter}"product"{tuple_delimiter}"AuroraDB 3.0 是发布版本"){record_delimiter}
("entity"{tuple_delimiter}"Raft 共识协议"{tuple_delimiter}"technology"{tuple_delimiter}"该版本基于 Raft 共识协议"){record_delimiter}
("entity"{tuple_delimiter}"30 秒"{tuple_delimiter}"metric"{tuple_delimiter}"平均故障切换时间降至 30 秒"){record_delimiter}
("entity"{tuple_delimiter}"2024 年"{tuple_delimiter}"time"{tuple_delimiter}"发布时间为 2024 年"){record_delimiter}
("entity"{tuple_delimiter}"AuroraDB 3.0 Availability Benchmark"{tuple_delimiter}"document"{tuple_delimiter}"官方技术报告记录了可用性数据"){record_delimiter}
("entity"{tuple_delimiter}"上海"{tuple_delimiter}"location"{tuple_delimiter}"发布会举办地点是上海"){record_delimiter}
("entity"{tuple_delimiter}"发布会"{tuple_delimiter}"event"{tuple_delimiter}"发布会是相关活动事件"){record_delimiter}
("relationship"{tuple_delimiter}"Acme Cloud"{tuple_delimiter}"AuroraDB 3.0"{tuple_delimiter}"Acme Cloud 发布了 AuroraDB 3.0"{tuple_delimiter}"发布, 拥有"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"AuroraDB 3.0"{tuple_delimiter}"Raft 共识协议"{tuple_delimiter}"AuroraDB 3.0 基于 Raft 共识协议"{tuple_delimiter}"基于, 使用"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"AuroraDB 3.0"{tuple_delimiter}"30 秒"{tuple_delimiter}"该版本将平均故障切换时间降至 30 秒"{tuple_delimiter}"指标, 改善"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"AuroraDB 3.0 Availability Benchmark"{tuple_delimiter}"30 秒"{tuple_delimiter}"技术报告记录了该指标"{tuple_delimiter}"记录, 说明"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"发布会"{tuple_delimiter}"上海"{tuple_delimiter}"发布会在上海举办"{tuple_delimiter}"位于, 举办于"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"产品发布, 可用性, 共识协议, 指标优化"){completion_delimiter}
#############################""",
    ]
}

CONTINUE_PROMPT = """
上次抽取遗漏了许多实体。请仅从文本中找出之前遗漏的实体和关系。

---Remember Steps---

1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体名称，使用原文语言
- entity_type: 以下类型之一: [{entity_types}]
- entity_description: *仅基于输入文本中明确存在的信息* 提供实体属性和活动的全面描述。**不要推断或虚构未明确说明的信息。**

格式：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1识别的实体中，识别所有相互之间有明显关系的(源实体, 目标实体)对。
对于每对相关实体，提取以下信息：
- source_entity: 源实体名称
- target_entity: 目标实体名称
- relationship_description: 解释为什么认为源实体和目标实体相关
- relationship_keywords: 精确的关系动词，如：属于、是、合作、指导、创作、包含、导致、参与、研究等
- relationship_strength: 关系强度评分 (1-10)

格式：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

---Output---

使用相同格式在下方添加新的实体和关系，不要包含之前已抽取的实体和关系：
""".strip()

LOOP_PROMPT = """
---Goal---

似乎还有一些实体被遗漏了。

---Output---

仅回答 `YES` 或 `NO` 是否还有需要添加的实体。
""".strip()


class EntityExtractionStrategy:
    """抽取策略基类"""

    SHORT_THRESHOLD = 500
    MEDIUM_THRESHOLD = 2000

    @classmethod
    def get_strategy(cls, text: str) -> str:
        length = len(text)
        if length < cls.SHORT_THRESHOLD:
            return "short"
        elif length < cls.MEDIUM_THRESHOLD:
            return "medium"
        else:
            return "long"


class EntityExtractor:
    """
    基于 GraphRAG 的 LLM 实体抽取器

    使用注入的 LLM 调用函数进行实体和关系抽取
    """

    # 默认实体类型（领域中性）
    DEFAULT_ENTITY_TYPES = [
        "Person",
        "Organization",
        "Location",
        "Event",
        "Concept",
        "Method",
        "Product",
        "Document",
        "Technology",
        "Metric",
        "Time",
        "Object",
        "Category",
    ]

    # 实体类型中文映射
    ENTITY_TYPE_ZH = {
        "人物": "Person",
        "人": "Person",
        "角色": "Person",
        "组织": "Organization",
        "机构": "Organization",
        "公司": "Organization",
        "企业": "Organization",
        "地点": "Location",
        "地方": "Location",
        "场所": "Location",
        "概念": "Concept",
        "理论": "Concept",
        "方法": "Method",
        "技能": "Method",
        "事件": "Event",
        "物品": "Object",
        "产品": "Product",
        "文档": "Document",
        "文件": "Document",
        "技术": "Technology",
        "指标": "Metric",
        "数据": "Metric",
        "时间": "Time",
        "日期": "Date",
        "类别": "Category",
        "分类": "Category",
    }

    def __init__(
        self,
        llm_invoke: LlmInvokeFn,
        model: str = None,
        max_gleanings: int = 1,
        entity_types: List[str] = None
    ):
        if llm_invoke is None:
            raise ValueError("llm_invoke 未配置")

        self.llm_invoke = llm_invoke
        self.model = model or "dify-reverse-invocation"
        self.max_gleanings = max_gleanings
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES

        # 提示词变量
        self.prompt_variables = {
            "tuple_delimiter": DEFAULT_TUPLE_DELIMITER,
            "record_delimiter": DEFAULT_RECORD_DELIMITER,
            "completion_delimiter": DEFAULT_COMPLETION_DELIMITER,
            "entity_types": ",".join(self.entity_types),
        }

    def extract(
        self,
        text: str,
        ontology: Dict[str, Any] = None,
        graph_context: str = ""
    ) -> ExtractionResult:
        """
        从文本中抽取实体和关系

        Args:
            text: 待抽取的文本
            ontology: 本体定义（可选）
            graph_context: 图谱上下文（可选）

        Returns:
            抽取结果
        """
        strategy = EntityExtractionStrategy.get_strategy(text)
        logger.info(f"开始实体抽取: 策略={strategy}, 文本长度={len(text)}")

        # 从本体中提取实体类型
        entity_types = self._get_entity_types_from_ontology(ontology)

        if strategy == "short":
            return self._extract_with_gleaning(text, entity_types, ontology)
        elif strategy == "medium":
            return self._extract_medium(text, entity_types, graph_context, ontology)
        else:
            return self._extract_long(text, entity_types, graph_context, ontology)

    def _get_entity_types_from_ontology(self, ontology: Dict[str, Any] = None) -> List[str]:
        """从本体定义中提取实体类型"""
        # 基础类型（始终包含）
        base_types = list(self.entity_types) if self.entity_types else list(self.DEFAULT_ENTITY_TYPES)

        if not ontology:
            return base_types

        ontology_types = [e.get("name") for e in ontology.get("entity_types", [])]

        # 合并本体类型和基础类型（去重）
        all_types = list(dict.fromkeys(ontology_types + base_types))

        return all_types if all_types else base_types

    def _build_entity_type_descriptions(self, ontology: Dict[str, Any] = None, entity_types: List[str] = None) -> str:
        """
        构建实体类型描述列表

        RAGFlow 模式：为每个实体类型提供清晰的描述，帮助 LLM 理解如何分类
        """
        # 基础类型描述（fallback）
        base_descriptions = {
            "Person": "人物：具体个人、角色、作者、发言人等",
            "Organization": "组织：公司、机构、团队、学校、部门等",
            "Location": "地点：国家、城市、园区、建筑、空间位置等",
            "Concept": "概念：理论、规则、术语、抽象思想等",
            "Method": "方法：流程、算法、策略、操作方式等",
            "Event": "事件：发布、会议、活动、事故、交易等有发生过程的事项",
            "Object": "对象：具体事物、设备、材料、作品等",
            "Product": "产品：软件、硬件、服务、SKU、型号等",
            "Document": "文档：标准、报告、论文、合同、说明文档等",
            "Technology": "技术：框架、协议、工具链、平台技术等",
            "Metric": "指标：KPI、统计值、测量口径、数值指标等",
            "Time": "时间：日期、时刻、周期、时间范围等",
        }

        if entity_types is None:
            entity_types = self._get_entity_types_from_ontology(ontology)

        descriptions = []
        for et in entity_types:
            # 首先从本体获取描述
            if ontology:
                for et_def in ontology.get("entity_types", []):
                    if et_def.get("name") == et:
                        desc = et_def.get("description", "")
                        if desc:
                            descriptions.append(f"- {et}: {desc}")
                            break
                else:
                    # 本体中没有，使用基础描述
                    if et in base_descriptions:
                        descriptions.append(f"- {et}: {base_descriptions[et]}")
                    else:
                        descriptions.append(f"- {et}: 实体类型")
            else:
                # 没有本体，使用基础描述
                if et in base_descriptions:
                    descriptions.append(f"- {et}: {base_descriptions[et]}")
                else:
                    descriptions.append(f"- {et}: 实体类型")

        return "\n".join(descriptions)

    def _extract_with_gleaning(
        self,
        text: str,
        entity_types: List[str],
        ontology: Dict[str, Any] = None
    ) -> ExtractionResult:
        """
        使用 GraphRAG 的多轮抽取方法

        核心思想：多次调用 LLM，每次询问是否还有遗漏的实体
        """
        results = ""
        total_tokens = 0

        # 构建变量
        variables = {**self.prompt_variables, "entity_types": ",".join(entity_types)}
        variables["input_text"] = text
        variables["entity_type_descriptions"] = self._build_entity_type_descriptions(ontology, entity_types)
        variables["examples"] = "\n\n".join(ENTITY_EXTRACTION_EXAMPLES["default"])

        # 构建初始提示词
        prompt = self._replace_variables(GRAPH_EXTRACTION_PROMPT, variables)

        # 第一次调用
        response = self._call_llm(prompt, temperature=0.3)
        total_tokens += self._count_tokens(prompt + response)
        results += response or ""

        history = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Output:"},
            {"role": "assistant", "content": response}
        ]

        # 多轮抽取 (Gleaning)
        for i in range(self.max_gleanings):
            # 构建继续抽取的 prompt（替换变量）
            continue_prompt = self._replace_variables(CONTINUE_PROMPT, {
                "tuple_delimiter": DEFAULT_TUPLE_DELIMITER,
                "record_delimiter": DEFAULT_RECORD_DELIMITER,
                "entity_types": ",".join(entity_types)
            })
            history.append({"role": "user", "content": continue_prompt})
            response = self._call_llm("", messages=history, temperature=0.3)
            total_tokens += self._count_tokens(continue_prompt + response)
            results += response or ""

            if i >= self.max_gleanings - 1:
                break

            history.append({"role": "assistant", "content": response})
            history.append({"role": "user", "content": LOOP_PROMPT})

            # 询问是否还有更多实体
            continuation = self._call_llm("", messages=history, temperature=0.0, max_tokens=1)
            total_tokens += self._count_tokens(LOOP_PROMPT + continuation)

            if continuation.strip().upper() != "Y":
                logger.debug(f"第 {i+1} 轮抽取完成，无更多实体")
                break

            history.append({"role": "assistant", "content": continuation})
            logger.debug(f"继续第 {i+2} 轮抽取...")

        # 解析结果
        entities, relations = self._parse_graphrag_results(results)

        logger.info(f"多轮抽取完成: 实体={len(entities)}, 关系={len(relations)}, tokens={total_tokens}")

        return ExtractionResult(
            entities=entities,
            relations=relations,
            tokens_used=total_tokens
        )

    def _extract_medium(
        self,
        text: str,
        entity_types: List[str],
        graph_context: str,
        ontology: Dict[str, Any] = None
    ) -> ExtractionResult:
        """
        中等文本抽取：先提取主题，再进行完整抽取
        """
        # 先提取主要主题
        topics_prompt = f"""请分析以下文本，提取出主要主题（3-5个关键词）：

{text[:1000]}

请以 JSON 格式返回：
{{"topics": ["主题1", "主题2", ...]}}
"""

        topics_response = self._call_llm(topics_prompt, temperature=0.3)
        topics = self._extract_topics(topics_response)

        # 进行完整抽取
        context = f"\n\n核心主题: {', '.join(topics)}"
        if graph_context:
            context += f"\n{graph_context}"

        variables = {**self.prompt_variables, "entity_types": ",".join(entity_types)}
        variables["input_text"] = text
        variables["entity_type_descriptions"] = self._build_entity_type_descriptions(ontology, entity_types)
        variables["examples"] = "\n\n".join(ENTITY_EXTRACTION_EXAMPLES["default"]) + context

        prompt = self._replace_variables(GRAPH_EXTRACTION_PROMPT, variables)
        response = self._call_llm(prompt, temperature=0.3)

        entities, relations = self._parse_graphrag_results(response)

        return ExtractionResult(
            entities=entities,
            relations=relations,
            topics=topics,
            tokens_used=self._count_tokens(prompt + response)
        )

    def _extract_long(
        self,
        text: str,
        entity_types: List[str],
        graph_context: str,
        ontology: Dict[str, Any] = None
    ) -> ExtractionResult:
        """
        长文本抽取：分段处理 + 实体去重
        """
        # 智能分段
        paragraphs = self._split_text_smartly(text)

        all_entities = []
        all_relations = []
        entity_keys = set()  # 用于去重: name|type

        total_tokens = 0
        processed_chunks = 0

        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < 50:
                continue

            # 限制单次输入长度
            chunk_text = paragraph[:2000]

            variables = {**self.prompt_variables, "entity_types": ",".join(entity_types)}
            variables["input_text"] = chunk_text
            variables["entity_type_descriptions"] = self._build_entity_type_descriptions(ontology, entity_types)
            variables["examples"] = "\n\n".join(ENTITY_EXTRACTION_EXAMPLES["default"])

            prompt = self._replace_variables(GRAPH_EXTRACTION_PROMPT, variables)
            response = self._call_llm(prompt, temperature=0.3)
            total_tokens += self._count_tokens(prompt + response)

            entities, relations = self._parse_graphrag_results(response)

            # 去重添加实体
            for entity in entities:
                key = f"{entity.name}|{entity.entity_type}"
                if key not in entity_keys:
                    all_entities.append(entity)
                    entity_keys.add(key)

            all_relations.extend(relations)
            processed_chunks += 1

            if processed_chunks % 5 == 0:
                logger.debug(f"已处理 {processed_chunks}/{len(paragraphs)} 个段落")

        logger.info(f"长文本抽取完成: 段落={processed_chunks}, 实体={len(all_entities)}, 关系={len(all_relations)}")

        return ExtractionResult(
            entities=all_entities,
            relations=all_relations,
            tokens_used=total_tokens
        )

    def _parse_graphrag_results(self, results: str) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        解析 GraphRAG 格式的抽取结果

        格式: ("entity"{tuple_delimiter}<name>{tuple_delimiter}<type>{tuple_delimiter}<desc>)
        或: ("relationship"{tuple_delimiter}<source>{tuple_delimiter}<target>{tuple_delimiter}<desc>{tuple_delimiter}<strength>)
        """
        entities = []
        relations = []

        # 按记录分隔符分割
        records = re.split(
            f'({re.escape(DEFAULT_RECORD_DELIMITER)}|{re.escape(DEFAULT_COMPLETION_DELIMITER)})',
            results
        )

        for record in records:
            record = record.strip()
            if not record or record in [DEFAULT_RECORD_DELIMITER, DEFAULT_COMPLETION_DELIMITER]:
                continue

            # 提取括号内的内容
            match = re.search(r'\((.*)\)', record)
            if not match:
                continue

            content = match.group(1)

            # 按元组分隔符分割
            parts = content.split(DEFAULT_TUPLE_DELIMITER)

            if len(parts) < 3:
                continue

            # 去除引号（处理 LLM 可能返回的转义引号）
            record_type = parts[0].strip().strip('"').strip("'")

            if record_type == "entity" and len(parts) >= 4:
                # ("entity"{delimiter}<name>{delimiter}<type>{delimiter}<description>)
                entity_name = parts[1].strip().strip('"').strip("'")
                entity_type = parts[2].strip().strip('"').strip("'")
                entity_desc = parts[3].strip().strip('"').strip("'")

                # 跳过空名称的实体
                if not entity_name or entity_name.lower() in ('none', 'null', 'n/a', ''):
                    logger.debug(f"跳过空名称实体: {entity_name}")
                    continue

                # 标准化实体类型
                normalized_type = self._normalize_entity_type(entity_type)

                # 如果类型为空或通用，尝试根据名称推断
                if not normalized_type or normalized_type == "Entity":
                    normalized_type = self._infer_entity_type(entity_name, entity_desc)

                entities.append(ExtractedEntity(
                    name=entity_name,
                    entity_type=normalized_type,
                    description=entity_desc
                ))

            elif record_type == "relationship":
                # 新格式: ("relationship"{delimiter}<source>{delimiter}<target>{delimiter}<description>{delimiter}<keywords>{delimiter}<strength>)
                # 旧格式: ("relationship"{delimiter}<source>{delimiter}<target>{delimiter}<description>{delimiter}<strength>)
                try:
                    if len(parts) >= 6:
                        # 新格式，带 keywords
                        keywords = parts[4].strip().strip('"').strip("'")
                        strength_idx = 5
                    else:
                        # 旧格式，不带 keywords
                        keywords = ""
                        strength_idx = 4

                    strength = 5
                    if len(parts) > strength_idx:
                        strength_str = parts[strength_idx].strip().strip('"').strip("'")
                        if strength_str.isdigit():
                            strength = int(strength_str)

                    # 使用 keywords 作为关系类型，并进行标准化
                    relation_type = "RELATED_TO"  # 默认值
                    if keywords:
                        # keywords 可能是逗号/顿号分隔的多个关键词，取第一个
                        primary_keyword = keywords.split(',')[0].split('、')[0].strip()
                        if primary_keyword:
                            relation_type = normalize_relation_type(primary_keyword)

                    relations.append(ExtractedRelation(
                        source=parts[1].strip().strip('"').strip("'"),
                        target=parts[2].strip().strip('"').strip("'"),
                        relation_type=relation_type,
                        description=parts[3].strip().strip('"').strip("'"),
                        strength=max(1, min(10, strength)),
                        attributes={"keywords": keywords} if keywords else {}
                    ))
                except Exception as e:
                    logger.debug(f"关系解析失败: {e}, parts={parts}")

        return entities, relations

    def _call_llm(
        self,
        prompt: str = "",
        messages: List[Dict] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> str:
        """调用 LLM"""
        try:
            if messages is None:
                messages = [
                    {"role": "system", "content": "你是一个专业的知识图谱构建助手，擅长从文本中提取实体和关系。"},
                    {"role": "user", "content": prompt}
                ]
            response_text = self.llm_invoke(messages, temperature, max_tokens)
            return (response_text or "").strip()

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return ""

    def _replace_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """替换模板变量"""
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def _count_tokens(self, text: str) -> int:
        """估算 token 数量（中文约 1.5 字符 = 1 token）"""
        return len(text) // 2

    def _extract_topics(self, response: str) -> List[str]:
        """从 LLM 响应中提取主题"""
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get("topics", [])

            # 尝试提取列表
            list_match = re.search(r'\[.*\]', response, re.DOTALL)
            if list_match:
                return json.loads(list_match.group(0))

            # 提取行内容
            lines = response.strip().split('\n')
            topics = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('{') and not line.startswith('['):
                    # 移除可能的序号和符号
                    line = re.sub(r'^[\d\.\-\*\+]+\s*', '', line)
                    line = re.sub(r'^["\']|["\']$', '', line)
                    if line:
                        topics.append(line)
            return topics[:5]
        except:
            return []

    def _normalize_entity_type(self, raw_type: str) -> str:
        """标准化实体类型"""
        raw_type = raw_type.strip()

        # 中文类型映射
        if raw_type in self.ENTITY_TYPE_ZH:
            return self.ENTITY_TYPE_ZH[raw_type]

        # 英文类型直接返回
        return raw_type if raw_type else "Entity"

    def _infer_entity_type(self, entity_name: str, description: str = "") -> str:
        """
        根据实体名称和描述推断实体类型

        用于处理 LLM 返回通用类型的情况。
        该推断器保持“弱假设”：仅在存在明显模式时归类，否则返回 Entity。
        """
        name_lower = entity_name.lower()
        desc_lower = description.lower()

        # 时间：明显时间格式/关键词
        if re.search(r"\b(19|20)\d{2}\b", entity_name) or re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", entity_name):
            return "Time"
        time_keywords = ["年", "月", "日", "时", "季度", "year", "month", "day", "quarter"]
        if any(kw in name_lower for kw in time_keywords):
            return "Time"

        # 组织
        org_keywords = [
            "公司", "集团", "大学", "学院", "研究院", "委员会", "部门", "实验室",
            "inc", "corp", "company", "university", "institute", "lab", "team",
        ]
        if any(kw in name_lower for kw in org_keywords):
            return "Organization"

        # 地点
        location_keywords = [
            "省", "市", "区", "县", "路", "街", "园区", "大厦", "机场", "车站",
            "city", "country", "province", "street", "park",
        ]
        if any(kw in name_lower for kw in location_keywords):
            return "Location"

        # 事件
        event_keywords = [
            "发布会", "会议", "峰会", "比赛", "活动", "事故", "收购", "并购",
            "launch", "summit", "meeting", "event", "incident", "acquisition",
        ]
        if any(kw in name_lower for kw in event_keywords):
            return "Event"

        # 文档
        doc_keywords = ["报告", "白皮书", "论文", "规范", "标准", "文档", "report", "paper", "spec", "standard"]
        if any(kw in name_lower for kw in doc_keywords):
            return "Document"

        # 技术/方法
        method_keywords = ["算法", "协议", "框架", "模型", "方法", "流程", "algorithm", "protocol", "framework", "model"]
        if any(kw in name_lower or kw in desc_lower for kw in method_keywords):
            return "Method"

        # 产品
        product_keywords = ["系统", "平台", "产品", "版本", "软件", "应用", "platform", "system", "product", "version", "app"]
        if any(kw in name_lower for kw in product_keywords):
            return "Product"

        # 指标
        metric_keywords = ["率", "比率", "得分", "指数", "增长", "下降", "ms", "秒", "%", "score", "rate", "latency"]
        if any(kw in name_lower for kw in metric_keywords):
            return "Metric"

        # 概念
        concept_keywords = ["理论", "概念", "原则", "思想", "策略", "theory", "concept", "principle", "idea"]
        if any(kw in name_lower or kw in desc_lower for kw in concept_keywords):
            return "Concept"

        # 人物：放在靠后，避免把多数短词都误判为人物
        if re.match(r"^[\u4e00-\u9fa5]{2,4}$", entity_name):
            return "Person"

        # 默认通用类型
        return "Entity"

    def _split_text_smartly(self, text: str) -> List[str]:
        """智能分段"""
        # 按双换行分段
        paragraphs = re.split(r'\n\n+', text)

        # 如果没有双换行，按单换行分段
        if len(paragraphs) <= 1:
            paragraphs = re.split(r'\n', text)

        # 合并过短的段落
        result = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) < 1000:
                current += (" " if current else "") + para
            else:
                if current:
                    result.append(current)
                current = para

        if current:
            result.append(current)

        return result


class GraphEntityExtractor:
    """
    图谱实体抽取器

    将抽取的实体和关系写入 Neo4j 图谱
    """

    def __init__(self, llm_invoke: LlmInvokeFn, model: Optional[str] = None):
        from .zep_adapter.graph import Neo4jRepository

        self.neo4j = Neo4jRepository()
        self.extractor = EntityExtractor(llm_invoke=llm_invoke, model=model)


    def extract_and_store(
        self,
        graph_id: str,
        text: str,
        ontology: Dict[str, Any] = None,
        enable_resolution: bool = True
    ) -> ExtractionResult:
        """
        抽取实体并存储到图谱

        Args:
            graph_id: 图谱 ID
            text: 待抽取的文本
            ontology: 本体定义
            enable_resolution: 是否启用实体解析去重

        Returns:
            抽取结果
        """
        # 获取已有实体作为上下文
        existing_entities = self._get_existing_entities(graph_id, limit=20)
        logger.info(f"图谱 {graph_id} 中已有实体数量: {len(existing_entities)}")
        context = ""
        if existing_entities:
            context = f"\n\n图谱中已有的实体: {', '.join(existing_entities)}"

        entity_types = self.extractor._get_entity_types_from_ontology(ontology)
        types_preview = ", ".join(entity_types[:5])
        logger.info(
            f"开始实体抽取: graph_id={graph_id}, "
            f"text_len={len(text)}, entity_types={len(entity_types)} "
            f"({types_preview}{'...' if len(entity_types) > 5 else ''}), "
            f"enable_resolution={enable_resolution}"
        )

        # 执行抽取
        result = self.extractor.extract(text, ontology, context)
        logger.info(
            f"抽取结果: entities={len(result.entities)}, relations={len(result.relations)}"
        )

        if not result.entities:
            logger.warning(
                f"未抽取到实体: graph_id={graph_id}, "
                f"model={self.extractor.model}"
            )
            text_preview = text[:200].replace("\n", " ")
            logger.debug(f"文本预览: {text_preview}")
        else:
            sample_entities = ", ".join([e.name for e in result.entities[:5] if e.name])
            if sample_entities:
                logger.info(f"实体样例: {sample_entities}")

        if result.relations:
            sample_relations = ", ".join([r.relation_type for r in result.relations[:5] if r.relation_type])
            if sample_relations:
                logger.info(f"关系样例: {sample_relations}")
        # 实体解析去重（可选）
        if enable_resolution and len(result.entities) > 1:
            logger.info("开始实体解析去重...")
            resolver = EntityResolver()
            resolved_entities = resolver.resolve_entities(
                result.entities,
                use_llm=False,  # 暂时禁用 LLM 以加快速度
                distance_threshold=0.7
            )
            # 更新结果
            result.entities = resolved_entities

        # 存储到 Neo4j
        self._store_entities(graph_id, result.entities)
        self._store_relations(graph_id, result.relations)

        logger.info(f"实体抽取并存储完成: {len(result.entities)} 个实体, {len(result.relations)} 个关系")

        return result

    def _get_existing_entities(self, graph_id: str, limit: int = 50) -> List[str]:
        """获取已有实体名称"""
        query = """
        MATCH (n {graph_id: $graph_id})
        WHERE n.name IS NOT NULL AND n:Entity
        RETURN n.name as name
        LIMIT $limit
        """
        results = self.neo4j._execute_query(query, {"graph_id": graph_id, "limit": limit})
        return [r["name"] for r in results if r.get("name")]

    def _store_entities(self, graph_id: str, entities: List[ExtractedEntity], merge_mode: bool = True):
        """存储实体到 Neo4j，支持合并模式"""
        created_count = 0
        updated_count = 0
        skipped_count = 0
        failed_count = 0

        for entity in entities:
            # 检查是否已存在同名同类实体
            check_query = """
            MATCH (n {graph_id: $graph_id, name: $name})
            WHERE $entity_type IN labels(n)
            RETURN n.uuid as uuid, n.summary as summary, n.attributes as attributes
            """
            existing = self.neo4j._execute_query(check_query, {
                "graph_id": graph_id,
                "name": entity.name,
                "entity_type": entity.entity_type
            })

            if existing:
                if merge_mode:
                    # 合并模式：叠加属性和描述
                    existing_uuid = existing[0].get("uuid")
                    existing_summary = existing[0].get("summary", "")
                    # 将 Neo4j Map 转换为纯 Python dict
                    existing_attrs = _convert_neo4j_record(existing[0].get("attributes")) or {}

                    # 合并描述
                    merged_summary = existing_summary
                    if entity.description and entity.description not in existing_summary:
                        merged_summary = f"{existing_summary}; {entity.description}" if existing_summary else entity.description

                    # 合并属性并清洗
                    merged_attrs = _sanitize_attributes({**existing_attrs, **entity.attributes})

                    # 更新实体
                    update_query = """
                    MATCH (n {uuid: $uuid})
                    SET n.summary = $summary, n.attributes = $attributes
                    RETURN n.uuid as uuid
                    """
                    logger.debug(f"准备更新实体: name={entity.name}, attrs={merged_attrs}")
                    # 将 attributes 字典序列化为 JSON 字符串（Neo4j 不支持 Map 类型作为属性值）
                    attrs_json = json.dumps(merged_attrs, ensure_ascii=False) if merged_attrs else "{}"
                    self.neo4j._execute_query(update_query, {
                        "uuid": existing_uuid,
                        "summary": merged_summary,
                        "attributes": attrs_json
                    })
                    logger.debug(f"合并实体: {entity.name} ({entity.entity_type})")
                    updated_count += 1
                else:
                    logger.debug(f"实体已存在，跳过: {entity.name} ({entity.entity_type})")
                    skipped_count += 1
                continue

            # 创建新实体
            try:
                # 清洗属性值，确保没有嵌套的 dict/map（Neo4j 不支持）
                sanitized_attrs = _sanitize_attributes({"entity_type": entity.entity_type, **entity.attributes})
                self.neo4j.create_node(
                    graph_id=graph_id,
                    name=entity.name,
                    labels=["Entity", entity.entity_type],
                    summary=entity.description,
                    attributes=sanitized_attrs
                )
                logger.debug(f"创建实体: {entity.name} ({entity.entity_type})")
                created_count += 1
            except Exception as e:
                logger.warning(f"创建实体失败 {entity.name}: {e}")
                failed_count += 1

        logger.info(
            f"实体存储结果: created={created_count}, updated={updated_count}, "
            f"skipped={skipped_count}, failed={failed_count}"
        )

    def _store_relations(self, graph_id: str, relations: List[ExtractedRelation], merge_mode: bool = True):
        """存储关系到 Neo4j，支持合并模式"""
        if not relations:
            return

        # 获取实体名称到 UUID 的映射
        entity_names = list(set([r.source for r in relations] + [r.target for r in relations]))

        name_to_uuid = {}
        for name in entity_names:
            query = """
            MATCH (n {graph_id: $graph_id, name: $name})
            RETURN n.uuid as uuid
            """
            result = self.neo4j._execute_query(query, {"graph_id": graph_id, "name": name})
            if result:
                name_to_uuid[name] = result[0]["uuid"]

        # 创建关系
        for relation in relations:
            source_uuid = name_to_uuid.get(relation.source)
            target_uuid = name_to_uuid.get(relation.target)

            if not source_uuid or not target_uuid:
                logger.debug(f"跳过关系: 源或目标实体不存在 - {relation.source} -> {relation.target}")
                continue

            # 检查关系是否已存在
            check_query = """
            MATCH (s {uuid: $source_uuid})-[r]->(t {uuid: $target_uuid})
            WHERE r.name = $relation_name
            RETURN r.uuid as uuid, r.fact as fact, r.attributes as attributes
            """
            existing = self.neo4j._execute_query(check_query, {
                "source_uuid": source_uuid,
                "target_uuid": target_uuid,
                "relation_name": relation.relation_type
            })

            if existing:
                if merge_mode:
                    # 合并模式：叠加描述、权重和关键词
                    existing_uuid = existing[0].get("uuid")
                    existing_fact = existing[0].get("fact", "")
                    # 将 Neo4j Map 转换为纯 Python dict
                    existing_attrs = _convert_neo4j_record(existing[0].get("attributes")) or {}

                    # 合并描述
                    merged_fact = existing_fact
                    if relation.description and relation.description not in existing_fact:
                        merged_fact = f"{existing_fact}; {relation.description}" if existing_fact else relation.description

                    # 合并属性（包括 strength 和 keywords）
                    merged_attrs = _sanitize_attributes({**existing_attrs, **relation.attributes})

                    # 关系强度取最大值
                    if relation.strength:
                        current_strength = merged_attrs.get("strength", 0)
                        merged_attrs["strength"] = max(current_strength, relation.strength)

                    # 更新关系
                    update_query = """
                    MATCH (s {uuid: $source_uuid})-[r]->(t {uuid: $target_uuid})
                    WHERE r.uuid = $uuid
                    SET r.fact = $fact, r.attributes = $attributes
                    RETURN r.uuid as uuid
                    """
                    self.neo4j._execute_query(update_query, {
                        "source_uuid": source_uuid,
                        "target_uuid": target_uuid,
                        "uuid": existing_uuid,
                        "fact": merged_fact,
                        "attributes": merged_attrs
                    })
                    logger.debug(f"合并关系: {relation.source} -> {relation.target}")
                else:
                    logger.debug(f"关系已存在，跳过: {relation.source} -> {relation.target}")
                continue

            # 创建新关系
            try:
                # 清洗关系属性值，确保没有嵌套的 dict/map
                sanitized_attrs = _sanitize_attributes({
                    "strength": relation.strength,
                    "relation_type": relation.relation_type,
                    **relation.attributes
                })
                self.neo4j.create_edge(
                    graph_id=graph_id,
                    source_uuid=source_uuid,
                    target_uuid=target_uuid,
                    name=relation.relation_type,
                    fact=relation.description,
                    attributes=sanitized_attrs
                )
                logger.debug(f"创建关系: {relation.source} -[{relation.relation_type}]-> {relation.target} (强度: {relation.strength})")
            except Exception as e:
                logger.warning(f"创建关系失败 {relation.source} -> {relation.target}: {e}")


# ========== 实体解析（Entity Resolution）模块 ==========

class EntityResolver:
    """
    实体解析器 - 基于 RAGFlow 的两步去重策略

    1. 初步相似度判断：使用编辑距离等工程手段筛选相似实体
    2. LLM 相似度判断：确定最终的实体合并结果
    """

    def __init__(self, llm_invoke: LlmInvokeFn, model: Optional[str] = None):
        self.llm_invoke = llm_invoke
        self.model = model or "dify-reverse-invocation"

    # 编辑距离算法
    def _edit_distance(self, s1: str, s2: str) -> int:
        """计算两个字符串的编辑距离（Levenshtein距离）"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        计算两个字符串的相似度 (0-1)

        使用归一化编辑距离
        """
        s1 = s1.strip().lower()
        s2 = s2.strip().lower()

        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        max_len = max(len(s1), len(s2))
        distance = self._edit_distance(s1, s2)
        similarity = 1.0 - (distance / max_len)

        return similarity

    def _are_similar_by_distance(
        self,
        name1: str,
        name2: str,
        threshold: float = 0.7
    ) -> bool:
        """使用编辑距离判断两个实体名称是否相似"""
        similarity = self._string_similarity(name1, name2)
        return similarity >= threshold

    def _should_merge_with_llm(
        self,
        entity1: ExtractedEntity,
        entity2: ExtractedEntity
    ) -> Tuple[bool, str]:
        """
        使用 LLM 判断两个实体是否应该合并

        Returns:
            (should_merge, reason)
        """
        prompt = f"""请判断以下两个实体是否指向同一个现实世界实体。

实体1:
- 名称: {entity1.name}
- 类型: {entity1.entity_type}
- 描述: {entity1.description}

实体2:
- 名称: {entity2.name}
- 类型: {entity2.entity_type}
- 描述: {entity2.description}

请考虑：
1. 两个名称是否指向同一事物（包括别名、简称、全称等）
2. 描述的内容是否相关且指向同一对象
3. 实体类型是否兼容

回答格式：
第一行: YES 或 NO（是否应该合并）
第二行: 简短理由（不超过50字）

请直接回答，不要有其他内容。"""

        try:
            content = self.llm_invoke(
                [
                    {"role": "system", "content": "你是一个专业的实体对齐专家，擅长判断两个实体是否指向同一事物。"},
                    {"role": "user", "content": prompt}
                ],
                0.0,
                100,
            ).strip()
            
            # 解析响应
            lines = content.split('\n', 1)
            decision = lines[0].strip().upper()

            should_merge = decision in ('YES', 'Y', '是', '应该', '合并')
            reason = lines[1].strip() if len(lines) > 1 else ""

            return should_merge, reason

        except Exception as e:
            logger.warning(f"LLM 判断失败: {e}")
            return False, ""

    def resolve_entities(
        self,
        entities: List[ExtractedEntity],
        use_llm: bool = True,
        distance_threshold: float = 0.6
    ) -> List[ExtractedEntity]:
        """
        实体解析去重

        Args:
            entities: 待去重的实体列表
            use_llm: 是否使用 LLM 进行最终判断
            distance_threshold: 编辑距离相似度阈值

        Returns:
            去重后的实体列表
        """
        if not entities:
            return []

        logger.info(f"开始实体解析: 输入实体数={len(entities)}, use_llm={use_llm}")

        # 按类型分组
        entities_by_type: Dict[str, List[ExtractedEntity]] = {}
        for entity in entities:
            etype = entity.entity_type
            if etype not in entities_by_type:
                entities_by_type[etype] = []
            entities_by_type[etype].append(entity)

        # 对每个类型组进行去重
        resolved_entities = []
        merge_count = 0

        for etype, type_entities in entities_by_type.items():
            # 按名称长度排序（优先保留较短的名称）
            type_entities.sort(key=lambda e: len(e.name))

            merged = set()
            for i, e1 in enumerate(type_entities):
                if i in merged:
                    continue

                # 找所有与 e1 相似的实体
                similar_entities = [e1]
                merge_candidates = []

                for j, e2 in enumerate(type_entities):
                    if i == j or j in merged:
                        continue

                    # 第一步：编辑距离筛选
                    if self._are_similar_by_distance(e1.name, e2.name, distance_threshold):
                        merge_candidates.append((j, e2))

                # 第二步：LLM 判断
                if use_llm and merge_candidates:
                    for j, e2 in merge_candidates:
                        should_merge, reason = self._should_merge_with_llm(e1, e2)
                        if should_merge:
                            similar_entities.append(e2)
                            merged.add(j)
                            logger.debug(f"合并实体: '{e1.name}' + '{e2.name}' | 理由: {reason}")
                            merge_count += 1
                else:
                    # 不使用 LLM 时，直接合并编辑距离相似的
                    for j, e2 in merge_candidates:
                        similar_entities.append(e2)
                        merged.add(j)
                        merge_count += 1

                # 合并相似实体（保留第一个，叠加描述）
                if len(similar_entities) > 1:
                    merged_entity = self._merge_entity_list(similar_entities)
                    resolved_entities.append(merged_entity)
                else:
                    resolved_entities.append(e1)

        logger.info(f"实体解析完成: 输出实体数={len(resolved_entities)}, 合并了 {merge_count} 个重复实体")

        return resolved_entities

    def _merge_entity_list(self, entities: List[ExtractedEntity]) -> ExtractedEntity:
        """合并多个实体为一个"""
        if len(entities) == 1:
            return entities[0]

        # 保留第一个实体作为基础
        base = entities[0]

        # 收集所有名称（用于别名）
        all_names = [e.name for e in entities]
        name_variants = ", ".join(all_names[1:]) if len(all_names) > 1 else ""

        # 合并描述
        all_descriptions = [e.description for e in entities if e.description]
        merged_description = base.description
        if len(all_descriptions) > 1:
            unique_descriptions = list(dict.fromkeys(all_descriptions))  # 去重
            merged_description = "; ".join(unique_descriptions[:3])  # 最多3个

        # 创建合并后的实体
        return ExtractedEntity(
            name=base.name,
            entity_type=base.entity_type,
            description=merged_description,
            attributes={
                **base.attributes,
                "aliases": all_names,
                "name_variants": name_variants
            }
        )
