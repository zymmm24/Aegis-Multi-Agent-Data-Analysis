"""
Knowledge Graph Builder - 自动从数据构建知识图谱

功能：
1. 实体识别：从表头和数据识别实体类型
2. 关系发现：识别外键、相关性、层级关系
3. 图谱构建：构建节点和边的图结构
4. 可视化：生成交互式HTML知识图谱

用途：
- 数据理解与探索
- 下游Agent分析依据
- 报告展示

Usage:
    from knowledge_graph_builder import KnowledgeGraphBuilder
    
    builder = KnowledgeGraphBuilder()
    kg = builder.build_from_dataframe(df, column_profiles)
    kg.visualize("output.html")
"""

import json
import logging
import re
import math
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np
import pandas as pd

# Optional visualization libraries
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    Network = None

# Import config
try:
    from config_loader import get_config
    _config = get_config()
except ImportError:
    _config = None

logger = logging.getLogger("KnowledgeGraphBuilder")


def _get_cfg(key: str, default: Any = None) -> Any:
    if _config is not None:
        return _config.get(key, default)
    return default


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Entity:
    """知识图谱中的实体（节点）"""
    id: str                          # 唯一标识
    name: str                        # 显示名称
    entity_type: str                 # 实体类型：table, column, value, type
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Relationship:
    """知识图谱中的关系（边）"""
    source_id: str                   # 源实体ID
    target_id: str                   # 目标实体ID
    relation_type: str               # 关系类型
    weight: float = 1.0              # 关系强度
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeGraph:
    """知识图谱"""
    name: str
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entity(self, entity: Entity):
        self.entities.append(entity)
    
    def add_relationship(self, relationship: Relationship):
        self.relationships.append(relationship)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "metadata": self.metadata,
            "statistics": {
                "entity_count": len(self.entities),
                "relationship_count": len(self.relationships),
                "entity_types": list(set(e.entity_type for e in self.entities)),
                "relation_types": list(set(r.relation_type for r in self.relationships))
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# =============================================================================
# Entity Type Detector
# =============================================================================

class EntityTypeDetector:
    """检测列的实体类型"""
    
    # 常见的ID列模式
    ID_PATTERNS = [
        r'.*_id$', r'.*Id$', r'.*ID$', r'^id$', r'^ID$',
        r'.*编号$', r'.*号$', r'^编号$', r'^序号$'
    ]
    
    # 常见的时间列模式
    TIME_PATTERNS = [
        r'.*date.*', r'.*time.*', r'.*_at$', r'.*_on$',
        r'.*日期.*', r'.*时间.*', r'^year$', r'^month$', r'^day$'
    ]
    
    # 常见的金额列模式
    AMOUNT_PATTERNS = [
        r'.*amount.*', r'.*price.*', r'.*cost.*', r'.*total.*',
        r'.*金额.*', r'.*价格.*', r'.*费用.*', r'.*合计.*'
    ]
    
    # 常见的名称列模式
    NAME_PATTERNS = [
        r'.*name.*', r'.*title.*', r'.*label.*',
        r'.*名称.*', r'.*姓名.*', r'.*标题.*'
    ]
    
    @classmethod
    def detect_column_type(cls, 
                           column_name: str, 
                           column_data: pd.Series,
                           column_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        检测列的语义类型
        
        Returns:
            {
                "semantic_type": str,  # id, datetime, numeric, categorical, text
                "is_primary_key": bool,
                "is_foreign_key": bool,
                "entity_reference": str or None,  # 引用的实体类型
                "confidence": float
            }
        """
        result = {
            "semantic_type": "unknown",
            "is_primary_key": False,
            "is_foreign_key": False,
            "entity_reference": None,
            "confidence": 0.5
        }
        
        col_lower = column_name.lower()
        
        # 1. 检测ID列
        for pattern in cls.ID_PATTERNS:
            if re.match(pattern, column_name, re.IGNORECASE):
                result["semantic_type"] = "id"
                result["confidence"] = 0.9
                
                # 判断是主键还是外键
                if col_lower in ['id', 'pk', 'primary_key', '编号', '序号']:
                    result["is_primary_key"] = True
                else:
                    result["is_foreign_key"] = True
                    # 提取引用实体名
                    entity_ref = cls._extract_entity_reference(column_name)
                    result["entity_reference"] = entity_ref
                break
        
        # 2. 检测金额列（优先于日期检测，避免数值被误判）
        if result["semantic_type"] == "unknown":
            for pattern in cls.AMOUNT_PATTERNS:
                if re.match(pattern, column_name, re.IGNORECASE):
                    result["semantic_type"] = "amount"
                    result["confidence"] = 0.85
                    break
        
        # 3. 检测时间列
        if result["semantic_type"] == "unknown":
            # 先检查列名是否匹配时间模式
            for pattern in cls.TIME_PATTERNS:
                if re.match(pattern, column_name, re.IGNORECASE):
                    result["semantic_type"] = "datetime"
                    result["confidence"] = 0.85
                    break
            
            # 如果列名不匹配，尝试解析数据（但要排除纯数值列）
            if result["semantic_type"] == "unknown":
                # 跳过数值类型列（避免将数字误判为日期）
                if not pd.api.types.is_numeric_dtype(column_data):
                    try:
                        sample = column_data.dropna().head(10)
                        if len(sample) > 0:
                            # 尝试解析，并验证结果确实是日期格式
                            parsed = pd.to_datetime(sample, errors='coerce')
                            valid_ratio = parsed.notna().sum() / len(sample)
                            # 至少80%能成功解析才认为是日期
                            if valid_ratio >= 0.8:
                                result["semantic_type"] = "datetime"
                                result["confidence"] = 0.7
                    except:
                        pass
        
        # 4. 检测名称列
        if result["semantic_type"] == "unknown":
            for pattern in cls.NAME_PATTERNS:
                if re.match(pattern, column_name, re.IGNORECASE):
                    result["semantic_type"] = "name"
                    result["confidence"] = 0.8
                    break
        
        # 5. 基于数据类型推断
        if result["semantic_type"] == "unknown":
            if column_profile:
                type_dist = column_profile.get("type_distribution", {})
                if type_dist:
                    dominant_type = max(type_dist.items(), key=lambda x: x[1])[0]
                    if dominant_type in ["integer", "float"]:
                        result["semantic_type"] = "numeric"
                        result["confidence"] = 0.7
                    elif dominant_type == "string":
                        # 判断是分类还是文本
                        unique_ratio = column_profile.get("unique_ratio", 1.0)
                        if unique_ratio < 0.3:
                            result["semantic_type"] = "categorical"
                        else:
                            result["semantic_type"] = "text"
                        result["confidence"] = 0.6
            else:
                # 直接从数据推断
                if pd.api.types.is_numeric_dtype(column_data):
                    result["semantic_type"] = "numeric"
                    result["confidence"] = 0.6
                else:
                    unique_ratio = column_data.nunique() / len(column_data) if len(column_data) > 0 else 1
                    if unique_ratio < 0.3:
                        result["semantic_type"] = "categorical"
                    else:
                        result["semantic_type"] = "text"
                    result["confidence"] = 0.5
        
        return result
    
    @classmethod
    def _extract_entity_reference(cls, column_name: str) -> str:
        """从列名提取引用的实体名称"""
        # user_id -> User
        # customer_id -> Customer
        # 用户ID -> 用户
        
        # 英文模式
        if column_name.lower().endswith('_id'):
            entity = column_name[:-3]
            return entity.title().replace('_', '')
        elif column_name.endswith('Id') or column_name.endswith('ID'):
            entity = column_name[:-2]
            return entity
        
        # 中文模式
        if column_name.endswith('ID') or column_name.endswith('Id'):
            return column_name[:-2]
        if column_name.endswith('编号') or column_name.endswith('号'):
            return column_name.replace('编号', '').replace('号', '')
        
        return column_name


# =============================================================================
# Relationship Detector
# =============================================================================

class RelationshipDetector:
    """发现数据中的关系"""
    
    @classmethod
    def detect_foreign_key_relationships(cls, 
                                         df: pd.DataFrame,
                                         column_types: Dict[str, Dict]) -> List[Relationship]:
        """检测外键关系"""
        relationships = []
        
        # 找出所有ID类型的列
        id_columns = {
            name: info for name, info in column_types.items()
            if info.get("semantic_type") == "id"
        }
        
        # 主键列
        primary_keys = [name for name, info in id_columns.items() if info.get("is_primary_key")]
        
        # 外键列
        foreign_keys = [
            (name, info) for name, info in id_columns.items() 
            if info.get("is_foreign_key")
        ]
        
        # 创建外键关系
        for fk_name, fk_info in foreign_keys:
            entity_ref = fk_info.get("entity_reference")
            if entity_ref:
                rel = Relationship(
                    source_id=f"column:{fk_name}",
                    target_id=f"entity:{entity_ref}",
                    relation_type="references",
                    weight=fk_info.get("confidence", 0.8),
                    properties={"foreign_key": fk_name}
                )
                relationships.append(rel)
        
        return relationships
    
    @classmethod
    def detect_correlation_relationships(cls,
                                         df: pd.DataFrame,
                                         threshold: float = 0.7) -> List[Relationship]:
        """检测数值列之间的相关性关系"""
        relationships = []
        
        # 获取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return relationships
        
        try:
            # 计算相关性矩阵
            corr_matrix = df[numeric_cols].corr()
            
            # 提取高相关性的列对
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) >= threshold:
                        rel = Relationship(
                            source_id=f"column:{col1}",
                            target_id=f"column:{col2}",
                            relation_type="correlates_with" if corr > 0 else "inversely_correlates",
                            weight=abs(corr),
                            properties={
                                "correlation": round(corr, 4),
                                "direction": "positive" if corr > 0 else "negative"
                            }
                        )
                        relationships.append(rel)
        except Exception as e:
            logger.warning(f"Correlation detection failed: {e}")
        
        return relationships
    
    @classmethod
    def detect_hierarchical_relationships(cls,
                                          df: pd.DataFrame,
                                          column_types: Dict[str, Dict]) -> List[Relationship]:
        """检测层级关系（如类别-子类别）"""
        relationships = []
        
        # 找出分类列
        categorical_cols = [
            name for name, info in column_types.items()
            if info.get("semantic_type") == "categorical"
        ]
        
        # 检测可能的层级关系
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                # 检查是否存在层级关系
                # 如果col2的每个值都只对应col1的一个值，可能是层级关系
                try:
                    grouped = df.groupby(col2)[col1].nunique()
                    if grouped.max() == 1:
                        # col2 是 col1 的子类别
                        rel = Relationship(
                            source_id=f"column:{col1}",
                            target_id=f"column:{col2}",
                            relation_type="has_subcategory",
                            weight=0.9,
                            properties={"hierarchy": "parent_child"}
                        )
                        relationships.append(rel)
                except:
                    pass
        
        return relationships
    
    @classmethod
    def detect_temporal_relationships(cls,
                                      df: pd.DataFrame,
                                      column_types: Dict[str, Dict]) -> List[Relationship]:
        """检测时间相关的关系"""
        relationships = []
        
        # 找出时间列
        time_cols = [
            name for name, info in column_types.items()
            if info.get("semantic_type") == "datetime"
        ]
        
        # 时间列与其他数值列可能有时序关系
        numeric_cols = [
            name for name, info in column_types.items()
            if info.get("semantic_type") in ["numeric", "amount"]
        ]
        
        for time_col in time_cols:
            for num_col in numeric_cols:
                rel = Relationship(
                    source_id=f"column:{time_col}",
                    target_id=f"column:{num_col}",
                    relation_type="temporal_context",
                    weight=0.7,
                    properties={"time_column": time_col, "value_column": num_col}
                )
                relationships.append(rel)
        
        return relationships


# =============================================================================
# Knowledge Graph Builder
# =============================================================================

class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, 
                 correlation_threshold: float = 0.7,
                 include_sample_values: bool = True,
                 max_sample_values: int = 5):
        """
        初始化构建器
        
        Args:
            correlation_threshold: 相关性阈值
            include_sample_values: 是否包含样本值作为节点
            max_sample_values: 每列最多包含的样本值数量
        """
        self.correlation_threshold = correlation_threshold
        self.include_sample_values = include_sample_values
        self.max_sample_values = max_sample_values
    
    def build_from_dataframe(self,
                             df: pd.DataFrame,
                             column_profiles: Dict[str, Any] = None,
                             table_name: str = "DataTable") -> KnowledgeGraph:
        """
        从DataFrame构建知识图谱
        
        Args:
            df: 输入数据
            column_profiles: L1层的列档案（可选）
            table_name: 表名称
        
        Returns:
            构建好的知识图谱
        """
        kg = KnowledgeGraph(name=f"KG_{table_name}")
        kg.metadata = {
            "source_table": table_name,
            "row_count": len(df),
            "column_count": len(df.columns)
        }
        
        # 1. 创建表实体（根节点）
        table_entity = Entity(
            id=f"table:{table_name}",
            name=table_name,
            entity_type="table",
            properties={
                "rows": len(df),
                "columns": len(df.columns)
            }
        )
        kg.add_entity(table_entity)
        
        # 2. 分析每列的类型
        column_types = {}
        for col in df.columns:
            profile = column_profiles.get(col, {}) if column_profiles else {}
            col_type = EntityTypeDetector.detect_column_type(col, df[col], profile)
            column_types[col] = col_type
        
        # 3. 为每列创建实体和关系
        for col, col_info in column_types.items():
            profile = column_profiles.get(col, {}) if column_profiles else {}
            
            # 列实体
            col_entity = Entity(
                id=f"column:{col}",
                name=col,
                entity_type="column",
                properties={
                    "semantic_type": col_info["semantic_type"],
                    "is_primary_key": col_info["is_primary_key"],
                    "is_foreign_key": col_info["is_foreign_key"],
                    "null_rate": profile.get("null_rate", df[col].isna().mean()),
                    "unique_ratio": profile.get("unique_ratio", df[col].nunique() / len(df) if len(df) > 0 else 0),
                    "data_type": str(df[col].dtype)
                }
            )
            kg.add_entity(col_entity)
            
            # 表到列的关系
            kg.add_relationship(Relationship(
                source_id=table_entity.id,
                target_id=col_entity.id,
                relation_type="has_column"
            ))
            
            # 如果是外键，创建引用实体
            if col_info["is_foreign_key"] and col_info["entity_reference"]:
                ref_entity_id = f"entity:{col_info['entity_reference']}"
                # 检查是否已存在
                if not kg.get_entity(ref_entity_id):
                    ref_entity = Entity(
                        id=ref_entity_id,
                        name=col_info["entity_reference"],
                        entity_type="referenced_entity",
                        properties={"inferred_from": col}
                    )
                    kg.add_entity(ref_entity)
            
            # 添加样本值节点（可选）
            if self.include_sample_values and col_info["semantic_type"] == "categorical":
                unique_values = df[col].dropna().unique()[:self.max_sample_values]
                for val in unique_values:
                    val_entity = Entity(
                        id=f"value:{col}:{val}",
                        name=str(val),
                        entity_type="value",
                        properties={"column": col}
                    )
                    kg.add_entity(val_entity)
                    kg.add_relationship(Relationship(
                        source_id=col_entity.id,
                        target_id=val_entity.id,
                        relation_type="has_value"
                    ))
        
        # 4. 检测列间关系
        # 外键关系
        fk_rels = RelationshipDetector.detect_foreign_key_relationships(df, column_types)
        for rel in fk_rels:
            kg.add_relationship(rel)
        
        # 相关性关系
        corr_rels = RelationshipDetector.detect_correlation_relationships(
            df, self.correlation_threshold
        )
        for rel in corr_rels:
            kg.add_relationship(rel)
        
        # 层级关系
        hier_rels = RelationshipDetector.detect_hierarchical_relationships(df, column_types)
        for rel in hier_rels:
            kg.add_relationship(rel)
        
        # 时序关系
        temp_rels = RelationshipDetector.detect_temporal_relationships(df, column_types)
        for rel in temp_rels:
            kg.add_relationship(rel)
        
        logger.info(f"Knowledge graph built: {len(kg.entities)} entities, "
                   f"{len(kg.relationships)} relationships")
        
        return kg
    
    def build_from_l1_output(self,
                             df: pd.DataFrame,
                             l1_metadata: Dict[str, Any]) -> KnowledgeGraph:
        """
        从L1层输出构建知识图谱
        
        Args:
            df: 处理后的DataFrame
            l1_metadata: L1层的元数据
        
        Returns:
            知识图谱
        """
        column_profiles = l1_metadata.get("column_profiles", {})
        table_name = l1_metadata.get("source_file", "DataTable")
        
        return self.build_from_dataframe(df, column_profiles, table_name)


# =============================================================================
# Knowledge Graph Visualizer
# =============================================================================

class KnowledgeGraphVisualizer:
    """知识图谱可视化"""
    
    # 实体类型的颜色映射
    ENTITY_COLORS = {
        "table": "#FF6B6B",       # 红色 - 表
        "column": "#4ECDC4",      # 青色 - 列
        "value": "#95E1D3",       # 浅绿 - 值
        "referenced_entity": "#F38181",  # 橙红 - 引用实体
        "type": "#AA96DA"         # 紫色 - 类型
    }
    
    # 关系类型的颜色映射
    RELATION_COLORS = {
        "has_column": "#666666",
        "references": "#FF6B6B",
        "correlates_with": "#4ECDC4",
        "inversely_correlates": "#F38181",
        "has_subcategory": "#AA96DA",
        "temporal_context": "#95E1D3",
        "has_value": "#CCCCCC"
    }
    
    @classmethod
    def to_networkx(cls, kg: KnowledgeGraph) -> 'nx.DiGraph':
        """转换为NetworkX图"""
        if not HAS_NETWORKX:
            raise ImportError("NetworkX not installed. Run: pip install networkx")
        
        G = nx.DiGraph()
        
        # 添加节点
        for entity in kg.entities:
            G.add_node(
                entity.id,
                label=entity.name,
                entity_type=entity.entity_type,
                color=cls.ENTITY_COLORS.get(entity.entity_type, "#888888"),
                **entity.properties
            )
        
        # 添加边
        for rel in kg.relationships:
            G.add_edge(
                rel.source_id,
                rel.target_id,
                relation_type=rel.relation_type,
                weight=rel.weight,
                color=cls.RELATION_COLORS.get(rel.relation_type, "#888888"),
                **rel.properties
            )
        
        return G
    
    @classmethod
    def visualize_pyvis(cls, 
                        kg: KnowledgeGraph,
                        output_path: str = "knowledge_graph.html",
                        height: str = "800px",
                        width: str = "100%",
                        notebook: bool = False) -> str:
        """
        使用Pyvis生成交互式HTML可视化
        
        Args:
            kg: 知识图谱
            output_path: 输出HTML文件路径
            height: 图高度
            width: 图宽度
            notebook: 是否在Jupyter中显示
        
        Returns:
            输出文件路径
        """
        if not HAS_PYVIS:
            raise ImportError("Pyvis not installed. Run: pip install pyvis")
        
        net = Network(height=height, width=width, directed=True, notebook=notebook)
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)
        
        # 添加节点
        for entity in kg.entities:
            color = cls.ENTITY_COLORS.get(entity.entity_type, "#888888")
            size = 30 if entity.entity_type == "table" else (20 if entity.entity_type == "column" else 15)
            
            # 构建title（悬停信息）
            title_parts = [f"<b>{entity.name}</b>", f"Type: {entity.entity_type}"]
            for k, v in entity.properties.items():
                if isinstance(v, float):
                    title_parts.append(f"{k}: {v:.4f}")
                else:
                    title_parts.append(f"{k}: {v}")
            title = "<br>".join(title_parts)
            
            net.add_node(
                entity.id,
                label=entity.name,
                title=title,
                color=color,
                size=size,
                shape="box" if entity.entity_type == "table" else "dot"
            )
        
        # 添加边
        for rel in kg.relationships:
            color = cls.RELATION_COLORS.get(rel.relation_type, "#888888")
            
            net.add_edge(
                rel.source_id,
                rel.target_id,
                title=rel.relation_type,
                color=color,
                width=rel.weight * 2,
                arrows="to"
            )
        
        # 生成HTML
        net.save_graph(output_path)
        logger.info(f"Knowledge graph visualization saved to: {output_path}")
        
        return output_path
    
    @classmethod
    def visualize_simple_html(cls,
                              kg: KnowledgeGraph,
                              output_path: str = "knowledge_graph.html") -> str:
        """
        生成简单的HTML可视化（不依赖Pyvis）
        使用D3.js进行渲染
        """
        html_template = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Knowledge Graph: {title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #1a1a2e; }}
        #graph {{ width: 100vw; height: 100vh; }}
        .node {{ cursor: pointer; }}
        .node text {{ font-size: 12px; fill: #fff; pointer-events: none; }}
        .link {{ stroke-opacity: 0.6; }}
        .link-label {{ font-size: 10px; fill: #888; }}
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: #fff;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
        }}
        #legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 8px;
            color: #fff;
        }}
        #legend h3 {{ margin: 0 0 10px 0; font-size: 14px; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; font-size: 12px; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 3px; margin-right: 8px; }}
        #title {{
            position: absolute;
            top: 20px;
            left: 20px;
            color: #fff;
            font-size: 24px;
            font-weight: bold;
        }}
        #stats {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            color: #888;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div id="title">{title}</div>
    <div id="graph"></div>
    <div id="legend">
        <h3>图例</h3>
        <div class="legend-item"><div class="legend-color" style="background:#FF6B6B"></div>表 (Table)</div>
        <div class="legend-item"><div class="legend-color" style="background:#4ECDC4"></div>列 (Column)</div>
        <div class="legend-item"><div class="legend-color" style="background:#95E1D3"></div>值 (Value)</div>
        <div class="legend-item"><div class="legend-color" style="background:#F38181"></div>引用实体 (Reference)</div>
    </div>
    <div id="stats">节点: {node_count} | 关系: {edge_count}</div>
    
    <script>
        const data = {graph_data};
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // 颜色映射
        const colorMap = {{
            "table": "#FF6B6B",
            "column": "#4ECDC4",
            "value": "#95E1D3",
            "referenced_entity": "#F38181"
        }};
        
        // 力导向图
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));
        
        // 绘制边
        const link = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke", "#666")
            .attr("stroke-width", d => d.weight * 2);
        
        // 绘制节点
        const node = svg.append("g")
            .selectAll("g")
            .data(data.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        node.append("circle")
            .attr("r", d => d.type === "table" ? 25 : (d.type === "column" ? 18 : 12))
            .attr("fill", d => colorMap[d.type] || "#888");
        
        node.append("text")
            .attr("dx", 0)
            .attr("dy", d => d.type === "table" ? 35 : 25)
            .attr("text-anchor", "middle")
            .text(d => d.label.length > 15 ? d.label.slice(0, 12) + "..." : d.label);
        
        // Tooltip
        const tooltip = d3.select("body").append("div").attr("class", "tooltip").style("opacity", 0);
        
        node.on("mouseover", function(event, d) {{
            tooltip.transition().duration(200).style("opacity", .9);
            let html = "<b>" + d.label + "</b><br>Type: " + d.type;
            if (d.properties) {{
                for (let key in d.properties) {{
                    html += "<br>" + key + ": " + d.properties[key];
                }}
            }}
            tooltip.html(html)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }})
        .on("mouseout", function() {{
            tooltip.transition().duration(500).style("opacity", 0);
        }});
        
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});
        
        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}
        
        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}
        
        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>'''
        
        # 准备D3.js格式的数据
        nodes = []
        for entity in kg.entities:
            nodes.append({
                "id": entity.id,
                "label": entity.name,
                "type": entity.entity_type,
                "properties": entity.properties
            })
        
        links = []
        for rel in kg.relationships:
            links.append({
                "source": rel.source_id,
                "target": rel.target_id,
                "type": rel.relation_type,
                "weight": rel.weight
            })
        
        graph_data = json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False)
        
        html = html_template.format(
            title=kg.name,
            graph_data=graph_data,
            node_count=len(nodes),
            edge_count=len(links)
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Knowledge graph visualization saved to: {output_path}")
        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================

def build_knowledge_graph(df: pd.DataFrame,
                          column_profiles: Dict[str, Any] = None,
                          table_name: str = "DataTable") -> KnowledgeGraph:
    """便捷函数：从DataFrame构建知识图谱"""
    builder = KnowledgeGraphBuilder()
    return builder.build_from_dataframe(df, column_profiles, table_name)


def visualize_knowledge_graph(kg: KnowledgeGraph,
                              output_path: str = "knowledge_graph.html",
                              use_pyvis: bool = None) -> str:
    """便捷函数：可视化知识图谱"""
    if use_pyvis is None:
        use_pyvis = HAS_PYVIS
    
    if use_pyvis and HAS_PYVIS:
        return KnowledgeGraphVisualizer.visualize_pyvis(kg, output_path)
    else:
        return KnowledgeGraphVisualizer.visualize_simple_html(kg, output_path)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    
    print("=" * 60)
    print("Knowledge Graph Builder Demo")
    print("=" * 60)
    print()
    
    # 创建示例数据
    df = pd.DataFrame({
        "order_id": [1001, 1002, 1003, 1004, 1005],
        "customer_id": ["C001", "C002", "C001", "C003", "C002"],
        "product_id": ["P001", "P002", "P003", "P001", "P002"],
        "amount": [100.50, 200.75, 150.25, 300.00, 250.50],
        "quantity": [1, 2, 1, 3, 2],
        "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "status": ["completed", "pending", "completed", "shipped", "completed"]
    })
    
    print("Sample Data:")
    print(df.to_string())
    print()
    
    # 构建知识图谱
    builder = KnowledgeGraphBuilder()
    kg = builder.build_from_dataframe(df, table_name="Orders")
    
    print("Knowledge Graph Summary:")
    print(f"  Entities: {len(kg.entities)}")
    print(f"  Relationships: {len(kg.relationships)}")
    print()
    
    print("Entities:")
    for entity in kg.entities:
        print(f"  [{entity.entity_type}] {entity.name}")
    print()
    
    print("Relationships:")
    for rel in kg.relationships:
        print(f"  {rel.source_id} --[{rel.relation_type}]--> {rel.target_id}")
    print()
    
    # 生成可视化
    output_path = visualize_knowledge_graph(kg, "knowledge_graph_demo.html")
    print(f"Visualization saved to: {output_path}")
    print()
    
    # 输出JSON
    print("JSON Output (first 500 chars):")
    print(kg.to_json()[:500] + "...")

