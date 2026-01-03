"""
L2: Dual-Channel Domain Perception Layer
双通道领域感知层

核心理念：
    从数据本身提取特征，识别数据的复合属性，输出带置信度的多标签向量。
    不是用预定义领域去"试"数据，而是用数据去分析其特性。

核心功能：
1. 语义通道：使用Sentence-BERT对列名和数据进行嵌入，提取语义特征
2. 统计通道：提取数据的统计形态特征（偏度/峰度/周期性/分布等）
3. 特征融合：Sigmoid激活 + 加权融合
4. 多标签输出：输出数据的多维属性标签向量，每个标签带置信度

属性标签体系：
    - temporal_pattern: 时序特性
    - numeric_density: 数值密集性
    - sparsity: 稀疏性
    - categorical_presence: 分类特性
    - relational_structure: 关联结构（外键/ID）
    - text_richness: 文本丰富性
    - distribution_anomaly: 分布异常（偏度/峰度）
    - periodicity: 周期性
    - high_cardinality: 高基数（唯一值多）
    - anomaly_presence: 异常值存在
    - hierarchical_structure: 层级结构
    - financial_indicators: 金融特征
    - geographical_indicators: 地理特征
    - pii_indicators: 个人信息特征

Usage:
    from domain_perception_l2 import DomainPerceptionL2
    
    l2 = DomainPerceptionL2()
    result = l2.process(df, l1_metadata)
    
    # 获取多标签向量
    for label in result.attribute_labels:
        print(f"{label.attribute}: {label.confidence:.2f}")
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft

# Optional: Sentence Transformers for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

# Import knowledge graph builder
try:
    from knowledge_graph_builder import (
        KnowledgeGraphBuilder, 
        KnowledgeGraph,
        visualize_knowledge_graph
    )
    HAS_KG_BUILDER = True
except ImportError:
    HAS_KG_BUILDER = False
    KnowledgeGraphBuilder = None
    KnowledgeGraph = None

# Import LLM client
try:
    from llm_client import LLMClient, LLMError, get_llm_client
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False
    LLMClient = None
    LLMError = Exception

# Import config
try:
    from config_loader import get_config
    _config = get_config()
except ImportError:
    _config = None

logger = logging.getLogger("DomainPerception_L2")


def _get_cfg(key: str, default: Any = None) -> Any:
    if _config is not None:
        return _config.get(key, default)
    return default


# =============================================================================
# Data Attribute Labels (数据属性标签体系)
# =============================================================================

class DataAttribute(str, Enum):
    """数据属性枚举"""
    # 结构特征
    TEMPORAL_PATTERN = "temporal_pattern"           # 时序特性
    NUMERIC_DENSITY = "numeric_density"             # 数值密集性
    SPARSITY = "sparsity"                          # 稀疏性（缺失值）
    CATEGORICAL_PRESENCE = "categorical_presence"   # 分类特性
    RELATIONAL_STRUCTURE = "relational_structure"   # 关联结构（外键/ID）
    TEXT_RICHNESS = "text_richness"                # 文本丰富性
    HIGH_CARDINALITY = "high_cardinality"          # 高基数
    
    # 分布特征
    DISTRIBUTION_SKEWED = "distribution_skewed"     # 分布偏斜
    DISTRIBUTION_HEAVY_TAIL = "distribution_heavy_tail"  # 重尾分布
    PERIODICITY = "periodicity"                    # 周期性
    ANOMALY_PRESENCE = "anomaly_presence"          # 异常值存在
    
    # 语义特征
    FINANCIAL_INDICATORS = "financial_indicators"   # 金融特征
    GEOGRAPHICAL_INDICATORS = "geographical_indicators"  # 地理特征
    PII_INDICATORS = "pii_indicators"              # 个人信息特征
    HIERARCHICAL_STRUCTURE = "hierarchical_structure"  # 层级结构


@dataclass
class AttributeLabel:
    """单个属性标签"""
    attribute: str          # 属性名称
    confidence: float       # 置信度 [0, 1]
    evidence: Dict[str, Any] = field(default_factory=dict)  # 支撑证据
    source: str = "fused"   # 来源: semantic, statistical, fused
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attribute": self.attribute,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "source": self.source
        }


@dataclass
class AttributeVector:
    """多标签属性向量"""
    labels: List[AttributeLabel] = field(default_factory=list)
    
    def add(self, attribute: str, confidence: float, evidence: Dict = None, source: str = "fused"):
        self.labels.append(AttributeLabel(
            attribute=attribute,
            confidence=confidence,
            evidence=evidence or {},
            source=source
        ))
    
    def get(self, attribute: str) -> Optional[AttributeLabel]:
        for label in self.labels:
            if label.attribute == attribute:
                return label
        return None
    
    def to_list(self) -> List[Dict[str, Any]]:
        return [l.to_dict() for l in sorted(self.labels, key=lambda x: -x.confidence)]
    
    def top_k(self, k: int = 5) -> List[AttributeLabel]:
        return sorted(self.labels, key=lambda x: -x.confidence)[:k]
    
    def above_threshold(self, threshold: float = 0.5) -> List[AttributeLabel]:
        return [l for l in self.labels if l.confidence >= threshold]


# =============================================================================
# Semantic Channel (语义通道)
# =============================================================================

class SemanticChannel:
    """
    语义通道：使用Sentence-BERT提取文本语义特征
    
    分析内容：
    - 列名的语义特征
    - 样本数据的文本特征
    - 语义模式匹配
    """
    
    # 语义模式关键词（用于增强检测）
    SEMANTIC_PATTERNS = {
        DataAttribute.FINANCIAL_INDICATORS: [
            "price", "amount", "balance", "transaction", "payment", "revenue",
            "profit", "cost", "fee", "tax", "currency", "rate", "interest",
            "价格", "金额", "余额", "交易", "支付", "收入", "利润", "费用"
        ],
        DataAttribute.GEOGRAPHICAL_INDICATORS: [
            "city", "country", "state", "province", "address", "location",
            "latitude", "longitude", "zip", "postal", "region", "area",
            "城市", "国家", "省份", "地址", "位置", "区域"
        ],
        DataAttribute.PII_INDICATORS: [
            "name", "email", "phone", "ssn", "passport", "id_card", "birth",
            "age", "gender", "address", "identity",
            "姓名", "邮箱", "电话", "身份证", "护照", "出生", "年龄", "性别"
        ],
        DataAttribute.TEMPORAL_PATTERN: [
            "date", "time", "timestamp", "datetime", "year", "month", "day",
            "hour", "minute", "created", "updated", "expired",
            "日期", "时间", "年", "月", "日", "时", "分", "秒"
        ]
    }
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or _get_cfg(
            "l2_perception.semantic_channel.embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.model = None
        self._initialized = False
    
    def _initialize(self):
        if self._initialized:
            return
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"Loading Sentence-BERT: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to load Sentence-BERT: {e}")
        
        self._initialized = True
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        分析DataFrame的语义特征
        
        Returns:
            {attribute: confidence} 语义通道的属性得分
        """
        self._initialize()
        scores = {attr.value: 0.0 for attr in DataAttribute}
        
        # 提取文本内容
        column_text = " ".join(df.columns.tolist()).lower()
        
        # 样本数据文本
        sample_texts = []
        for col in df.columns:
            sample = df[col].dropna().head(20).astype(str).tolist()
            sample_texts.extend(sample)
        value_text = " ".join(sample_texts).lower()
        
        combined_text = column_text + " " + value_text
        
        # 1. 基于关键词模式匹配
        for attr, keywords in self.SEMANTIC_PATTERNS.items():
            matches = sum(1 for kw in keywords if kw.lower() in combined_text)
            if matches > 0:
                # 归一化到 [0, 1]
                scores[attr.value] = min(1.0, matches / (len(keywords) * 0.15))
        
        # 2. 如果有Sentence-BERT，使用嵌入进行更深层语义分析
        if self.model is not None:
            semantic_features = self._analyze_with_embeddings(df, column_text)
            for attr, score in semantic_features.items():
                # 只更新已存在的属性键
                if attr in scores:
                    scores[attr] = max(scores[attr], score)
        
        # 3. 基于列名模式的附加检测
        pattern_scores = self._detect_column_patterns(df)
        for attr, score in pattern_scores.items():
            if attr in scores:
                scores[attr] = max(scores[attr], score)
        
        return scores
    
    def _analyze_with_embeddings(self, df: pd.DataFrame, column_text: str) -> Dict[str, float]:
        """使用Sentence-BERT嵌入进行语义分析"""
        features = {}
        
        try:
            # 对列名生成嵌入
            col_embeddings = self.model.encode(df.columns.tolist(), convert_to_numpy=True)
            
            # 计算列名之间的语义相似度（用于检测语义一致性）
            if len(col_embeddings) > 1:
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(col_embeddings)
                avg_similarity = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
                # 高语义相似度可能表示结构化数据
                features["semantic_consistency"] = float(avg_similarity)
        except Exception as e:
            logger.debug(f"Embedding analysis failed: {e}")
        
        return features
    
    def _detect_column_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """基于列名模式检测属性"""
        scores = {}
        columns_lower = [c.lower() for c in df.columns]
        
        # ID列检测 -> 关联结构
        id_patterns = ['_id', 'id_', '_key', 'key_', '_pk', '_fk']
        id_count = sum(1 for c in columns_lower if any(p in c for p in id_patterns))
        if id_count > 0:
            scores[DataAttribute.RELATIONAL_STRUCTURE.value] = min(1.0, id_count / len(df.columns))
        
        return scores


# =============================================================================
# Statistical Channel (统计通道)
# =============================================================================

class StatisticalChannel:
    """
    统计通道：提取数据的统计形态特征
    
    特征提取：
    - 类型分布（数值/分类/文本/时间）
    - 稀疏性（缺失率）
    - 分布特征（偏度、峰度）
    - 周期性（FFT分析）
    - 基数分析（唯一值比例）
    - 异常检测指标
    """
    
    def analyze(self, df: pd.DataFrame, anomaly_scores: List[float] = None) -> Dict[str, float]:
        """
        分析DataFrame的统计特征
        
        Args:
            df: 输入数据
            anomaly_scores: L1层的异常分数（可选）
        
        Returns:
            {attribute: confidence} 统计通道的属性得分
        """
        scores = {attr.value: 0.0 for attr in DataAttribute}
        n_cols = len(df.columns)
        n_rows = len(df)
        
        if n_cols == 0 or n_rows == 0:
            return scores
        
        # 1. 数值密集性
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scores[DataAttribute.NUMERIC_DENSITY.value] = len(numeric_cols) / n_cols
        
        # 2. 稀疏性（缺失率）
        missing_ratio = df.isna().sum().sum() / (n_cols * n_rows)
        scores[DataAttribute.SPARSITY.value] = missing_ratio
        
        # 3. 时序特性检测
        datetime_score = self._detect_temporal(df)
        scores[DataAttribute.TEMPORAL_PATTERN.value] = datetime_score
        
        # 4. 分类特性
        categorical_score = self._detect_categorical(df)
        scores[DataAttribute.CATEGORICAL_PRESENCE.value] = categorical_score
        
        # 5. 文本丰富性
        text_score = self._detect_text_richness(df)
        scores[DataAttribute.TEXT_RICHNESS.value] = text_score
        
        # 6. 高基数
        cardinality_score = self._detect_high_cardinality(df)
        scores[DataAttribute.HIGH_CARDINALITY.value] = cardinality_score
        
        # 7. 分布特征（偏度、峰度）
        skew_score, kurtosis_score = self._analyze_distribution(df)
        scores[DataAttribute.DISTRIBUTION_SKEWED.value] = skew_score
        scores[DataAttribute.DISTRIBUTION_HEAVY_TAIL.value] = kurtosis_score
        
        # 8. 周期性检测
        periodicity_score = self._detect_periodicity(df)
        scores[DataAttribute.PERIODICITY.value] = periodicity_score
        
        # 9. 异常值存在
        if anomaly_scores:
            high_anomaly_ratio = sum(1 for s in anomaly_scores if s > 0.7) / len(anomaly_scores)
            scores[DataAttribute.ANOMALY_PRESENCE.value] = high_anomaly_ratio
        else:
            scores[DataAttribute.ANOMALY_PRESENCE.value] = self._detect_anomalies(df)
        
        # 10. 层级结构检测
        hierarchical_score = self._detect_hierarchical(df)
        scores[DataAttribute.HIERARCHICAL_STRUCTURE.value] = hierarchical_score
        
        return scores
    
    def _detect_temporal(self, df: pd.DataFrame) -> float:
        """检测时序特性"""
        datetime_count = 0
        
        for col in df.columns:
            # 跳过数值列
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            try:
                sample = df[col].dropna().head(20)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    valid_ratio = parsed.notna().sum() / len(sample)
                    if valid_ratio >= 0.8:
                        datetime_count += 1
            except:
                pass
        
        return min(1.0, datetime_count / max(1, len(df.columns)))
    
    def _detect_categorical(self, df: pd.DataFrame) -> float:
        """检测分类特性"""
        categorical_count = 0
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1
                if unique_ratio < 0.3:  # 低唯一比例 = 分类变量
                    categorical_count += 1
        
        return categorical_count / max(1, len(df.columns))
    
    def _detect_text_richness(self, df: pd.DataFrame) -> float:
        """检测文本丰富性"""
        text_score = 0.0
        text_cols = 0
        
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(50).astype(str)
                if len(sample) > 0:
                    avg_length = sample.str.len().mean()
                    if avg_length > 50:  # 长文本
                        text_cols += 1
                        text_score += min(1.0, avg_length / 200)
        
        return text_score / max(1, len(df.columns))
    
    def _detect_high_cardinality(self, df: pd.DataFrame) -> float:
        """检测高基数"""
        high_card_count = 0
        
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1
            if unique_ratio > 0.8:  # 高唯一比例
                high_card_count += 1
        
        return high_card_count / max(1, len(df.columns))
    
    def _analyze_distribution(self, df: pd.DataFrame) -> Tuple[float, float]:
        """分析数值分布的偏度和峰度"""
        skewness_values = []
        kurtosis_values = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 10:
                try:
                    sk = stats.skew(data)
                    kt = stats.kurtosis(data)
                    skewness_values.append(abs(sk))
                    kurtosis_values.append(abs(kt))
                except:
                    pass
        
        # 偏度 > 1 表示显著偏斜
        skew_score = 0.0
        if skewness_values:
            avg_skew = np.mean(skewness_values)
            skew_score = min(1.0, avg_skew / 2.0)  # 归一化
        
        # 峰度 > 3 表示重尾（超额峰度 > 0）
        kurtosis_score = 0.0
        if kurtosis_values:
            avg_kurtosis = np.mean(kurtosis_values)
            kurtosis_score = min(1.0, max(0, avg_kurtosis) / 5.0)
        
        return skew_score, kurtosis_score
    
    def _detect_periodicity(self, df: pd.DataFrame) -> float:
        """使用FFT检测周期性"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0 or len(df) < 20:
            return 0.0
        
        periodicity_scores = []
        
        for col in numeric_cols[:5]:  # 最多分析5列
            data = df[col].dropna().values
            if len(data) < 20:
                continue
            
            try:
                # 去均值
                data = data - np.mean(data)
                
                # FFT
                fft_result = np.abs(fft(data))
                
                # 找主频率（排除DC分量）
                half_len = len(fft_result) // 2
                fft_half = fft_result[1:half_len]
                
                if len(fft_half) > 0:
                    max_power = np.max(fft_half)
                    avg_power = np.mean(fft_half)
                    
                    # 如果主频率显著高于平均，可能有周期性
                    if avg_power > 0:
                        periodicity = max_power / (avg_power * 5)
                        periodicity_scores.append(min(1.0, periodicity))
            except:
                pass
        
        return np.mean(periodicity_scores) if periodicity_scores else 0.0
    
    def _detect_anomalies(self, df: pd.DataFrame) -> float:
        """简单异常检测"""
        anomaly_score = 0.0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 10:
                try:
                    z_scores = np.abs(stats.zscore(data))
                    outlier_ratio = np.sum(z_scores > 3) / len(data)
                    anomaly_score = max(anomaly_score, outlier_ratio)
                except:
                    pass
        
        return min(1.0, anomaly_score * 10)  # 放大
    
    def _detect_hierarchical(self, df: pd.DataFrame) -> float:
        """检测层级结构"""
        categorical_cols = [
            col for col in df.columns 
            if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.3
        ]
        
        if len(categorical_cols) < 2:
            return 0.0
        
        hierarchical_pairs = 0
        
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                try:
                    grouped = df.groupby(col2)[col1].nunique()
                    if grouped.max() == 1:
                        hierarchical_pairs += 1
                except:
                    pass
        
        max_pairs = len(categorical_cols) * (len(categorical_cols) - 1) / 2
        return hierarchical_pairs / max(1, max_pairs)


# =============================================================================
# Feature Fusion (特征融合)
# =============================================================================

class FeatureFusion:
    """
    特征融合：将语义通道和统计通道的结果融合
    
    使用Sigmoid激活 + 加权平均
    """
    
    def __init__(self, 
                 semantic_weight: float = 0.5,
                 statistical_weight: float = 0.5,
                 threshold: float = 0.15):  # 降低阈值
        self.semantic_weight = semantic_weight
        self.statistical_weight = statistical_weight
        self.threshold = threshold
    
    @staticmethod
    def sigmoid(x: float, k: float = 3.0) -> float:
        """Sigmoid激活，k控制陡峭程度，中心点在0.3"""
        # 调整中心点，使得低分数也能有合理的输出
        return 1.0 / (1.0 + math.exp(-k * (x - 0.3)))
    
    def fuse(self,
             semantic_scores: Dict[str, float],
             statistical_scores: Dict[str, float]) -> AttributeVector:
        """
        融合双通道得分，输出多标签向量
        
        Args:
            semantic_scores: 语义通道得分
            statistical_scores: 统计通道得分
        
        Returns:
            AttributeVector 多标签属性向量
        """
        result = AttributeVector()
        
        all_attributes = set(semantic_scores.keys()) | set(statistical_scores.keys())
        
        for attr in all_attributes:
            sem_score = semantic_scores.get(attr, 0)
            stat_score = statistical_scores.get(attr, 0)
            
            # 加权融合（取两者的最大值作为基础，避免低分拉低高分）
            max_score = max(sem_score, stat_score)
            avg_score = (self.semantic_weight * sem_score + 
                        self.statistical_weight * stat_score)
            
            # 融合策略：60% 最大值 + 40% 加权平均
            fused = 0.6 * max_score + 0.4 * avg_score
            
            # Sigmoid激活（平滑输出）
            confidence = self.sigmoid(fused)
            
            # 只保留超过阈值的标签
            if confidence >= self.threshold:
                result.add(
                    attribute=attr,
                    confidence=confidence,
                    evidence={
                        "semantic_score": round(sem_score, 4),
                        "statistical_score": round(stat_score, 4),
                        "raw_fused": round(fused, 4)
                    },
                    source="fused"
                )
        
        return result


# =============================================================================
# L2 Result
# =============================================================================

@dataclass
class HeaderReviewResult:
    """LLM表头重测结果"""
    original_header_row: int = 0
    reviewed_header_row: int = 0
    changed: bool = False
    llm_confidence: str = "N/A"
    llm_reasoning: str = ""
    review_source: str = "skipped"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class L2Result:
    """L2层处理结果"""
    # 核心输出：多标签属性向量
    attribute_labels: List[Dict[str, Any]] = field(default_factory=list)
    
    # 原始得分
    semantic_scores: Dict[str, float] = field(default_factory=dict)
    statistical_scores: Dict[str, float] = field(default_factory=dict)
    
    # LLM表头重测
    header_review: Optional[HeaderReviewResult] = None
    
    # 知识图谱
    knowledge_graph: Optional[Any] = None
    kg_visualization_path: Optional[str] = None
    
    # 元信息
    top_attributes: List[str] = field(default_factory=list)
    processing_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attribute_labels": self.attribute_labels,
            "top_attributes": self.top_attributes,
            "semantic_scores": {k: round(v, 4) for k, v in self.semantic_scores.items()},
            "statistical_scores": {k: round(v, 4) for k, v in self.statistical_scores.items()},
            "header_review": self.header_review.to_dict() if self.header_review else None,
            "knowledge_graph_summary": {
                "entities": len(self.knowledge_graph.entities) if self.knowledge_graph else 0,
                "relationships": len(self.knowledge_graph.relationships) if self.knowledge_graph else 0
            },
            "kg_visualization_path": self.kg_visualization_path,
            "processing_info": self.processing_info
        }


# =============================================================================
# L2 Main Processor
# =============================================================================

class DomainPerceptionL2:
    """
    L2层：双通道领域感知
    
    核心功能：
    1. 从数据中提取语义特征和统计特征
    2. 融合双通道，输出多标签属性向量
    3. 构建知识图谱
    4. LLM辅助表头重测（对L1低置信度结果）
    """
    
    HEADER_REVIEW_SYSTEM_PROMPT = """You are a CSV data analyst. Identify the header row.
Rules:
1. Headers are descriptive text labels (like "name", "date", "amount")
2. Data rows contain actual values
3. Comment lines start with #
Respond in JSON: {"header_row": <number>, "confidence": "HIGH|MEDIUM|LOW", "reasoning": "<brief>"}"""
    
    def __init__(self,
                 enable_semantic: bool = True,
                 enable_statistical: bool = True,
                 enable_knowledge_graph: bool = True,
                 enable_llm_review: bool = True,
                 output_dir: str = None):
        
        self.enable_semantic = enable_semantic
        self.enable_statistical = enable_statistical
        self.enable_knowledge_graph = enable_knowledge_graph and HAS_KG_BUILDER
        self.enable_llm_review = enable_llm_review and HAS_LLM_CLIENT
        
        self.output_dir = output_dir or _get_cfg("l1_etl.output_dir", "ir_output")
        
        # 初始化通道
        self.semantic_channel = SemanticChannel() if enable_semantic else None
        self.statistical_channel = StatisticalChannel() if enable_statistical else None
        self.fusion = FeatureFusion()
        self.kg_builder = KnowledgeGraphBuilder() if self.enable_knowledge_graph else None
        
        # LLM
        self._llm_client = None
        self._llm_available = None
        
        logger.info(f"L2 initialized: semantic={self.enable_semantic}, "
                   f"statistical={self.enable_statistical}, "
                   f"kg={self.enable_knowledge_graph}, "
                   f"llm={self.enable_llm_review}")
    
    def process(self,
                df: pd.DataFrame,
                l1_metadata: Dict[str, Any] = None,
                table_name: str = None,
                raw_lines: List[str] = None,
                anomaly_scores: List[float] = None) -> L2Result:
        """
        完整的L2处理流程
        
        Args:
            df: 输入DataFrame
            l1_metadata: L1层元数据
            table_name: 表名称
            raw_lines: 原始CSV行（用于LLM）
            anomaly_scores: L1层异常分数
        
        Returns:
            L2Result
        """
        result = L2Result()
        result.processing_info = {"capabilities": self.get_capabilities()}
        
        # 表名
        if table_name is None:
            table_name = l1_metadata.get("source_file", "data") if l1_metadata else "data"
        
        # 0. LLM表头重测
        if self.enable_llm_review and l1_metadata:
            result.header_review = self._review_header(l1_metadata, raw_lines)
        
        # 1. 语义通道分析
        if self.semantic_channel:
            logger.info("Running semantic channel...")
            result.semantic_scores = self.semantic_channel.analyze(df)
        
        # 2. 统计通道分析
        if self.statistical_channel:
            logger.info("Running statistical channel...")
            result.statistical_scores = self.statistical_channel.analyze(df, anomaly_scores)
        
        # 3. 特征融合 -> 多标签向量
        if result.semantic_scores or result.statistical_scores:
            semantic = result.semantic_scores or {}
            statistical = result.statistical_scores or {}
            
            # 确保两个字典有相同的键
            all_keys = set(semantic.keys()) | set(statistical.keys())
            semantic = {k: semantic.get(k, 0) for k in all_keys}
            statistical = {k: statistical.get(k, 0) for k in all_keys}
            
            attribute_vector = self.fusion.fuse(semantic, statistical)
            result.attribute_labels = attribute_vector.to_list()
            result.top_attributes = [l.attribute for l in attribute_vector.top_k(5)]
        
        # 4. 知识图谱
        if self.enable_knowledge_graph and self.kg_builder:
            try:
                logger.info("Building knowledge graph...")
                column_profiles = l1_metadata.get("column_profiles", {}) if l1_metadata else {}
                kg = self.kg_builder.build_from_dataframe(df, column_profiles, table_name)
                result.knowledge_graph = kg
                
                import os
                os.makedirs(self.output_dir, exist_ok=True)
                output_path = os.path.join(self.output_dir, f"{table_name}_kg.html")
                result.kg_visualization_path = visualize_knowledge_graph(kg, output_path)
            except Exception as e:
                logger.error(f"KG building failed: {e}")
        
        logger.info(f"L2 complete. Top attributes: {result.top_attributes}")
        
        return result
    
    def _review_header(self, l1_metadata: Dict, raw_lines: List[str] = None) -> HeaderReviewResult:
        """LLM表头重测"""
        header_detection = l1_metadata.get("header_detection", {})
        original_row = header_detection.get("header_row", 0)
        needs_review = header_detection.get("needs_review", False)
        
        if not needs_review:
            return HeaderReviewResult(
                original_header_row=original_row,
                reviewed_header_row=original_row,
                changed=False,
                review_source="skipped"
            )
        
        if not self._is_llm_available():
            return HeaderReviewResult(
                original_header_row=original_row,
                reviewed_header_row=original_row,
                changed=False,
                llm_reasoning="LLM not available",
                review_source="fallback"
            )
        
        try:
            prompt = self._build_header_prompt(header_detection, raw_lines)
            client = self._get_llm()
            response = client.chat(prompt, self.HEADER_REVIEW_SYSTEM_PROMPT, temperature=0.1)
            return self._parse_header_response(response, original_row)
        except Exception as e:
            logger.error(f"LLM review failed: {e}")
            return HeaderReviewResult(
                original_header_row=original_row,
                reviewed_header_row=original_row,
                changed=False,
                llm_reasoning=str(e),
                review_source="fallback"
            )
    
    def _build_header_prompt(self, header_detection: Dict, raw_lines: List[str] = None) -> str:
        parts = ["Analyze this CSV:\n"]
        if raw_lines:
            for i, line in enumerate(raw_lines[:10]):
                parts.append(f"Row {i}: {line.strip()}")
        candidates = header_detection.get("candidate_summary", [])
        if candidates:
            parts.append("\nL1 candidates:")
            for c in candidates:
                parts.append(f"  Row {c['candidate']}: score={c['score']:.3f}")
        return "\n".join(parts)
    
    def _parse_header_response(self, response: str, fallback: int) -> HeaderReviewResult:
        try:
            match = re.search(r'\{[^{}]*\}', response)
            data = json.loads(match.group()) if match else json.loads(response)
            row = int(data.get("header_row", fallback))
            return HeaderReviewResult(
                original_header_row=fallback,
                reviewed_header_row=row,
                changed=(row != fallback),
                llm_confidence=data.get("confidence", "MEDIUM"),
                llm_reasoning=data.get("reasoning", ""),
                review_source="llm"
            )
        except:
            return HeaderReviewResult(
                original_header_row=fallback,
                reviewed_header_row=fallback,
                changed=False,
                review_source="fallback"
            )
    
    def _get_llm(self):
        if self._llm_client is None and HAS_LLM_CLIENT:
            self._llm_client = get_llm_client()
        return self._llm_client
    
    def _is_llm_available(self) -> bool:
        if self._llm_available is None:
            client = self._get_llm()
            self._llm_available = client.is_available() if client else False
        return self._llm_available
    
    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "semantic_channel": self.enable_semantic,
            "statistical_channel": self.enable_statistical,
            "knowledge_graph": self.enable_knowledge_graph,
            "llm_review": self.enable_llm_review,
            "llm_available": self._is_llm_available() if self.enable_llm_review else False,
            "sentence_transformers": HAS_SENTENCE_TRANSFORMERS
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    
    print("=" * 70)
    print("L2: Dual-Channel Domain Perception - Multi-Label Attribute Analysis")
    print("=" * 70)
    print()
    
    # 支持命令行参数读取文件
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        table_name = Path(csv_path).stem
        print(f"Loading: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        # 默认使用 sample_mixed.csv（如果存在）
        default_file = Path("sample_mixed.csv")
        if default_file.exists():
            print(f"Loading default: {default_file}")
            df = pd.read_csv(default_file)
            table_name = "sample_mixed"
        else:
            # 如果没有文件，使用内置演示数据
            print("Using built-in demo data (no input file specified)")
            df = pd.DataFrame({
                "order_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
                "customer_id": ["C001", "C002", "C001", "C003", "C002", "C004", "C001", "C005"],
                "product_name": ["Laptop", "Phone", "Tablet", "Watch", "Headphones", "Camera", "Mouse", "Keyboard"],
                "amount": [999.99, 599.50, 449.00, 299.99, 149.50, 799.00, 29.99, 89.99],
                "quantity": [1, 2, 1, 1, 3, 1, 2, 1],
                "order_date": pd.date_range("2024-01-01", periods=8, freq="D"),
                "status": ["completed", "pending", "completed", "shipped", "completed", "pending", "completed", "shipped"],
                "category": ["Electronics", "Electronics", "Electronics", "Wearable", "Audio", "Electronics", "Accessories", "Accessories"]
            })
            table_name = "demo_orders"
    
    print()
    print("Data Preview:")
    print(df.head(10).to_string())
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print()
    
    # L2处理
    l2 = DomainPerceptionL2()
    
    print("Capabilities:")
    for k, v in l2.get_capabilities().items():
        print(f"  {k}: {'✓' if v else '✗'}")
    print()
    
    result = l2.process(df, table_name=table_name)
    
    print("=" * 70)
    print("Multi-Label Attribute Vector (数据特性多标签向量)")
    print("=" * 70)
    print()
    
    for label in result.attribute_labels:
        bar = "█" * int(label["confidence"] * 20)
        print(f"  {label['attribute']:30} {label['confidence']:.2f} {bar}")
    
    print()
    print("Top 5 Attributes:", result.top_attributes)
    print()
    
    print("Semantic Scores:")
    for attr, score in sorted(result.semantic_scores.items(), key=lambda x: -x[1])[:5]:
        print(f"  {attr}: {score:.3f}")
    
    print()
    print("Statistical Scores:")
    for attr, score in sorted(result.statistical_scores.items(), key=lambda x: -x[1])[:5]:
        print(f"  {attr}: {score:.3f}")
    
    if result.knowledge_graph:
        print()
        print(f"Knowledge Graph: {len(result.knowledge_graph.entities)} entities, "
              f"{len(result.knowledge_graph.relationships)} relationships")
        print(f"Visualization: {result.kg_visualization_path}")
