# Aegis: 基于混合特征融合与多 Agent 协同的自适应结构化数据分析系统

> **当前进度**: L1 ✅ 已完成 | L2 ✅ 已完成 | L3-L5 🚧 待开发

## 一、总体设计概述 (System Overview)

针对当前结构化数据分析领域存在的"多源异构数据标准化困难"、"分析模型泛化能力不足"以及"生成式报告幻觉风险"三大核心挑战，本课题提出了一种基于知识图谱引导与多智能体协同（Multi-Agent Collaboration）的自适应分析架构。

本系统采用分层递进式流水线（Hierarchical Pipeline）设计，通过集成自适应ETL、双通道特征融合、动态计算路由及逻辑仲裁机制，实现从原始异构数据到可信决策情报的端到端自动化处理。

## 二、项目结构

```
Aegis/
├── L1/                          # L1 自适应预处理层
│   ├── adaptive_etl_l1.py       # 核心ETL处理器
│   └── test_header_detection.py # L1单元测试
│
├── L2/                          # L2 双通道领域感知层
│   ├── domain_perception_l2.py  # 双通道感知处理器
│   ├── knowledge_graph_builder.py # 知识图谱构建器
│   └── test_l2_perception.py    # L2单元测试
│
├── config.yaml                  # 全局配置文件
├── config_loader.py             # 配置加载器
├── llm_client.py                # LLM统一接口
├── run_pipeline.py              # L1+L2 流水线入口
├── requirements.txt             # 依赖管理
└── ir_output/                   # 中间表示输出目录
    ├── chunk_*.csv              # 预处理后的数据
    ├── chunk_*_meta.json        # 元数据
    └── *_kg.html                # 知识图谱可视化
```

## 三、已完成功能

### L1. 自适应数据预处理层 ✅

| 功能模块 | 状态 | 说明 |
|---------|------|------|
| 智能编码检测 | ✅ | 自动探测 GBK/UTF-8 等编码格式 |
| 分隔符嗅探 | ✅ | 自动识别 CSV/TSV/分号等分隔符 |
| 表头智能检测 | ✅ | 基于统计启发式 + 首行优先策略 |
| 置信度评分 | ✅ | HIGH/MEDIUM/LOW/UNCERTAIN 四级置信度 |
| 集成异常检测 | ✅ | Z-Score + Isolation Forest + 缺失率融合 |
| 列类型推断 | ✅ | 自动推断数值/日期/分类/文本类型 |
| 中间表示输出 | ✅ | CSV + JSON 元数据 |

**运行方式**:
```bash
python L1/adaptive_etl_l1.py sample_mixed.csv
```

### L2. 双通道领域感知层 ✅

| 功能模块 | 状态 | 说明 |
|---------|------|------|
| 语义通道 | ✅ | Sentence-BERT 嵌入 + 属性原型匹配 |
| 统计通道 | ✅ | 15+ 维度统计特征提取 |
| 特征融合 | ✅ | Sigmoid激活 + 加权融合 |
| 多标签输出 | ✅ | 输出带置信度的数据属性向量 |
| 知识图谱 | ✅ | 自动从数据构建 + 交互式可视化 |
| LLM表头复审 | ✅ | 对低置信度表头进行LLM二次判断 |

**多标签属性体系**:
- `temporal_pattern`: 时序特性
- `numeric_density`: 数值密集性
- `sparsity`: 稀疏性
- `categorical_presence`: 分类特性
- `relational_structure`: 关联结构
- `financial_indicators`: 金融指标
- `pii_indicators`: 个人标识信息
- `high_cardinality`: 高基数特性
- `text_richness`: 文本丰富度
- `periodicity`: 周期性
- `distribution_skewed`: 分布偏斜
- `distribution_heavy_tail`: 重尾分布
- `anomaly_presence`: 异常存在性
- `hierarchical_structure`: 层级结构
- `geographical_indicators`: 地理指标

**运行方式**:
```bash
python L2/domain_perception_l2.py sample_mixed.csv
```

### 完整流水线

```bash
python run_pipeline.py sample_mixed.csv
```

## 四、配置说明

编辑 `config.yaml` 进行配置：

```yaml
# L1 配置
l1_etl:
  chunk_rows: 10000
  confidence_thresholds:
    high: 0.75
    medium: 0.50
    low: 0.30

# L2 配置
l2_perception:
  semantic_channel:
    model_name: "paraphrase-multilingual-MiniLM-L12-v2"
  fusion_weights:
    semantic: 0.5
    statistical: 0.5

# LLM 配置
llm:
  provider: "ollama"  # ollama | openai | azure
  ollama:
    base_url: "http://localhost:11434"
    model: "qwen2.5:7b"
```

## 五、安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖**:
- pandas, numpy, scipy
- scikit-learn (可选，用于 Isolation Forest)
- sentence-transformers (可选，用于语义通道)
- networkx, pyvis (用于知识图谱)
- pyyaml

## 六、待开发功能

### L3. 多Agent协同分析层 🚧
- 智能路由分发
- 时序预测 Agent (Prophet/ARIMA)
- 相关性分析 Agent
- 自适应模型降级

### L4. 决策仲裁与融合层 🚧
- 置信度博弈机制
- 逻辑一致性校验
- 冲突消解

### L5. 知识合成与报告生成层 🚧
- 幻觉抑制
- 槽位填充生成
- 证据锚点注入

## 七、技术亮点

1. **混合表头检测**: 统计启发式 + 首行优先 + LLM回退
2. **双通道融合**: 语义理解 + 统计形态 = 全方位数据建模
3. **多标签分类**: 识别数据复合属性，支持多标签输出
4. **自动知识图谱**: 从数据自动抽取实体关系并可视化
5. **非破坏性异常处理**: 软标记而非硬删除，保留原始分布

## 八、示例输出

```
======================================================================
Multi-Label Attribute Vector (数据特性多标签向量)
======================================================================

  temporal_pattern               0.83 ████████████████
  high_cardinality               0.63 ████████████
  numeric_density                0.52 ██████████
  financial_indicators           0.47 █████████
  sparsity                       0.35 ██████

Knowledge Graph: 6 entities, 6 relationships
Visualization: ir_output/sample_mixed_kg.html
```

---

**License**: MIT
