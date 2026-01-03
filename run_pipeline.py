"""
L1 + L2 集成测试脚本

完整流程：
1. L1: 自适应ETL（表头检测、异常检测、列档案）
2. L2: 双通道领域感知（LLM表头重测、知识图谱、领域分类）

Usage:
    python run_pipeline.py sample_mixed.csv
    python run_pipeline.py your_data.csv
"""

import sys
import os
import json
import logging
from pathlib import Path

import pandas as pd

# 添加 L1 和 L2 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "L1"))
sys.path.insert(0, str(Path(__file__).parent / "L2"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("Pipeline")


def run_l1(file_path: str) -> tuple:
    """
    运行L1层处理
    
    Returns:
        (df, metadata, raw_lines)
    """
    from adaptive_etl_l1 import AdaptiveETL
    
    logger.info("=" * 60)
    logger.info("L1: Adaptive ETL Layer")
    logger.info("=" * 60)
    
    etl = AdaptiveETL()
    
    # 读取原始行（用于L2的LLM审核）
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        raw_lines = f.readlines()[:20]  # 最多20行
    
    # 检测文件格式
    format_info = etl.detect_file_format(file_path)
    
    logger.info(f"Encoding: {format_info['encoding']}")
    logger.info(f"Delimiter: {repr(format_info['delimiter'])}")
    
    header_detection = format_info.get('header_detection', {})
    logger.info(f"Header Row: {header_detection.get('header_row', 0)}")
    logger.info(f"Confidence: {header_detection.get('confidence', 'N/A')}")
    logger.info(f"Needs Review: {header_detection.get('needs_review', False)}")
    
    # 处理文件
    output_dir = "ir_output"
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_count = 0
    df = None
    metadata = None
    
    for chunk_idx, chunk_df, chunk_meta in etl.process_file(file_path, output_dir=output_dir):
        chunk_count += 1
        if chunk_idx == 0:
            df = chunk_df
            metadata = chunk_meta
        
        logger.info(f"Chunk {chunk_idx}: {len(chunk_df)} rows, {len(chunk_df.columns)} columns")
    
    logger.info(f"L1 Complete: {chunk_count} chunks processed")
    logger.info("")
    
    return df, metadata, raw_lines


def run_l2(df: pd.DataFrame, l1_metadata: dict, raw_lines: list) -> dict:
    """
    运行L2层处理
    
    Returns:
        L2Result字典
    """
    from domain_perception_l2 import DomainPerceptionL2
    
    logger.info("=" * 60)
    logger.info("L2: Dual-Channel Domain Perception")
    logger.info("=" * 60)
    
    l2 = DomainPerceptionL2()
    
    # 显示L2能力
    caps = l2.get_capabilities()
    logger.info("Capabilities:")
    for cap, status in caps.items():
        symbol = "✓" if status else "✗"
        logger.info(f"  {cap}: {symbol}")
    logger.info("")
    
    # 获取L1异常分数
    anomaly_scores = None
    if "_anomaly_score" in df.columns:
        anomaly_scores = df["_anomaly_score"].tolist()
    
    # 处理
    table_name = Path(l1_metadata.get("source_file", "data")).stem
    result = l2.process(
        df=df,
        l1_metadata=l1_metadata,
        table_name=table_name,
        raw_lines=raw_lines,
        anomaly_scores=anomaly_scores
    )
    
    # 显示结果
    logger.info("Results:")
    
    # 表头审核
    if result.header_review:
        hr = result.header_review
        logger.info(f"  Header Review:")
        logger.info(f"    Original: row {hr.original_header_row}")
        logger.info(f"    Reviewed: row {hr.reviewed_header_row}")
        logger.info(f"    Changed: {hr.changed}")
        logger.info(f"    Source: {hr.review_source}")
    
    # 知识图谱
    if result.knowledge_graph:
        kg = result.knowledge_graph
        logger.info(f"  Knowledge Graph:")
        logger.info(f"    Entities: {len(kg.entities)}")
        logger.info(f"    Relationships: {len(kg.relationships)}")
        logger.info(f"    Visualization: {result.kg_visualization_path}")
    
    # 多标签属性向量
    logger.info(f"  Multi-Label Attribute Vector:")
    for label in result.attribute_labels[:5]:  # Top 5
        bar = "█" * int(label["confidence"] * 10)
        logger.info(f"    {label['attribute']:25} {label['confidence']:.2f} {bar}")
    
    logger.info(f"  Top Attributes: {result.top_attributes}")
    
    logger.info("")
    logger.info("L2 Complete")
    
    return result.to_dict()


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <csv_file>")
        print("Example: python run_pipeline.py sample_mixed.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Aegis: Multi-Agent Data Analysis Pipeline")
    print("=" * 60)
    print(f"Input: {file_path}")
    print()
    
    # L1处理
    df, l1_metadata, raw_lines = run_l1(file_path)
    
    if df is None or df.empty:
        print("Error: L1 produced no data")
        sys.exit(1)
    
    print("L1 Output DataFrame:")
    print(df.head(10).to_string())
    print()
    
    # L2处理
    l2_result = run_l2(df, l1_metadata, raw_lines)
    
    # 保存完整结果
    output_path = "ir_output/pipeline_result.json"
    
    # 准备输出（移除不可序列化的对象）
    output = {
        "l1_metadata": {
            k: v for k, v in l1_metadata.items() 
            if k not in ['header_candidates']  # 太大
        },
        "l2_result": {
            k: v for k, v in l2_result.items()
            if k not in ['knowledge_graph']  # 单独保存
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print()
    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print(f"Knowledge Graph: {l2_result.get('kg_visualization_path', 'N/A')}")
    print()
    
    # 总结
    header_detection = l1_metadata.get("header_detection", {})
    print("Summary:")
    print(f"  L1 Header Detection: row={header_detection.get('header_row', 0)}, "
          f"confidence={header_detection.get('confidence', 'N/A')}, "
          f"needs_review={header_detection.get('needs_review', False)}")
    
    header_review = l2_result.get("header_review")
    if header_review:
        print(f"  L2 Header Review: source={header_review.get('review_source', 'N/A')}, "
              f"changed={header_review.get('changed', False)}")
    
    # 多标签属性
    top_attrs = l2_result.get("top_attributes", [])
    print(f"  Top Data Attributes: {', '.join(top_attrs[:5])}")


if __name__ == "__main__":
    main()

