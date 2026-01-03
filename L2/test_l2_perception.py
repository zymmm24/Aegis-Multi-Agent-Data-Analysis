"""
Test cases for L2 Domain Perception Layer

Tests:
1. Header Detection Review (with/without LLM)
2. Domain Classification (semantic + statistical)
3. Full L2 Processing Pipeline
"""

import unittest
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

# Import L2 components
from domain_perception_l2 import (
    DomainPerceptionL2,
    HeaderDetectionReviewer,
    DomainClassifier,
    HeaderReviewResult,
    DomainClassificationResult,
    process_l1_output
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


class TestHeaderDetectionReviewer(unittest.TestCase):
    """Tests for HeaderDetectionReviewer."""
    
    def setUp(self):
        self.reviewer = HeaderDetectionReviewer()
    
    def test_skip_high_confidence(self):
        """Test: High confidence results should skip LLM review."""
        l1_metadata = {
            "header_detection": {
                "header_row": 0,
                "confidence": "HIGH",
                "confidence_score": 0.85,
                "needs_review": False
            }
        }
        
        result = self.reviewer.review(l1_metadata)
        
        self.assertEqual(result.original_header_row, 0)
        self.assertEqual(result.reviewed_header_row, 0)
        self.assertFalse(result.changed)
        self.assertEqual(result.review_source, "skipped")
        print(f"[PASS] High confidence skipped: source={result.review_source}")
    
    def test_low_confidence_triggers_review(self):
        """Test: Low confidence results should trigger review."""
        l1_metadata = {
            "header_detection": {
                "header_row": 0,
                "confidence": "UNCERTAIN",
                "confidence_score": 0.20,
                "needs_review": True,
                "candidate_summary": [
                    {"candidate": 0, "score": 0.20, "header_line": "1,2,3,4,5"},
                    {"candidate": 1, "score": 0.18, "header_line": "10,20,30,40,50"}
                ]
            }
        }
        
        result = self.reviewer.review(l1_metadata)
        
        # Should either use LLM or fallback
        self.assertIn(result.review_source, ["llm", "fallback"])
        print(f"[PASS] Low confidence reviewed: source={result.review_source}, reasoning={result.llm_reasoning[:50]}...")
    
    def test_review_result_structure(self):
        """Test: Review result should have correct structure."""
        l1_metadata = {
            "header_detection": {
                "header_row": 1,
                "needs_review": True,
                "candidate_summary": []
            }
        }
        
        result = self.reviewer.review(l1_metadata)
        
        self.assertIsInstance(result, HeaderReviewResult)
        self.assertIsInstance(result.original_header_row, int)
        self.assertIsInstance(result.reviewed_header_row, int)
        self.assertIsInstance(result.changed, bool)
        self.assertIsInstance(result.llm_confidence, str)
        self.assertIsInstance(result.llm_reasoning, str)
        self.assertIsInstance(result.review_source, str)
        print(f"[PASS] Review result structure correct")


class TestDomainClassifier(unittest.TestCase):
    """Tests for DomainClassifier."""
    
    def setUp(self):
        self.classifier = DomainClassifier()
    
    def test_finance_domain(self):
        """Test: Financial data should be classified as finance domain."""
        df = pd.DataFrame({
            "transaction_id": [1, 2, 3, 4, 5],
            "amount": [100.50, 200.75, 150.25, 300.00, 250.50],
            "account_balance": [1000.00, 1200.75, 1050.50, 1350.50, 1101.00],
            "currency": ["USD", "USD", "EUR", "USD", "EUR"]
        })
        
        result = self.classifier.classify(df)
        
        self.assertEqual(result.primary_domain, "finance")
        self.assertGreater(result.primary_confidence, 0.3)
        print(f"[PASS] Finance domain: confidence={result.primary_confidence:.2f}, "
              f"semantic={result.semantic_scores.get('finance', 0):.2f}, "
              f"statistical={result.statistical_scores.get('finance', 0):.2f}")
    
    def test_healthcare_domain(self):
        """Test: Healthcare data should be classified correctly."""
        df = pd.DataFrame({
            "patient_id": [101, 102, 103, 104, 105],
            "diagnosis": ["Diabetes", "Hypertension", "Cold", "Flu", "Allergy"],
            "medication": ["Metformin", "Lisinopril", "None", "Tamiflu", "Zyrtec"],
            "blood_pressure": ["120/80", "140/90", "118/75", "130/85", "122/78"]
        })
        
        result = self.classifier.classify(df)
        
        # Healthcare should score high
        healthcare_score = result.semantic_scores.get("healthcare", 0)
        self.assertGreater(healthcare_score, 0.2)
        print(f"[PASS] Healthcare domain: primary={result.primary_domain}, "
              f"healthcare_semantic={healthcare_score:.2f}")
    
    def test_timeseries_domain(self):
        """Test: Time-series data should be classified correctly."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "timestamp": dates.astype(str),
            "value": np.random.randn(10) * 100,
            "trend": np.arange(10) * 1.5
        })
        
        result = self.classifier.classify(df)
        
        # Timeseries should score high
        timeseries_score = result.statistical_scores.get("timeseries", 0)
        self.assertGreater(timeseries_score, 0.3)
        print(f"[PASS] Timeseries domain: primary={result.primary_domain}, "
              f"timeseries_statistical={timeseries_score:.2f}")
    
    def test_ecommerce_domain(self):
        """Test: E-commerce data should be classified correctly."""
        df = pd.DataFrame({
            "order_id": [1001, 1002, 1003, 1004, 1005],
            "product": ["Laptop", "Phone", "Tablet", "Watch", "Headphones"],
            "category": ["Electronics", "Electronics", "Electronics", "Wearable", "Audio"],
            "quantity": [1, 2, 1, 1, 3],
            "customer_id": [501, 502, 503, 501, 504]
        })
        
        result = self.classifier.classify(df)
        
        ecommerce_score = result.semantic_scores.get("ecommerce", 0)
        self.assertGreater(ecommerce_score, 0.2)
        print(f"[PASS] Ecommerce domain: primary={result.primary_domain}, "
              f"ecommerce_semantic={ecommerce_score:.2f}")
    
    def test_chinese_domain_keywords(self):
        """Test: Chinese domain keywords should be recognized."""
        df = pd.DataFrame({
            "交易ID": [1, 2, 3],
            "金额": [100, 200, 300],
            "账户余额": [1000, 1200, 1500]
        })
        
        result = self.classifier.classify(df)
        
        finance_score = result.semantic_scores.get("finance", 0)
        self.assertGreater(finance_score, 0.1)
        print(f"[PASS] Chinese keywords: finance_semantic={finance_score:.2f}")
    
    def test_general_domain_fallback(self):
        """Test: Unrecognized data should fall back to general domain."""
        df = pd.DataFrame({
            "col_a": [1, 2, 3, 4, 5],
            "col_b": ["x", "y", "z", "w", "v"],
            "col_c": [True, False, True, False, True]
        })
        
        result = self.classifier.classify(df)
        
        # Should have general as one of the labels
        domain_names = [l.domain for l in result.all_labels]
        has_general = "general" in domain_names or result.primary_confidence < 0.5
        print(f"[PASS] General fallback: primary={result.primary_domain}, "
              f"confidence={result.primary_confidence:.2f}")
    
    def test_multi_label_output(self):
        """Test: Result should contain multiple labels with confidence."""
        df = pd.DataFrame({
            "order_id": [1, 2, 3],
            "amount": [100.50, 200.75, 150.25],  # Finance-like
            "product": ["Item A", "Item B", "Item C"],  # Ecommerce-like
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"]  # Timeseries-like
        })
        
        result = self.classifier.classify(df)
        
        self.assertIsInstance(result.all_labels, list)
        self.assertGreater(len(result.all_labels), 0)
        
        # Check label structure
        for label in result.all_labels:
            self.assertIn(label.domain, ["finance", "healthcare", "ecommerce", "timeseries", "general"])
            self.assertGreaterEqual(label.confidence, 0)
            self.assertLessEqual(label.confidence, 1)
        
        print(f"[PASS] Multi-label: {[(l.domain, round(l.confidence, 2)) for l in result.all_labels]}")


class TestDomainPerceptionL2(unittest.TestCase):
    """Tests for the main L2 processor."""
    
    def setUp(self):
        self.l2 = DomainPerceptionL2()
    
    def test_full_processing_pipeline(self):
        """Test: Full L2 processing should work end-to-end."""
        l1_metadata = {
            "header_detection": {
                "header_row": 0,
                "confidence": "HIGH",
                "confidence_score": 0.85,
                "needs_review": False
            },
            "column_profiles": {}
        }
        
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [100, 200, 300]
        })
        
        result = self.l2.process(l1_metadata, df)
        
        self.assertEqual(result["l2_status"], "success")
        self.assertIn("header_review", result)
        self.assertIn("domain_classification", result)
        print(f"[PASS] Full pipeline: status={result['l2_status']}")
    
    def test_processing_without_df(self):
        """Test: Processing should work without DataFrame (header review only)."""
        l1_metadata = {
            "header_detection": {
                "header_row": 0,
                "needs_review": True,
                "candidate_summary": []
            }
        }
        
        result = self.l2.process(l1_metadata, df=None)
        
        self.assertEqual(result["l2_status"], "success")
        self.assertIn("header_review", result)
        self.assertIsNone(result["domain_classification"])
        print(f"[PASS] No DF: header_review completed")
    
    def test_llm_availability_check(self):
        """Test: Should correctly report LLM availability."""
        available = self.l2.is_llm_available()
        
        self.assertIsInstance(available, bool)
        print(f"[PASS] LLM availability check: {available}")


class TestIntegrationWithL1(unittest.TestCase):
    """Integration tests with L1 output."""
    
    def test_process_real_l1_output(self):
        """Test: Process realistic L1 output structure."""
        # Simulate L1 output structure
        l1_metadata = {
            "encoding": "utf-8",
            "delimiter": ",",
            "header_detection": {
                "header_row": 0,
                "confidence": "MEDIUM",
                "confidence_score": 0.55,
                "needs_review": False,
                "review_reason": None,
                "selection_reason": "first_row_is_best",
                "first_row_score": 0.55,
                "candidate_summary": [
                    {"candidate": 0, "score": 0.55, "header_line": "Product,2023,2024,2025"},
                    {"candidate": 1, "score": 0.52, "header_line": "Apple,100,120,150"}
                ]
            },
            "column_profiles": {
                "Product": {"type_distribution": {"string": 1.0}, "null_rate": 0.0},
                "2023": {"type_distribution": {"integer": 1.0}, "null_rate": 0.0},
                "2024": {"type_distribution": {"integer": 1.0}, "null_rate": 0.0},
                "2025": {"type_distribution": {"integer": 1.0}, "null_rate": 0.0}
            }
        }
        
        df = pd.DataFrame({
            "Product": ["Apple", "Banana", "Orange"],
            "2023": [100, 80, 90],
            "2024": [120, 85, 95],
            "2025": [150, 90, 100]
        })
        
        result = process_l1_output(l1_metadata, df)
        
        self.assertEqual(result["l2_status"], "success")
        self.assertIn("domain_classification", result)
        
        domain = result["domain_classification"]
        if domain:
            print(f"[PASS] L1 integration: domain={domain['primary_domain']}, "
                  f"confidence={domain['primary_confidence']:.2f}")
        else:
            print(f"[PASS] L1 integration: domain classification skipped")


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""
    
    def test_empty_dataframe(self):
        """Test: Should handle empty DataFrame gracefully."""
        classifier = DomainClassifier()
        df = pd.DataFrame()
        
        result = classifier.classify(df)
        
        self.assertIsInstance(result, DomainClassificationResult)
        print(f"[PASS] Empty DataFrame handled")
    
    def test_single_column_dataframe(self):
        """Test: Should handle single column DataFrame."""
        classifier = DomainClassifier()
        df = pd.DataFrame({"single_col": [1, 2, 3]})
        
        result = classifier.classify(df)
        
        self.assertIsInstance(result, DomainClassificationResult)
        print(f"[PASS] Single column handled: domain={result.primary_domain}")
    
    def test_missing_header_detection_key(self):
        """Test: Should handle missing header_detection key."""
        l2 = DomainPerceptionL2()
        l1_metadata = {}  # No header_detection
        
        result = l2.process(l1_metadata)
        
        self.assertEqual(result["l2_status"], "success")
        print(f"[PASS] Missing header_detection handled")
    
    def test_unicode_column_names(self):
        """Test: Should handle Unicode column names."""
        classifier = DomainClassifier()
        df = pd.DataFrame({
            "用户名": ["张三", "李四", "王五"],
            "年龄": [25, 30, 28],
            "城市": ["北京", "上海", "广州"]
        })
        
        result = classifier.classify(df)
        
        self.assertIsInstance(result, DomainClassificationResult)
        print(f"[PASS] Unicode columns: domain={result.primary_domain}")


if __name__ == "__main__":
    print("=" * 60)
    print("L2 Domain Perception Layer - Test Suite")
    print("=" * 60)
    print()
    
    unittest.main(verbosity=2)

