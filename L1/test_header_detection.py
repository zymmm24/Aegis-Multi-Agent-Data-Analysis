"""
Unit tests for header detection in AdaptiveETL L1 layer.
Tests various scenarios: standard headers, headers with comment rows, no-header files.
Uses purely statistical/heuristic features, no vocabulary matching.
"""

import os
import tempfile
import unittest
from adaptive_etl_l1 import AdaptiveETL, score_header_candidate, _is_header_like_token


class TestHeaderDetection(unittest.TestCase):
    """Test cases for header row detection."""

    def setUp(self):
        """Create temporary test files."""
        self.etl = AdaptiveETL()
        self.temp_files = []

    def tearDown(self):
        """Clean up temporary files."""
        for f in self.temp_files:
            if os.path.exists(f):
                os.remove(f)

    def _create_temp_csv(self, content: str) -> str:
        """Helper to create a temporary CSV file."""
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        self.temp_files.append(path)
        return path

    def test_standard_header_first_row(self):
        """Test: Standard CSV with header in first row."""
        content = """id,timestamp,amount,category,description
1,2025-01-01 10:00:00,123.45,A,normal entry
2,2025-01-01 10:05:00,130.00,A,normal entry
3,2025-01-01 10:10:00,99999999,A,outlier amount
4,2025-01-02 10:00:00,200.00,B,another entry
5,2025-01-02 10:20:00,150.00,A,normal entry
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        self.assertEqual(result["header_row"], 0, 
                         f"Expected header_row=0, got {result['header_row']}")
        self.assertEqual(result["delimiter"], ",")
        
        # Verify the best candidate is row 0
        best_candidate = result["header_candidates"][0]
        self.assertEqual(best_candidate["candidate"], 0)
        print(f"[PASS] Standard header: row={result['header_row']}, "
              f"score={best_candidate['score']:.4f}")

    def test_header_with_comment_rows(self):
        """Test: CSV with comment/metadata rows before actual header."""
        content = """# This is a comment line
# Generated on 2025-01-01
id,name,value,status
1,Alpha,100,active
2,Beta,200,inactive
3,Gamma,300,active
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        # Header should be row 2 (0-indexed: comment, comment, header)
        self.assertEqual(result["header_row"], 2,
                         f"Expected header_row=2 (after comments), got {result['header_row']}")
        print(f"[PASS] Header with comments: row={result['header_row']}")

    def test_no_header_numeric_data(self):
        """Test: CSV with no header, all numeric data."""
        content = """1,100,50.5,0.1
2,200,60.3,0.2
3,300,70.1,0.3
4,400,80.9,0.4
5,500,90.7,0.5
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        # All rows are data rows - first row should get penalized
        # The algorithm should still pick row 0 but with lower confidence
        best_candidate = result["header_candidates"][0]
        # Check that data_row_penalty is applied
        self.assertIn("data_row_penalty", best_candidate["reason"])
        self.assertGreater(best_candidate["reason"]["data_row_penalty"], 0,
                           "Expected data_row_penalty > 0 for numeric data")
        print(f"[PASS] No header (numeric): row={result['header_row']}, "
              f"penalty={best_candidate['reason']['data_row_penalty']:.4f}")

    def test_identifier_style_header(self):
        """Test: Header with identifier-style column names (underscores, camelCase)."""
        content = """user_id,createdAt,total_amount,customFieldXyz
u001,2025-01-01,500.00,some_value
u002,2025-01-02,600.00,another_value
u003,2025-01-03,700.00,third_value
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        self.assertEqual(result["header_row"], 0)
        best_candidate = result["header_candidates"][0]
        # Token score should be high for identifier-style headers
        self.assertGreater(best_candidate["reason"]["token_score"], 0.5,
                           "Expected token_score > 0.5 for identifier-style header")
        print(f"[PASS] Identifier header: row={result['header_row']}, "
              f"token_score={best_candidate['reason']['token_score']:.4f}")

    def test_sample_mixed_csv(self):
        """Test: The actual sample_mixed.csv file from the project."""
        sample_path = os.path.join(os.path.dirname(__file__), "sample_mixed.csv")
        if not os.path.exists(sample_path):
            self.skipTest("sample_mixed.csv not found")
        
        result = self.etl.detect_file_format(sample_path)
        
        # The correct header should be row 0: id,timestamp,amount,category,description
        self.assertEqual(result["header_row"], 0,
                         f"sample_mixed.csv: Expected header_row=0, got {result['header_row']}")
        
        best_candidate = result["header_candidates"][0]
        print(f"[PASS] sample_mixed.csv: row={result['header_row']}, "
              f"score={best_candidate['score']:.4f}, "
              f"reason={best_candidate['reason']}")


class TestScoreHeaderCandidate(unittest.TestCase):
    """Test the score_header_candidate function directly."""

    def test_header_vs_data_row(self):
        """Test that header row scores higher than data row."""
        lines = [
            "id,name,price,quantity",  # Row 0: header
            "1,Apple,1.50,10",         # Row 1: data
            "2,Banana,0.75,20",        # Row 2: data
            "3,Orange,2.00,15",        # Row 3: data
        ]
        delimiter = ","
        
        header_score = score_header_candidate(lines, 0, delimiter)
        data_score = score_header_candidate(lines, 1, delimiter)
        
        self.assertGreater(header_score["score"], data_score["score"],
                           f"Header row score ({header_score['score']:.4f}) should be > "
                           f"data row score ({data_score['score']:.4f})")
        print(f"[PASS] Header score={header_score['score']:.4f} > "
              f"Data score={data_score['score']:.4f}")

    def test_header_like_token_heuristic(self):
        """Test the _is_header_like_token heuristic function."""
        # Header-like tokens (alphabetic, identifier-style)
        self.assertGreater(_is_header_like_token("user_id"), 0.7)
        self.assertGreater(_is_header_like_token("timestamp"), 0.7)
        self.assertGreater(_is_header_like_token("ProductName"), 0.7)
        
        # Data-like tokens (numeric, datetime)
        self.assertLess(_is_header_like_token("12345"), 0.5)
        self.assertLess(_is_header_like_token("2025-01-01"), 0.5)
        self.assertLess(_is_header_like_token("99.99"), 0.5)
        
        # Very long strings are penalized (likely data)
        long_text = "This is a very long description that is unlikely to be a header name"
        self.assertLess(_is_header_like_token(long_text), 0.6)
        
        print("[PASS] Header-like token heuristic works correctly")
    
    def test_alphabetic_vs_numeric_headers(self):
        """Test that alphabetic headers score higher than numeric-looking ones."""
        lines_alphabetic = [
            "name,category,status,description",
            "Apple,Fruit,active,A red fruit",
        ]
        lines_numeric_header = [
            "1,2,3,4",  # Numeric "header"
            "Apple,Fruit,active,A red fruit",
        ]
        delimiter = ","
        
        score_alpha = score_header_candidate(lines_alphabetic, 0, delimiter)
        score_numeric = score_header_candidate(lines_numeric_header, 0, delimiter)
        
        self.assertGreater(score_alpha["score"], score_numeric["score"],
                           "Alphabetic header should score higher than numeric")
        print(f"[PASS] Alphabetic header: {score_alpha['score']:.4f} > "
              f"Numeric header: {score_numeric['score']:.4f}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and robustness of header detection."""

    def setUp(self):
        self.etl = AdaptiveETL()
        self.temp_files = []

    def tearDown(self):
        for f in self.temp_files:
            if os.path.exists(f):
                os.remove(f)

    def _create_temp_csv(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        self.temp_files.append(path)
        return path

    def test_chinese_headers(self):
        """Test: Chinese language headers."""
        content = """用户ID,姓名,年龄,城市,注册时间
1,张三,25,北京,2025-01-01
2,李四,30,上海,2025-01-02
3,王五,28,广州,2025-01-03
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        self.assertEqual(result["header_row"], 0,
                         f"Chinese headers: Expected row=0, got {result['header_row']}")
        print(f"[PASS] Chinese headers: row={result['header_row']}, "
              f"score={result['header_candidates'][0]['score']:.4f}")

    def test_mixed_language_headers(self):
        """Test: Mixed Chinese-English headers."""
        content = """ID,用户名,Email,手机号,Address
1,张三,zhang@test.com,13800138000,Beijing
2,李四,li@test.com,13900139000,Shanghai
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        self.assertEqual(result["header_row"], 0)
        print(f"[PASS] Mixed language headers: row={result['header_row']}")

    def test_short_single_char_headers(self):
        """Test: Very short single-character headers."""
        content = """A,B,C,D,E
1,2,3,4,5
6,7,8,9,10
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        # Single letters should still be recognized as headers over numbers
        self.assertEqual(result["header_row"], 0)
        print(f"[PASS] Single char headers: row={result['header_row']}")

    def test_numeric_looking_headers(self):
        """Test: Headers that look like numbers (e.g., year columns).
        
        With first-row preference logic, we should correctly select row 0
        even when it contains numeric-looking values like years.
        """
        content = """Product,2023,2024,2025,Growth
Apple,100,120,150,50%
Banana,80,90,100,25%
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        # With first-row preference, should correctly select row 0
        self.assertEqual(result["header_row"], 0,
                         f"Expected header_row=0 (first-row preference), got {result['header_row']}")
        
        # Find first row candidate score
        first_row_cand = next((c for c in result["header_candidates"] if c["candidate"] == 0), None)
        print(f"[PASS] Numeric-looking headers: row={result['header_row']}, "
              f"first_row_score={first_row_cand['score']:.4f}")

    def test_header_with_special_chars(self):
        """Test: Headers with special characters."""
        content = """user-id,created_at,total$amount,tax%,notes#1
1,2025-01-01,100.00,10,note1
2,2025-01-02,200.00,20,note2
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        self.assertEqual(result["header_row"], 0)
        print(f"[PASS] Special char headers: row={result['header_row']}")

    def test_empty_first_rows(self):
        """Test: CSV with empty rows before header."""
        content = """

id,name,value
1,Alpha,100
2,Beta,200
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        # Should skip empty rows and find the actual header
        best = result["header_candidates"][0]
        print(f"[INFO] Empty first rows: header_row={result['header_row']}, "
              f"header_line='{best.get('header_line', '')[:30]}...'")

    def test_tsv_format(self):
        """Test: Tab-separated values."""
        content = """id\tname\tvalue\tstatus
1\tAlpha\t100\tactive
2\tBeta\t200\tinactive
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        self.assertEqual(result["delimiter"], "\t", "Should detect tab delimiter")
        self.assertEqual(result["header_row"], 0)
        print(f"[PASS] TSV format: delimiter=TAB, header_row={result['header_row']}")

    def test_semicolon_delimiter(self):
        """Test: Semicolon-separated values (common in European locales)."""
        content = """id;name;price;quantity
1;Apple;1,50;10
2;Banana;0,75;20
"""
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        self.assertEqual(result["delimiter"], ";", "Should detect semicolon delimiter")
        self.assertEqual(result["header_row"], 0)
        print(f"[PASS] Semicolon CSV: delimiter=';', header_row={result['header_row']}")

    def test_quoted_headers(self):
        """Test: Headers with quotes."""
        content = '''"User ID","Full Name","Email Address","Phone Number"
1,"John Doe","john@test.com","123-456-7890"
2,"Jane Smith","jane@test.com","098-765-4321"
'''
        path = self._create_temp_csv(content)
        result = self.etl.detect_file_format(path)
        
        self.assertEqual(result["header_row"], 0)
        print(f"[PASS] Quoted headers: row={result['header_row']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

