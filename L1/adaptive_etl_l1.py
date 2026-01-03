
"""
Adaptive ETL L1 with integrated L1.5 features (header candidates + column semantic profiles)

- No demo data or hard-coded business rules.
- Outputs per-chunk IR: CSV + JSON metadata.
- Metadata now includes:
    - header_candidates: list of candidate header rows with explainable scores/reasons
    - column_profiles: per-column semantic profiles (type distribution, null_rate, unique_ratio, entropy, samples, is_id_like)
"""

from typing import List, Dict, Any, Tuple, Iterator
import os
import io
import sys
import csv
import json
import math
import logging
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

# Optional libs
try:
    import chardet
except Exception:
    chardet = None

try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None

# Configuration loader
try:
    from config_loader import get_config
    _config = get_config()
except ImportError:
    _config = None

def _get_cfg(key: str, default: Any = None) -> Any:
    """Get configuration value with fallback to default."""
    if _config is not None:
        return _config.get(key, default)
    return default

# Setup logging from config
_log_level = _get_cfg("logging.level", "INFO")
_log_format = _get_cfg("logging.format", "%(levelname)s:%(name)s:%(message)s")
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO), format=_log_format)
logger = logging.getLogger("AdaptiveETL_L1Fused")


# ---------------- utilities ----------------

def detect_encoding_from_bytes(sample_bytes: bytes) -> str:
    if chardet is not None:
        res = chardet.detect(sample_bytes)
        enc = res.get("encoding") or "utf-8"
        logger.debug("chardet.detect -> %s", res)
        return enc
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin1", "cp1252"):
        try:
            sample_bytes.decode(enc)
            return enc
        except Exception:
            continue
    return "utf-8"


def sniff_delimiter_from_text(sample_text: str, candidates: List[str] = None) -> str:
    if candidates is None:
        candidates = [",", "\t", "|", ";"]
    lines = [ln for ln in sample_text.splitlines() if ln.strip()]
    if not lines:
        return ","
    best = candidates[0]
    best_score = -1.0
    for d in candidates:
        try:
            counts = [len(list(csv.reader([line], delimiter=d))[0]) for line in lines[:200]]
        except Exception:
            continue
        if not counts:
            continue
        avg = float(np.mean(counts))
        std = float(np.std(counts))
        score = (avg + 1e-9) / (std + 1e-9)
        if score > best_score:
            best_score = score
            best = d
    logger.debug("sniff_delimiter -> chosen '%s' score=%.3f", best, best_score)
    return best


def make_unique_columns(columns: List[str]) -> List[str]:
    seen = {}
    out = []
    for c in columns:
        name = str(c)
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}__dup{seen[name]}")
    return out


def entropy_of_series(s: pd.Series) -> float:
    vals = s.dropna().astype(str)
    if vals.empty:
        return 0.0
    counts = Counter(vals)
    probs = np.array([v / len(vals) for v in counts.values()])
    ent = -np.sum(probs * np.log(probs + 1e-12))
    return float(ent)


def is_id_like(series: pd.Series) -> bool:
    """
    Heuristic: ID-like if high uniqueness and mostly numeric or alphanumeric short strings.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return False
    unique_ratio = s.nunique() / len(s)
    # short average length and uniqueness high
    avg_len = s.map(len).mean()
    numeric_ratio = s.apply(lambda x: x.isdigit()).mean()
    alnum_ratio = s.apply(lambda x: x.isalnum()).mean()
    return (unique_ratio > 0.9 and avg_len <= 32 and (numeric_ratio > 0.5 or alnum_ratio > 0.7))


# ---------------- type heuristics ----------------

def _is_numeric_str(x: str) -> bool:
    try:
        float(str(x).replace(",", ""))
        return True
    except Exception:
        return False


def _is_datetime_str(x: str) -> bool:
    try:
        pd.to_datetime(x, errors="raise")
        return True
    except Exception:
        return False


def column_type_scores(values: List[Any]) -> Dict[str, float]:
    vals = [v for v in values if v is not None and str(v).strip() != ""]
    n = len(vals)
    if n == 0:
        return {"numeric": 0.0, "datetime": 0.0, "categorical": 0.0, "text": 0.0}
    numeric_ok = 0
    dt_ok = 0
    uniq = set()
    lengths = []
    for v in vals:
        s = str(v).strip()
        uniq.add(s)
        lengths.append(len(s))
        if _is_numeric_str(s):
            numeric_ok += 1
        elif _is_datetime_str(s):
            dt_ok += 1
    numeric_score = numeric_ok / n
    dt_score = dt_ok / n
    uniq_ratio = len(uniq) / n
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    categorical_score = float(uniq_ratio < 0.2 and avg_len < 30)
    text_score = 1.0 - max(numeric_score, dt_score, categorical_score)
    raw = np.array([numeric_score, dt_score, categorical_score, text_score], dtype=float)
    if raw.sum() == 0:
        return {"numeric": 0.0, "datetime": 0.0, "categorical": 0.0, "text": 1.0}
    raw = raw / raw.sum()
    return {"numeric": float(raw[0]), "datetime": float(raw[1]), "categorical": float(raw[2]), "text": float(raw[3])}

def infer_header_row_by_consistency(
    text: str,
    delimiter: str,
    max_header_scan: int = 8,
    look_ahead: int = 20,
) -> int:
    """
    Infer header row index by minimizing column type entropy
    and maximizing type consistency in subsequent rows.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0

    best_idx = 0
    best_score = -1.0

    max_scan = min(max_header_scan, max(0, len(lines) - 1))

    for i in range(0, max_scan + 1):
        block = lines[i : i + 1 + look_ahead]
        block_text = "\n".join(block)
        try:
            df = pd.read_csv(
                io.StringIO(block_text),
                delimiter=delimiter,
                nrows=look_ahead + 1,
                dtype=str,
            )
        except Exception:
            continue

        # skip degenerate cases
        if df.shape[1] <= 1:
            continue

        # compute column-wise dominant type ratio
        col_scores = []
        for col in df.columns:
            s = df[col].dropna().astype(str)
            if s.empty:
                continue
            scores = column_type_scores(s.tolist())
            col_scores.append(max(scores.values()))

        if not col_scores:
            continue

        consistency = float(np.mean(col_scores))

        # slight penalty for numeric-looking header row
        header_vals = df.iloc[0].astype(str)
        numeric_header_penalty = sum(_is_numeric_str(v) for v in header_vals) / len(header_vals)
        final_score = consistency - 0.3 * numeric_header_penalty

        if final_score > best_score:
            best_score = final_score
            best_idx = i

    logger.info("inferred header row index: %d (score %.3f)", best_idx, best_score)
    return best_idx


# ---------------- header candidate scoring (L1.5 merged) ----------------

# ---------------- header heuristic helpers ----------------

def _is_header_like_token(token: str) -> float:
    """
    Compute a score [0, 1] indicating how 'header-like' a token is.
    Uses purely statistical/heuristic features, no vocabulary matching.
    
    Header-like characteristics:
    - Pure alphabetic or alphanumeric with underscores (identifiers)
    - Not parseable as number or datetime
    - Reasonable length (not too short, not too long)
    - Contains letters (not just punctuation/symbols)
    """
    s = token.strip()
    if not s:
        return 0.0
    
    score = 0.0
    
    # 1. Non-numeric: headers are usually not numbers
    if not _is_numeric_str(s):
        score += 0.3
    
    # 2. Non-datetime: headers are usually not datetime values
    if not _is_datetime_str(s):
        score += 0.2
    
    # 3. Contains letters: headers should have alphabetic content
    alpha_ratio = sum(1 for c in s if c.isalpha()) / len(s) if s else 0
    score += 0.25 * alpha_ratio
    
    # 4. Identifier-like pattern: letters, digits, underscores, no spaces
    is_identifier = s.replace("_", "").replace("-", "").isalnum() and not s[0].isdigit()
    if is_identifier:
        score += 0.15
    
    # 5. Reasonable length: typical column names are 2-30 chars
    if 2 <= len(s) <= 30:
        score += 0.1
    elif len(s) > 50:  # Very long strings are likely data, not headers
        score -= 0.2
    
    return max(0.0, min(1.0, score))


def score_header_candidate(lines: List[str], candidate_index: int, delimiter: str, look_ahead: int = 20) -> Dict[str, Any]:
    """
    Given raw text lines and a candidate row index, compute explainable scores:
      - stability_score: how consistent the following rows appear as data
      - token_score: whether tokens look like header names (not numeric, contain letters)
      - uniqueness_score: do candidate tokens look unique (not repeated)
      - consistency_score: averaged dominant-type consistency across columns
    Returns a dict with scores and short reasons.
    """
    n_lines = len(lines)
    # build a small test block: header candidate line + next look_ahead rows
    header_line = lines[candidate_index] if candidate_index < n_lines else ""
    content_lines = lines[candidate_index + 1 : candidate_index + 1 + look_ahead]
    test_text = "\n".join([header_line] + content_lines)
    # parse to DataFrame
    try:
        df = pd.read_csv(io.StringIO(test_text), delimiter=delimiter, nrows=look_ahead + 1, dtype=str)
    except Exception:
        # fallback: try python engine
        try:
            df = pd.read_csv(io.StringIO(test_text), delimiter=delimiter, nrows=look_ahead + 1, dtype=str, engine="python")
        except Exception:
            return {"candidate": candidate_index, "score": 0.0, "reason": "parse_failed"}
    # token heuristics - using statistical features, no vocabulary matching
    tokens = [t.strip() for t in header_line.split(delimiter)] if header_line else []
    token_count = len(tokens)
    if token_count == 0:
        token_score = 0.0
        data_row_penalty = 0.0
    else:
        # Compute header-like score for each token using heuristic features
        header_like_scores = [_is_header_like_token(t) for t in tokens]
        token_score = sum(header_like_scores) / token_count
        
        # Data row penalty: penalize rows that look like data (contain many numbers/dates)
        numeric_ratio = sum(1 for t in tokens if _is_numeric_str(t)) / token_count
        datetime_ratio = sum(1 for t in tokens if _is_datetime_str(t)) / token_count
        # Combined penalty: higher when row contains more numeric/datetime values
        data_row_penalty = 0.4 * (numeric_ratio + 0.8 * datetime_ratio)
    # uniqueness
    uniq_tokens = len(set(tokens)) / (token_count + 1e-9) if token_count else 0.0
    uniqueness_score = float(uniq_tokens)
    # column consistency: for each inferred column, compute dominant-type proportion across rows
    col_consistency_scores = []
    for col in df.columns:
        s = df[col].dropna().astype(str).tolist()
        if len(s) == 0:
            col_consistency_scores.append(0.0)
            continue
        t_scores = column_type_scores(s)
        dom = max(t_scores.values())
        col_consistency_scores.append(dom)
    consistency_score = float(np.mean(col_consistency_scores)) if col_consistency_scores else 0.0
    # stability: how much columns change across consecutive rows (lower variance better)
    stability = 0.0
    try:
        # use simple per-column zscore variance as proxy
        numeric_mask = []
        for col in df.columns:
            colvals = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
            numeric_mask.append(colvals.notna().sum() > 0)
        stability = 1.0 - float(np.nanmean([np.nanstd(pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce").dropna()) if numeric_mask[i] else 0.0 for i, c in enumerate(df.columns)]))
        stability = max(0.0, min(1.0, stability))
    except Exception:
        stability = 0.0
    # combined score (weights favor header-like tokens over pure consistency)
    # Apply data row penalty to discourage selecting data rows as headers
    combined_score = 0.30 * consistency_score + 0.40 * token_score + 0.15 * uniqueness_score + 0.15 * stability - data_row_penalty
    combined_score = max(0.0, combined_score)  # Ensure non-negative
    reason = {
        "consistency_score": round(consistency_score, 4),
        "token_score": round(token_score, 4),
        "uniqueness_score": round(uniqueness_score, 4),
        "stability": round(stability, 4),
        "data_row_penalty": round(data_row_penalty, 4)
    }
    return {
        "candidate": int(candidate_index),
        "score": float(combined_score),
        "reason": reason,
        "header_line": header_line,
        "token_count": int(token_count)
    }


# ---------------- anomaly ensemble (with missing value detection) ----------------

def ensemble_anomaly_scores(df: pd.DataFrame) -> Tuple[List[float], Dict[str, List[float]]]:
    n_cols = len(df.columns)
    
    # Compute per-row missing ratio as an anomaly signal
    # High missing ratio = more anomalous
    missing_counts = df.isna().sum(axis=1) + df.astype(str).apply(lambda row: (row == '').sum(), axis=1)
    missing_ratio = (missing_counts / n_cols).values.astype(float)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        lens = df.astype(str).applymap(lambda v: len(str(v))).sum(axis=1).values.astype(float)
        z = np.abs(stats.zscore(lens, nan_policy="omit"))
        z = np.nan_to_num(z)
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
        # Combine with missing ratio (weight: 0.6 length-based, 0.4 missing-based)
        row_scores = (0.6 * z_norm + 0.4 * missing_ratio).tolist()
        return row_scores, {"row_length_z": z.tolist(), "missing_ratio": missing_ratio.tolist()}
    
    mat = df[numeric_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    per_col_z = {}
    for c in numeric_cols:
        col_vals = mat[c].values.astype(float)
        zc = np.abs(stats.zscore(col_vals, nan_policy="omit"))
        zc = np.nan_to_num(zc)
        per_col_z[c] = zc.tolist()
    zs = np.vstack([np.array(per_col_z[c]) for c in numeric_cols])
    agg_z = np.nanmean(zs, axis=0)
    agg_z = np.nan_to_num(agg_z)
    agg_norm = (agg_z - agg_z.min()) / (agg_z.max() - agg_z.min() + 1e-9)
    
    # Store missing ratio for downstream use
    per_col_z["missing_ratio"] = missing_ratio.tolist()
    
    # Get config values
    missing_weight = _get_cfg("l1_etl.anomaly_detection.missing_ratio_weight", 0.30)
    enable_iso = _get_cfg("l1_etl.anomaly_detection.enable_isolation_forest", True)
    iso_estimators = _get_cfg("l1_etl.anomaly_detection.isolation_forest_estimators", 100)
    
    if enable_iso and IsolationForest is not None and len(df) >= 5:
        try:
            iso = IsolationForest(n_estimators=iso_estimators, contamination="auto", random_state=42)
            X = mat.fillna(mat.median()).values
            iso.fit(X)
            iso_scores = -iso.decision_function(X)
            iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
            # Combine: z-score + isolation forest + missing ratio (weights from config)
            z_iso_weight = (1.0 - missing_weight) / 2
            combined = z_iso_weight * agg_norm + z_iso_weight * iso_norm + missing_weight * missing_ratio
            row_scores = combined.tolist()
            per_col_z["isolation_forest_score"] = iso_scores.tolist()
            return row_scores, per_col_z
        except Exception as e:
            logger.warning("IsolationForest failed: %s", e)
            # Fallback: z-score + missing ratio
            z_weight = 1.0 - missing_weight
            combined = z_weight * agg_norm + missing_weight * missing_ratio
            return combined.tolist(), per_col_z
    else:
        # No isolation forest: z-score + missing ratio
        z_weight = 1.0 - missing_weight
        combined = z_weight * agg_norm + missing_weight * missing_ratio
        return combined.tolist(), per_col_z


# ---------------- column profile builder (L1.5) ----------------

def build_column_profile(series: pd.Series, sample_limit: int = 10) -> Dict[str, Any]:
    s = series.dropna().astype(str)
    total = len(series)
    non_null = len(s)
    null_rate = float(1.0 - non_null / total) if total > 0 else 0.0
    uniq = s.nunique() if non_null > 0 else 0
    unique_ratio = float(uniq / non_null) if non_null > 0 else 0.0
    type_scores = column_type_scores(s.tolist() if non_null > 0 else [])
    ent = entropy_of_series(series)
    top_samples = []
    if non_null > 0:
        c = Counter(s)
        top_samples = [v for v, _ in c.most_common(sample_limit)]
    id_like = is_id_like(series)
    avg_len = float(s.map(len).mean()) if non_null > 0 else 0.0
    return {
        "null_rate": round(null_rate, 6),
        "unique_count": int(uniq),
        "unique_ratio": round(unique_ratio, 6),
        "type_scores": type_scores,
        "entropy": round(ent, 6),
        "top_samples": top_samples,
        "is_id_like": bool(id_like),
        "avg_length": round(avg_len, 3)
    }


# ---------------- confidence level and review system ----------------

class HeaderConfidence:
    """Constants for header detection confidence levels."""
    HIGH = "HIGH"           # Score >= high threshold (default 0.75), highly reliable
    MEDIUM = "MEDIUM"       # Score >= medium threshold (default 0.50), reasonably reliable  
    LOW = "LOW"             # Score >= low threshold (default 0.30), needs attention
    UNCERTAIN = "UNCERTAIN" # Score < 0.30, likely needs manual review or LLM


def _compute_confidence_level(score: float) -> str:
    """
    Compute confidence level based on header detection score.
    Uses thresholds from config file.
    
    Returns:
        Confidence level string (HIGH/MEDIUM/LOW/UNCERTAIN)
    """
    # Get thresholds from config
    high_threshold = _get_cfg("l1_etl.confidence_thresholds.high", 0.75)
    medium_threshold = _get_cfg("l1_etl.confidence_thresholds.medium", 0.50)
    low_threshold = _get_cfg("l1_etl.confidence_thresholds.low", 0.30)
    
    if score >= high_threshold:
        return HeaderConfidence.HIGH
    elif score >= medium_threshold:
        return HeaderConfidence.MEDIUM
    elif score >= low_threshold:
        return HeaderConfidence.LOW
    else:
        return HeaderConfidence.UNCERTAIN


def _should_request_review(confidence: str, prefer_reason: str) -> Tuple[bool, str]:
    """
    Determine if human review or L2 LLM assistance is recommended.
    
    Returns:
        (needs_review, review_reason): Tuple of flag and explanation
    """
    if confidence == HeaderConfidence.UNCERTAIN:
        return True, "confidence_too_low"
    
    if confidence == HeaderConfidence.LOW:
        # Low confidence but not critical - optional review
        return True, "low_confidence_suggest_review"
    
    if "rejected" in prefer_reason:
        # First row was rejected, might want to verify
        return False, "first_row_rejected_but_confident"
    
    return False, "confident"


def create_header_detection_result(
    header_row: int,
    selected_score: float,
    header_candidates: List[Dict],
    prefer_reason: str
) -> Dict[str, Any]:
    """
    Create a structured header detection result with confidence info.
    
    This result can be passed to L2 layer for optional LLM verification.
    """
    confidence = _compute_confidence_level(selected_score)
    needs_review, review_reason = _should_request_review(confidence, prefer_reason)
    
    # Find first row candidate for comparison
    first_row_cand = next((c for c in header_candidates if c["candidate"] == 0), None)
    first_row_score = first_row_cand["score"] if first_row_cand else 0.0
    
    return {
        "header_row": header_row,
        "confidence": confidence,
        "confidence_score": round(selected_score, 4),
        "needs_review": needs_review,
        "review_reason": review_reason,
        "selection_reason": prefer_reason,
        "first_row_score": round(first_row_score, 4),
        # Preview data for L2 LLM analysis
        "candidate_summary": [
            {
                "row": c["candidate"],
                "score": round(c["score"], 4),
                "preview": c.get("header_line", "")[:80]
            }
            for c in header_candidates[:3]  # Top 3 candidates
        ]
    }


# ---------------- first-row preference logic ----------------

def _is_comment_line(line: str) -> bool:
    """Check if a line looks like a comment."""
    stripped = line.strip()
    if not stripped:
        return False
    # Common comment prefixes
    comment_prefixes = ('#', '//', '--', '/*', '<!--', ';', '%')
    return stripped.startswith(comment_prefixes)


def _should_prefer_first_row(header_candidates: List[Dict], score_gap_threshold: float = None) -> Tuple[bool, str]:
    """
    Intelligent decision: should we prefer the first row as header?
    
    Core logic: Most CSV files have headers in the first row.
    Only reject the first row when it's CLEARLY a data row or a comment.
    
    Args:
        header_candidates: Sorted list of candidates (best first)
        score_gap_threshold: Max gap to still prefer first row (from config, default 0.40)
    
    Returns:
        (should_prefer, reason): Tuple of decision and explanation
    """
    # Get threshold from config if not provided
    if score_gap_threshold is None:
        score_gap_threshold = _get_cfg("l1_etl.score_gap_threshold", 0.40)
    
    if not header_candidates:
        return True, "no_candidates"
    
    best = header_candidates[0]
    
    # Find first row candidate
    first_row_candidate = None
    for c in header_candidates:
        if c["candidate"] == 0:
            first_row_candidate = c
            break
    
    if first_row_candidate is None:
        return False, "first_row_not_in_candidates"
    
    # Check if first row is a comment line
    first_row_line = first_row_candidate.get("header_line", "")
    if _is_comment_line(first_row_line):
        return False, "first_row_is_comment"
    
    # If first row is already the best, no change needed
    if best["candidate"] == 0:
        return True, "first_row_is_best"
    
    first_row_score = first_row_candidate["score"]
    best_score = best["score"]
    score_gap = best_score - first_row_score
    
    # Key insight: if gap is within threshold, trust the first row (prior knowledge)
    # Increased threshold to 0.40 to handle numeric-looking headers like years
    if score_gap < score_gap_threshold:
        return True, f"small_gap_{score_gap:.3f}"
    
    # Only reject first row when gap is too large
    return False, f"first_row_rejected_gap_{score_gap:.3f}"


# ---------------- AdaptiveETL class (with L1.5 merged) ----------------

class AdaptiveETL:
    def __init__(self, 
                 chunk_rows: int = None, 
                 sniff_bytes: int = None, 
                 max_header_scan: int = None, 
                 header_lookahead: int = None):
        # Use config values with fallback to defaults
        self.chunk_rows = int(chunk_rows or _get_cfg("l1_etl.chunk_rows", 10000))
        self.sniff_bytes = int(sniff_bytes or _get_cfg("l1_etl.sniff_bytes", 65536))
        self.max_header_scan = int(max_header_scan or _get_cfg("l1_etl.max_header_scan", 8))
        self.header_lookahead = int(header_lookahead or _get_cfg("l1_etl.header_lookahead", 20))
        
        # Get score gap threshold from config
        self.score_gap_threshold = _get_cfg("l1_etl.score_gap_threshold", 0.40)
        
        # Get confidence thresholds from config
        self.confidence_high = _get_cfg("l1_etl.confidence_thresholds.high", 0.75)
        self.confidence_medium = _get_cfg("l1_etl.confidence_thresholds.medium", 0.50)
        self.confidence_low = _get_cfg("l1_etl.confidence_thresholds.low", 0.30)

    def _sample_bytes(self, file_path: str) -> bytes:
        with open(file_path, "rb") as f:
            return f.read(self.sniff_bytes)

    def detect_file_format(self, file_path: str) -> Dict[str, Any]:
        sample = self._sample_bytes(file_path)
        encoding = detect_encoding_from_bytes(sample)
        try:
            text = sample.decode(encoding, errors="replace")
        except Exception:
            text = sample.decode("utf-8", errors="replace")
            encoding = "utf-8"
        delimiter = sniff_delimiter_from_text(text)
        
        # Create header candidates list with scoring
        lines = [ln for ln in text.splitlines() if ln.strip()]
        header_candidates = []
        max_scan = min(self.max_header_scan, max(0, len(lines) - 1))
        for i in range(0, max_scan + 1):
            cand = score_header_candidate(lines, i, delimiter, look_ahead=self.header_lookahead)
            header_candidates.append(cand)
        
        # Sort by score descending - best candidate first
        header_candidates = sorted(header_candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Apply first-row preference logic
        # Most CSVs have headers in first row; only reject when clearly a data row
        prefer_first, prefer_reason = _should_prefer_first_row(header_candidates)
        
        if prefer_first and header_candidates:
            # Find and use first row
            first_row_candidate = next((c for c in header_candidates if c["candidate"] == 0), None)
            if first_row_candidate:
                header_row = 0
                selected_score = first_row_candidate["score"]
            else:
                header_row = header_candidates[0]["candidate"]
                selected_score = header_candidates[0]["score"]
        else:
            # Use best scoring candidate
            header_row = header_candidates[0]["candidate"] if header_candidates else 0
            selected_score = header_candidates[0].get("score", 0.0) if header_candidates else 0.0
        
        # Create structured detection result with confidence info
        detection_result = create_header_detection_result(
            header_row=header_row,
            selected_score=selected_score,
            header_candidates=header_candidates,
            prefer_reason=prefer_reason
        )
        
        # Log with confidence level
        logger.info("Selected header_row=%d (confidence=%s, score=%.4f, needs_review=%s)", 
                    header_row, detection_result["confidence"], selected_score, 
                    detection_result["needs_review"])
        
        return {
            "encoding": encoding, 
            "delimiter": delimiter, 
            "header_row": header_row, 
            "header_candidates": header_candidates,
            "header_detection": detection_result  # New: structured detection info for L2
        }

    def infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        types = {}
        for c in df.columns:
            s = df[c].dropna().astype(str)
            if s.empty:
                types[c] = "empty"
            elif s.apply(_is_numeric_str).mean() > 0.9:
                types[c] = "numeric"
            elif s.apply(_is_datetime_str).mean() > 0.9:
                types[c] = "datetime"
            elif s.nunique() / max(1, len(s)) < 0.2:
                types[c] = "categorical"
            else:
                types[c] = "text"
        return types

    def stream_parse(self, file_path: str) -> Iterator[Dict[str, Any]]:
        fmt = self.detect_file_format(file_path)
        encoding = fmt["encoding"]
        delimiter = fmt["delimiter"]
        header_row = fmt["header_row"]
        header_candidates = fmt.get("header_candidates", [])
        header_detection = fmt.get("header_detection", {})  # New: confidence info
        # try header_row as header index
        reader = pd.read_csv(
            file_path,
            delimiter=delimiter,
            encoding=encoding,
            header=header_row,
            chunksize=self.chunk_rows,
            dtype=str,
            low_memory=False,
        )
        row_offset = 0
        for chunk in reader:
            df = chunk.copy()
            # dedupe columns
            df.columns = make_unique_columns(list(df.columns))
            # build column profiles BEFORE coercion to preserve textual signals
            column_profiles = {c: build_column_profile(df[c]) for c in df.columns}
            # coerce numeric-like columns where possible (for anomaly math)
            for c in df.columns:
                coerced = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
                if coerced.notna().sum() > 0:
                    df[c] = coerced
            # infer types (semantic-agnostic)
            column_types = self.infer_column_types(df)
            # anomaly detection
            row_scores, per_column = ensemble_anomaly_scores(df)
            df["_anomaly_score"] = row_scores
            metadata = {
                "source_file": os.path.basename(file_path),
                "encoding": encoding,
                "delimiter": delimiter,
                "header_row": int(header_row),
                "header_detection": header_detection,  # New: confidence & review info for L2
                "header_candidates": header_candidates,
                "row_offset": int(row_offset),
                "chunk_rows": int(len(df)),
                "column_profiles": column_profiles,
                "column_types": column_types,
                "anomaly": {
                    "methods": ["z_score"] + (["isolation_forest"] if IsolationForest is not None else []),
                    "row_score": row_scores,
                    "per_column": per_column,
                },
            }
            yield {"data": df, "metadata": metadata}
            row_offset += len(df)


# ---------------- serialization helpers ----------------

def _to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.ndarray,)):
        return _to_serializable(obj.tolist())
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------- minimal CLI runner ----------------

if __name__ == "__main__":
    """
    Usage:
        python adaptive_etl_l1.py path/to/input.csv

    Produces:
        ir_output/chunk_0000.csv
        ir_output/chunk_0000_meta.json
    """
    if len(sys.argv) < 2:
        print("Usage: python adaptive_etl_l1.py <input_csv_path>", file=sys.stderr)
        sys.exit(2)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(3)

    out_dir = "ir_output"
    os.makedirs(out_dir, exist_ok=True)

    etl = AdaptiveETL()

    saved_chunks = []
    try:
        for idx, ir in enumerate(etl.stream_parse(input_path)):
            df = ir["data"]
            meta = ir["metadata"]

            data_path = os.path.join(out_dir, f"chunk_{idx:04d}.csv")
            df.to_csv(data_path, index=False, encoding="utf-8")

            meta_serial = _to_serializable(meta)
            meta_serial["shape"] = (int(df.shape[0]), int(df.shape[1]))
            meta_path = os.path.join(out_dir, f"chunk_{idx:04d}_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_serial, f, ensure_ascii=False, indent=2)

            saved_chunks.append({"idx": idx, "data": data_path, "meta": meta_path, "rows": int(df.shape[0])})
            print(f"[L1] chunk={idx} rows={int(df.shape[0])} header_row={meta.get('header_row')} cols={int(df.shape[1])}")

    except Exception as e:
        logger.exception("Exception during L1 parsing: %s", e)
        sys.exit(4)

    print(f"[L1] finished. chunks_saved={len(saved_chunks)} output_dir={out_dir}")
