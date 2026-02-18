"""
Data Quality Engine — Data Fabric Core Component
─────────────────────────────────────────────────
Automated quality scoring based on 5 dimensions:
  1. Completeness — % non-null values
  2. Uniqueness   — % unique values (for key columns)
  3. Validity     — % values matching expected type/constraints
  4. Consistency  — cross-column consistency checks
  5. Timeliness   — data freshness assessment

Produces a composite quality score (0–100) used by:
  - MetadataCatalog (active metadata)
  - FederatedGovernance (compliance checks)
  - Trust Judge agent (pipeline validation)
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field
from ados.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# QUALITY DIMENSION MODELS
# ═══════════════════════════════════════════════════════════════════════

class DimensionScore(BaseModel):
    """Score for a single quality dimension."""
    dimension: str
    score: float          # 0.0 to 100.0
    weight: float = 1.0   # Relative weight in composite score
    details: Dict[str, Any] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)


class ColumnQuality(BaseModel):
    """Quality assessment for a single column."""
    column_name: str
    completeness: float = 100.0    # % non-null
    uniqueness: float = 0.0        # % unique
    validity: float = 100.0        # % valid values
    issues: List[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    """Complete quality report for a data product."""
    product_name: str
    composite_score: float = 0.0
    grade: str = "F"                # A, B, C, D, F
    dimensions: List[DimensionScore] = Field(default_factory=list)
    column_scores: List[ColumnQuality] = Field(default_factory=list)
    total_issues: int = 0
    critical_issues: List[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════════════
# QUALITY ENGINE
# ═══════════════════════════════════════════════════════════════════════

class DataQualityEngine:
    """
    Automated Data Quality Engine — core Data Fabric capability.
    
    Assesses data quality across 5 dimensions and produces
    a composite score that feeds into:
    - Active metadata (quality alerts)
    - Governance compliance checks
    - Trust judge validation
    - Catalog quality badges
    """

    # Weights for each dimension in the composite score
    DIMENSION_WEIGHTS = {
        "completeness": 0.30,
        "uniqueness": 0.15,
        "validity": 0.25,
        "consistency": 0.20,
        "timeliness": 0.10,
    }

    def __init__(self):
        self._reports: Dict[str, QualityReport] = {}

    def assess(self, product_name: str, df: pd.DataFrame,
               contract=None, last_modified: Optional[datetime] = None) -> QualityReport:
        """
        Run a full quality assessment on a dataframe.
        Uses the data contract (if available) for validation rules.
        """
        logger.info(f"QualityEngine: assessing '{product_name}' ({len(df)} rows × {len(df.columns)} cols)")

        dimensions = []
        all_issues = []

        # 1. COMPLETENESS — % non-null values
        completeness = self._assess_completeness(df)
        dimensions.append(completeness)
        all_issues.extend(completeness.issues)

        # 2. UNIQUENESS — detect unexpected duplicates
        uniqueness = self._assess_uniqueness(df, contract)
        dimensions.append(uniqueness)
        all_issues.extend(uniqueness.issues)

        # 3. VALIDITY — values match expected types/constraints
        validity = self._assess_validity(df, contract)
        dimensions.append(validity)
        all_issues.extend(validity.issues)

        # 4. CONSISTENCY — cross-column logical consistency
        consistency = self._assess_consistency(df)
        dimensions.append(consistency)
        all_issues.extend(consistency.issues)

        # 5. TIMELINESS — data freshness
        timeliness = self._assess_timeliness(last_modified)
        dimensions.append(timeliness)
        all_issues.extend(timeliness.issues)

        # Column-level scores
        column_scores = self._assess_columns(df, contract)

        # Composite score (weighted average)
        composite = sum(
            d.score * self.DIMENSION_WEIGHTS.get(d.dimension, 0.2)
            for d in dimensions
        )

        # Grade
        grade = self._score_to_grade(composite)

        critical = [i for i in all_issues if "CRITICAL" in i.upper()]

        report = QualityReport(
            product_name=product_name,
            composite_score=round(composite, 1),
            grade=grade,
            dimensions=dimensions,
            column_scores=column_scores,
            total_issues=len(all_issues),
            critical_issues=critical,
        )

        self._reports[product_name] = report
        logger.info(
            f"QualityEngine: '{product_name}' → score={composite:.1f}/100, "
            f"grade={grade}, issues={len(all_issues)}"
        )
        return report

    # ── Dimension assessments ──────────────────────────────────────

    def _assess_completeness(self, df: pd.DataFrame) -> DimensionScore:
        """Dimension 1: What % of values are non-null?"""
        total_cells = df.shape[0] * df.shape[1]
        null_cells = int(df.isnull().sum().sum())
        completeness_pct = ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0

        # Per-column completeness
        col_completeness = {}
        issues = []
        for col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            col_completeness[col] = round(100 - null_pct, 2)
            if null_pct > 20:
                issues.append(f"CRITICAL: Column '{col}' has {null_pct:.1f}% nulls")
            elif null_pct > 5:
                issues.append(f"Warning: Column '{col}' has {null_pct:.1f}% nulls")

        return DimensionScore(
            dimension="completeness",
            score=round(completeness_pct, 2),
            weight=self.DIMENSION_WEIGHTS["completeness"],
            details={"per_column": col_completeness, "total_nulls": null_cells},
            issues=issues,
        )

    def _assess_uniqueness(self, df: pd.DataFrame, contract=None) -> DimensionScore:
        """Dimension 2: Are there unexpected duplicate rows?"""
        total_rows = len(df)
        dup_rows = int(df.duplicated().sum())
        dup_pct = (dup_rows / total_rows * 100) if total_rows > 0 else 0
        uniqueness_score = 100 - dup_pct

        issues = []
        if dup_pct > 5:
            issues.append(f"CRITICAL: {dup_pct:.1f}% duplicate rows ({dup_rows}/{total_rows})")
        elif dup_pct > 1:
            issues.append(f"Warning: {dup_pct:.1f}% duplicate rows")

        # Check contract-required unique columns
        details = {"duplicate_rows": dup_rows, "duplicate_pct": round(dup_pct, 2)}
        if contract and hasattr(contract, "schema_contracts"):
            for sc in contract.schema_contracts:
                if sc.unique:
                    col_dups = int(df[sc.column_name].duplicated().sum()) if sc.column_name in df.columns else 0
                    if col_dups > 0:
                        issues.append(
                            f"Column '{sc.column_name}' should be unique but has {col_dups} duplicates"
                        )
                        uniqueness_score -= 10  # Penalty

        return DimensionScore(
            dimension="uniqueness",
            score=max(0, round(uniqueness_score, 2)),
            weight=self.DIMENSION_WEIGHTS["uniqueness"],
            details=details,
            issues=issues,
        )

    def _assess_validity(self, df: pd.DataFrame, contract=None) -> DimensionScore:
        """Dimension 3: Do values match expected types and constraints?"""
        issues = []
        validity_score = 100.0
        details = {}

        if contract and hasattr(contract, "schema_contracts"):
            for sc in contract.schema_contracts:
                if sc.column_name not in df.columns:
                    issues.append(f"Missing expected column: {sc.column_name}")
                    validity_score -= 5
                    continue

                col = df[sc.column_name]

                # Check allowed values
                if sc.allowed_values:
                    invalid = set(col.dropna().unique()) - set(sc.allowed_values)
                    if invalid:
                        invalid_pct = len(invalid) / max(col.nunique(), 1) * 100
                        issues.append(
                            f"Column '{sc.column_name}': {len(invalid)} unexpected values"
                        )
                        validity_score -= min(invalid_pct, 10)
                        details[sc.column_name] = {"invalid_values": list(invalid)[:5]}
        else:
            # Without a contract, do basic type consistency checks
            for col in df.columns:
                if df[col].dtype == "object":
                    # Check for mixed types in string columns
                    try:
                        numeric_count = pd.to_numeric(df[col], errors="coerce").notna().sum()
                        total_count = df[col].notna().sum()
                        if 0 < numeric_count < total_count * 0.9:
                            mixed_pct = numeric_count / total_count * 100
                            issues.append(
                                f"Column '{col}': mixed types ({mixed_pct:.0f}% look numeric)"
                            )
                            validity_score -= 3
                    except Exception:
                        pass

        return DimensionScore(
            dimension="validity",
            score=max(0, round(validity_score, 2)),
            weight=self.DIMENSION_WEIGHTS["validity"],
            details=details,
            issues=issues,
        )

    def _assess_consistency(self, df: pd.DataFrame) -> DimensionScore:
        """Dimension 4: Cross-column logical consistency checks."""
        issues = []
        consistency_score = 100.0
        details = {}

        # Check: numeric columns with suspicious distributions
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_count = int(
                    ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
                )
                if outlier_count > 0:
                    outlier_pct = outlier_count / len(df) * 100
                    details[col] = {"outliers": outlier_count, "outlier_pct": round(outlier_pct, 2)}
                    if outlier_pct > 5:
                        issues.append(
                            f"Column '{col}': {outlier_pct:.1f}% extreme outliers"
                        )
                        consistency_score -= 5

        # Check: categorical columns with very rare values (possible typos)
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            value_counts = df[col].value_counts()
            total = value_counts.sum()
            rare = value_counts[value_counts < max(2, total * 0.001)]
            if len(rare) > 0 and len(value_counts) > 5:
                details[col] = {"rare_values": list(rare.index[:5])}
                issues.append(
                    f"Column '{col}': {len(rare)} very rare values (possible typos)"
                )
                consistency_score -= 2

        return DimensionScore(
            dimension="consistency",
            score=max(0, round(consistency_score, 2)),
            weight=self.DIMENSION_WEIGHTS["consistency"],
            details=details,
            issues=issues,
        )

    def _assess_timeliness(self, last_modified: Optional[datetime] = None) -> DimensionScore:
        """Dimension 5: How fresh is the data?"""
        if last_modified is None:
            return DimensionScore(
                dimension="timeliness",
                score=80.0,  # Unknown freshness → assume acceptable
                weight=self.DIMENSION_WEIGHTS["timeliness"],
                details={"status": "unknown", "last_modified": None},
                issues=["Data freshness unknown — no timestamp available"],
            )

        now = datetime.now(timezone.utc)
        age_hours = (now - last_modified).total_seconds() / 3600
        details = {
            "last_modified": last_modified.isoformat(),
            "age_hours": round(age_hours, 1),
        }

        issues = []
        if age_hours > 168:  # > 1 week
            score = 30.0
            issues.append(f"CRITICAL: Data is {age_hours:.0f} hours old (>1 week)")
        elif age_hours > 48:
            score = 60.0
            issues.append(f"Warning: Data is {age_hours:.0f} hours old")
        elif age_hours > 24:
            score = 80.0
        else:
            score = 100.0

        return DimensionScore(
            dimension="timeliness",
            score=score,
            weight=self.DIMENSION_WEIGHTS["timeliness"],
            details=details,
            issues=issues,
        )

    # ── Column-level quality ──────────────────────────────────────

    def _assess_columns(self, df: pd.DataFrame, contract=None) -> List[ColumnQuality]:
        """Per-column quality scores."""
        results = []
        for col in df.columns:
            series = df[col]
            completeness = round((1 - series.isnull().mean()) * 100, 2)
            uniqueness = round(series.nunique() / max(len(series), 1) * 100, 2)

            issues = []
            validity = 100.0
            if contract and hasattr(contract, "schema_contracts"):
                sc = next((s for s in contract.schema_contracts if s.column_name == col), None)
                if sc and sc.allowed_values:
                    valid_count = series.isin(sc.allowed_values).sum() + series.isnull().sum()
                    validity = round(valid_count / len(series) * 100, 2)
                    if validity < 95:
                        issues.append(f"Only {validity}% of values are in allowed list")

            results.append(ColumnQuality(
                column_name=col,
                completeness=completeness,
                uniqueness=uniqueness,
                validity=validity,
                issues=issues,
            ))
        return results

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def get_report(self, product_name: str) -> Optional[QualityReport]:
        return self._reports.get(product_name)

    def get_all_reports(self) -> Dict[str, QualityReport]:
        return self._reports

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all quality assessments."""
        if not self._reports:
            return {"total_assessed": 0}
        return {
            "total_assessed": len(self._reports),
            "average_score": round(
                sum(r.composite_score for r in self._reports.values()) / len(self._reports), 1
            ),
            "grades": {name: r.grade for name, r in self._reports.items()},
            "scores": {name: r.composite_score for name, r in self._reports.items()},
            "total_issues": sum(r.total_issues for r in self._reports.values()),
        }
