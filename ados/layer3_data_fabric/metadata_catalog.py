"""
Active Metadata Catalog — Data Fabric Core Component
─────────────────────────────────────────────────────
Enhanced metadata catalog with:
  - Active metadata: usage tracking, anomaly alerts, recommendations
  - Quality scores integration
  - Semantic annotations
  - Data profiling and change detection
  - AI-driven recommendations for data consumers

Active metadata = metadata that DRIVES automation, not just documents.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ados.logging_config import get_logger

logger = get_logger(__name__)


class ColumnMeta(BaseModel):
    name: str
    data_type: str
    nunique: int = 0
    null_count: int = 0
    sample_values: List[Any] = Field(default_factory=list)
    # Active metadata enrichments
    business_name: str = ""         # Semantic layer annotation
    description: str = ""           # Business description
    sensitivity: str = "public"     # PII classification from governance
    semantic_type: str = ""         # identifier | dimension | measure | attribute


class DataProductEntry(BaseModel):
    domain_name: str
    file_path: str
    columns: List[ColumnMeta] = Field(default_factory=list)
    row_count: int = 0
    quality_score: Optional[float] = None
    quality_grade: Optional[str] = None
    owner: str = "unknown"
    domain: str = "general"
    contract_compliant: Optional[bool] = None
    tags: List[str] = Field(default_factory=list)
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UsageRecord(BaseModel):
    """Track how data products are queried (active metadata)."""
    product_name: str
    query: str
    columns_accessed: List[str] = Field(default_factory=list)
    user_role: str = "analyst"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MetadataAlert(BaseModel):
    """Active metadata alert — triggered by profiling or quality changes."""
    alert_type: str     # quality_drop | schema_change | usage_spike | stale_data
    severity: str       # info | warning | critical
    product_name: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MetadataCatalog:
    """
    Active Metadata Catalog — the brain of the Data Fabric.
    
    Beyond passive schema storage, it:
    1. Tracks usage patterns (which products/columns are queried most)
    2. Generates alerts when quality drops or schemas change
    3. Provides AI-driven recommendations for data consumers
    4. Integrates quality scores and semantic annotations
    5. Maintains a change log for auditability
    """

    def __init__(self):
        self._products: Dict[str, DataProductEntry] = {}
        self._usage_log: List[UsageRecord] = []
        self._alerts: List[MetadataAlert] = []
        self._change_log: List[Dict[str, Any]] = []

    def register_from_product(self, product) -> None:
        """Register a data product with enriched metadata."""
        columns = [
            ColumnMeta(
                name=col,
                data_type=product.schema.get(col, "unknown"),
                nunique=product.stats.get("unique_counts", {}).get(col, 0),
                null_count=product.stats.get("null_counts", {}).get(col, 0),
                sample_values=(
                    list(product.stats.get("categorical_values", {}).get(col, {}).keys())[:5]
                    if col in product.stats.get("categorical_values", {})
                    else []
                ),
            )
            for col in product.stats.get("columns", [])
        ]

        # Extract contract metadata if available
        owner = "unknown"
        domain = "general"
        contract_compliant = None
        tags = []
        if hasattr(product, "contract") and product.contract:
            owner = product.contract.owner
            domain = product.contract.domain
            tags = product.contract.tags
            if hasattr(product, "contract_status") and product.contract_status:
                contract_compliant = product.contract_status.get("is_compliant")

        entry = DataProductEntry(
            domain_name=product.domain_name,
            file_path=str(product.csv_path),
            columns=columns,
            row_count=product.stats.get("rows", 0),
            owner=owner,
            domain=domain,
            contract_compliant=contract_compliant,
            tags=tags,
        )
        self._products[product.domain_name] = entry

        # Log the change
        self._change_log.append({
            "action": "register",
            "product": product.domain_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rows": entry.row_count,
            "columns": len(entry.columns),
        })
        logger.info(f"Catalog: registered '{product.domain_name}' ({entry.row_count} rows, owner={owner})")

    def enrich_with_quality(self, product_name: str, quality_score: float, quality_grade: str) -> None:
        """Enrich a product entry with quality assessment results (active metadata)."""
        entry = self._products.get(product_name)
        if entry:
            old_score = entry.quality_score
            entry.quality_score = quality_score
            entry.quality_grade = quality_grade

            # Generate alert if quality drops
            if old_score is not None and quality_score < old_score - 10:
                self._alerts.append(MetadataAlert(
                    alert_type="quality_drop",
                    severity="warning",
                    product_name=product_name,
                    message=f"Quality dropped from {old_score:.1f} to {quality_score:.1f}",
                ))
                logger.warning(f"Catalog ALERT: quality drop for '{product_name}'")

            logger.info(f"Catalog: enriched '{product_name}' with quality={quality_score:.1f} ({quality_grade})")

    def enrich_with_semantics(self, product_name: str, semantic_layer) -> None:
        """Enrich column metadata with semantic annotations (active metadata)."""
        entry = self._products.get(product_name)
        if not entry:
            return

        annotations = semantic_layer._annotations.get(product_name, {})
        for col_meta in entry.columns:
            ann = annotations.get(col_meta.name)
            if ann:
                col_meta.business_name = ann.business_name
                col_meta.description = ann.description
                col_meta.sensitivity = ann.sensitivity
                col_meta.semantic_type = ann.semantic_type

        logger.info(f"Catalog: enriched '{product_name}' with {len(annotations)} semantic annotations")

    def record_usage(self, product_name: str, query: str,
                     columns: List[str], user_role: str = "analyst") -> None:
        """Track query usage for active metadata analytics."""
        self._usage_log.append(UsageRecord(
            product_name=product_name,
            query=query,
            columns_accessed=columns,
            user_role=user_role,
        ))

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage analytics — which products/columns are most accessed."""
        if not self._usage_log:
            return {"total_queries": 0}

        product_counts: Dict[str, int] = {}
        column_counts: Dict[str, int] = {}
        for rec in self._usage_log:
            product_counts[rec.product_name] = product_counts.get(rec.product_name, 0) + 1
            for col in rec.columns_accessed:
                column_counts[col] = column_counts.get(col, 0) + 1

        return {
            "total_queries": len(self._usage_log),
            "most_used_products": sorted(product_counts.items(), key=lambda x: -x[1])[:5],
            "most_used_columns": sorted(column_counts.items(), key=lambda x: -x[1])[:10],
            "roles": list(set(r.user_role for r in self._usage_log)),
        }

    def get_recommendations(self, product_name: str) -> List[str]:
        """AI-driven recommendations based on active metadata."""
        entry = self._products.get(product_name)
        if not entry:
            return []

        recs = []

        # Quality-based recommendations
        if entry.quality_score is not None:
            if entry.quality_score < 70:
                recs.append(
                    f"⚠️ Quality score is {entry.quality_score:.1f}/100 — investigate data issues"
                )
            if entry.quality_grade in ["D", "F"]:
                recs.append(
                    "🔴 Quality grade is below acceptable threshold — data contract review needed"
                )

        # Completeness recommendations
        for col in entry.columns:
            null_pct = (col.null_count / entry.row_count * 100) if entry.row_count > 0 else 0
            if null_pct > 20:
                recs.append(f"Column '{col.name}' has {null_pct:.0f}% nulls — consider imputation")

        # Contract compliance
        if entry.contract_compliant is False:
            recs.append("📋 Data contract violations detected — review contract compliance")

        # Owner recommendation
        if entry.owner == "unknown" or entry.owner == "auto-detected":
            recs.append("👤 No owner assigned — assign a domain owner for accountability")

        # Usage-based recommendations
        usage = self.get_usage_stats()
        if usage["total_queries"] > 10:
            popular = [c for c, _ in usage.get("most_used_columns", [])[:5]]
            if popular:
                recs.append(f"📊 Most queried columns: {', '.join(popular)} — consider pre-aggregation")

        return recs

    def get_alerts(self) -> List[MetadataAlert]:
        """Get all active metadata alerts."""
        return self._alerts

    def get_product(self, name: str) -> Optional[DataProductEntry]:
        return self._products.get(name)

    def list_products(self) -> List[str]:
        return list(self._products.keys())

    def list_all_columns(self) -> Dict[str, List[str]]:
        return {n: [c.name for c in e.columns] for n, e in self._products.items()}

    def get_schema_context(self) -> str:
        """Build an enriched text description of all schemas for LLM agents."""
        lines = []
        for name, entry in self._products.items():
            quality_info = ""
            if entry.quality_score is not None:
                quality_info = f", quality={entry.quality_score:.0f}/100 ({entry.quality_grade})"
            lines.append(
                f"Table: {name} ({entry.row_count} rows, owner={entry.owner}{quality_info})"
            )
            for col in entry.columns:
                vals = f" | values: {col.sample_values}" if col.sample_values else ""
                semantic = ""
                if col.business_name:
                    semantic = f" [{col.business_name}]"
                desc = f" — {col.description}" if col.description else ""
                lines.append(
                    f"  - {col.name}{semantic} ({col.data_type}, {col.nunique} unique){vals}{desc}"
                )
        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_products": len(self._products),
            "products": {
                n: {
                    "columns": len(e.columns), "rows": e.row_count,
                    "quality": e.quality_score, "grade": e.quality_grade,
                    "owner": e.owner, "domain": e.domain,
                    "contract_compliant": e.contract_compliant,
                }
                for n, e in self._products.items()
            },
            "total_queries": len(self._usage_log),
            "active_alerts": len(self._alerts),
            "change_log_entries": len(self._change_log),
        }
