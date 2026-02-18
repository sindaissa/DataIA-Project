"""
Layer 4 — Data Mesh: Data Products with Contracts & Ownership
──────────────────────────────────────────────────────────────
Implements the 4 Data Mesh principles (Zhamak Dehghani):
  1. Domain-oriented ownership — each product has an owner/team
  2. Data as a Product — SLAs, contracts, versioning, output ports
  3. Self-serve platform — registry allows domain teams to publish
  4. Federated governance — contracts enforce global + domain policies
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from enum import Enum
import pandas as pd
from pydantic import BaseModel, Field
from ados.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA CONTRACT — Schema contract, SLAs, quality expectations
# ═══════════════════════════════════════════════════════════════════════

class OutputPort(str, Enum):
    """Consumption formats — Data as a Product must offer multiple ports."""
    CSV = "csv"
    GRAPH = "graph"     # Neo4j property graph
    API = "api"         # REST endpoint via FastAPI
    PARQUET = "parquet"
    DATAFRAME = "dataframe"


class SLA(BaseModel):
    """Service Level Agreement for a Data Product."""
    freshness_hours: int = 24              # Max age before stale
    availability_percent: float = 99.0     # Uptime target
    max_query_time_ms: int = 5000          # Max response time
    support_contact: str = ""              # Escalation contact


class SchemaContract(BaseModel):
    """Schema contract — columns, types, nullability rules."""
    column_name: str
    expected_type: str                      # e.g. "int64", "object", "float64"
    nullable: bool = True                   # Is NULL allowed?
    unique: bool = False                    # Must be unique?
    allowed_values: Optional[List[str]] = None  # Enum constraint for categoricals
    description: str = ""                   # Business description


class QualityExpectation(BaseModel):
    """Quality expectations enforced by the contract."""
    min_completeness: float = 0.95          # % non-null rows required
    min_uniqueness: float = 0.0             # % unique values required
    min_validity: float = 0.90              # % rows matching type/allowed_values
    max_duplicates_percent: float = 5.0     # Max % duplicate rows allowed


class DataContract(BaseModel):
    """
    Data Contract — the binding agreement of a Data Product.
    Defines: who owns it, what shape it has, what quality is guaranteed.
    """
    version: str = "1.0.0"
    owner: str = "unknown"                  # Domain owner (person or team)
    team: str = "data-team"                 # Responsible domain team
    domain: str = "default"                 # Business domain
    description: str = ""
    sla: SLA = Field(default_factory=SLA)
    schema_contracts: List[SchemaContract] = Field(default_factory=list)
    quality: QualityExpectation = Field(default_factory=QualityExpectation)
    output_ports: List[OutputPort] = Field(
        default_factory=lambda: [OutputPort.CSV, OutputPort.GRAPH, OutputPort.DATAFRAME]
    )
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def validate_against_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate actual data against the contract — returns violations."""
        violations = []
        for sc in self.schema_contracts:
            if sc.column_name not in df.columns:
                violations.append(f"Missing column: {sc.column_name}")
                continue
            col = df[sc.column_name]
            # Nullability check
            if not sc.nullable and col.isnull().any():
                null_pct = col.isnull().mean() * 100
                violations.append(
                    f"{sc.column_name}: has {null_pct:.1f}% nulls but nullable=False"
                )
            # Uniqueness check
            if sc.unique and col.duplicated().any():
                dup_pct = col.duplicated().mean() * 100
                violations.append(
                    f"{sc.column_name}: has {dup_pct:.1f}% duplicates but unique=True"
                )
            # Allowed values check
            if sc.allowed_values:
                invalid = set(col.dropna().unique()) - set(sc.allowed_values)
                if invalid:
                    violations.append(
                        f"{sc.column_name}: invalid values {invalid} "
                        f"(allowed: {sc.allowed_values})"
                    )
        return {
            "contract_version": self.version,
            "violations": violations,
            "is_compliant": len(violations) == 0,
            "violation_count": len(violations),
        }


# ═══════════════════════════════════════════════════════════════════════
# CSV DATA PRODUCT — with contract & domain ownership
# ═══════════════════════════════════════════════════════════════════════

class CSVDataProduct:
    """
    A Data Product backed by a real CSV file.
    Implements Data Mesh principle #2: Data as a Product.
    - Has an owner and domain (principle #1)
    - Exposes multiple output ports (principle #2)
    - Publishes via a self-serve registry (principle #3)
    - Enforces a DataContract (principle #4)
    """

    def __init__(self, csv_path: str | Path, contract: Optional[DataContract] = None):
        self.csv_path = Path(csv_path)
        self.domain_name = self.csv_path.stem
        self._df: Optional[pd.DataFrame] = None
        self._schema: Dict[str, str] = {}
        self._stats: Dict[str, Any] = {}
        self.contract: DataContract = contract or self._auto_generate_contract()
        self._contract_validation: Dict[str, Any] = {}

    def _auto_generate_contract(self) -> DataContract:
        """Auto-generate a DataContract from loaded data (self-serve)."""
        return DataContract(
            owner="auto-detected",
            team="data-platform",
            domain=self.domain_name.split("_")[0] if "_" in self.domain_name else "general",
            description=f"Auto-generated contract for {self.domain_name}",
            tags=[self.domain_name],
        )

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        return self._df

    def load(self) -> pd.DataFrame:
        """Load CSV from disk — the only data source."""
        logger.info(f"Loading CSV: {self.csv_path}")
        self._df = pd.read_csv(self.csv_path)

        # Auto-detect schema
        self._schema = {
            col: str(dtype) for col, dtype in self._df.dtypes.items()
        }

        # Compute statistics for the LLM context
        self._stats = {
            "rows": len(self._df),
            "columns": list(self._df.columns),
            "dtypes": self._schema,
            "null_counts": self._df.isnull().sum().to_dict(),
            "unique_counts": {c: int(self._df[c].nunique()) for c in self._df.columns},
            "categorical_values": {
                col: self._df[col].value_counts().head(10).to_dict()
                for col in self._df.select_dtypes(include=["object"]).columns
                if self._df[col].nunique() <= 20
            },
            "numeric_stats": {
                col: {
                    "min": float(self._df[col].min()),
                    "max": float(self._df[col].max()),
                    "mean": float(self._df[col].mean()),
                    "median": float(self._df[col].median()),
                }
                for col in self._df.select_dtypes(include=["int64", "float64"]).columns
            },
        }

        logger.info(
            f"Loaded {self.domain_name}: {self._stats['rows']} rows, "
            f"{len(self._stats['columns'])} columns"
        )

        # Enrich the contract with auto-detected schema contracts
        if not self.contract.schema_contracts:
            self._enrich_contract_from_data()

        # Validate data against contract
        self._contract_validation = self.contract.validate_against_data(self._df)
        if not self._contract_validation["is_compliant"]:
            logger.warning(
                f"Contract violations for {self.domain_name}: "
                f"{self._contract_validation['violations']}"
            )
        else:
            logger.info(f"Contract validated ✔ for {self.domain_name}")

        return self._df

    def _enrich_contract_from_data(self) -> None:
        """Auto-populate schema contracts from actual data (self-serve principle)."""
        for col in self._df.columns:
            dtype = str(self._df[col].dtype)
            nunique = int(self._df[col].nunique())
            null_pct = self._df[col].isnull().mean()

            allowed = None
            if dtype == "object" and nunique <= 20:
                allowed = list(self._df[col].dropna().unique())

            self.contract.schema_contracts.append(SchemaContract(
                column_name=col,
                expected_type=dtype,
                nullable=null_pct > 0,
                unique=nunique == len(self._df),
                allowed_values=allowed,
                description=f"Auto-detected: {dtype}, {nunique} unique values",
            ))

    @property
    def contract_status(self) -> Dict[str, Any]:
        """Return contract compliance status."""
        return self._contract_validation

    @property
    def schema(self) -> Dict[str, str]:
        return self._schema

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

    def get_context_for_llm(self) -> str:
        """Build a rich text description for LLM agents (includes contract info)."""
        if self._df is None:
            return "No data loaded."

        lines = [
            f"## Data Product: {self.domain_name}",
            f"- Source: {self.csv_path.name}",
            f"- Owner: {self.contract.owner} ({self.contract.team})",
            f"- Domain: {self.contract.domain}",
            f"- Contract Version: {self.contract.version}",
            f"- SLA: freshness={self.contract.sla.freshness_hours}h, "
            f"availability={self.contract.sla.availability_percent}%",
            f"- Output Ports: {[p.value for p in self.contract.output_ports]}",
            f"- Rows: {self._stats['rows']}",
            f"- Columns ({len(self._stats['columns'])}): {', '.join(self._stats['columns'])}",
            f"- Contract Compliant: {self._contract_validation.get('is_compliant', 'N/A')}",
            "",
            "### Column Details:",
        ]

        for col in self._df.columns:
            dtype = self._schema[col]
            nunique = self._stats["unique_counts"][col]
            nulls = self._stats["null_counts"][col]
            line = f"  - **{col}** ({dtype}): {nunique} unique values, {nulls} nulls"

            if col in self._stats.get("categorical_values", {}):
                vals = list(self._stats["categorical_values"][col].keys())[:5]
                line += f" | values: {vals}"
            elif col in self._stats.get("numeric_stats", {}):
                ns = self._stats["numeric_stats"][col]
                line += f" | range [{ns['min']:.1f} — {ns['max']:.1f}], mean={ns['mean']:.1f}"

            lines.append(line)

        lines.append(f"\n### Sample (first 3 rows):")
        lines.append(self._df.head(3).to_markdown(index=False))

        return "\n".join(lines)

    def __repr__(self) -> str:
        rows = len(self._df) if self._df is not None else 0
        compliant = self._contract_validation.get("is_compliant", "?")
        return f"<CSVDataProduct:{self.domain_name} rows={rows} contract={compliant}>"


class DataProductRegistry:
    """
    Self-serve data platform — discovers and registers Data Products.
    Implements Data Mesh principle #3: Self-serve data infrastructure.
    Domain teams can publish their CSVs and the platform auto-generates contracts.
    """

    def __init__(self, csv_dir: str | Path):
        self.csv_dir = Path(csv_dir)
        self.products: Dict[str, CSVDataProduct] = {}
        self._contract_report: Dict[str, Any] = {}

    def discover_and_load(self) -> Dict[str, CSVDataProduct]:
        """Scan for CSV files, load them, auto-generate contracts, and validate."""
        csv_files = list(self.csv_dir.glob("*.csv"))
        logger.info(f"DataProductRegistry: found {len(csv_files)} CSV files in {self.csv_dir}")

        for csv_path in csv_files:
            product = CSVDataProduct(csv_path)
            product.load()
            self.products[product.domain_name] = product
            self._contract_report[product.domain_name] = product.contract_status
            logger.info(f"  ✔ Registered: {product}")

        return self.products

    def publish_product(self, csv_path: str | Path,
                        contract: Optional[DataContract] = None) -> CSVDataProduct:
        """
        Self-serve publishing — a domain team provides their CSV + optional contract.
        If no contract, one is auto-generated.
        """
        product = CSVDataProduct(csv_path, contract=contract)
        product.load()
        self.products[product.domain_name] = product
        self._contract_report[product.domain_name] = product.contract_status
        logger.info(f"Published data product: {product}")
        return product

    def get_compliance_report(self) -> Dict[str, Any]:
        """Get contract compliance report for all products."""
        return {
            "total_products": len(self.products),
            "compliant": sum(
                1 for r in self._contract_report.values()
                if r.get("is_compliant", False)
            ),
            "non_compliant": sum(
                1 for r in self._contract_report.values()
                if not r.get("is_compliant", True)
            ),
            "details": self._contract_report,
        }

    def get_all_context_for_llm(self) -> str:
        """Concatenate all product contexts for LLM."""
        return "\n\n---\n\n".join(
            p.get_context_for_llm() for p in self.products.values()
        )
