"""
Semantic Layer — Data Fabric Intelligence Component
────────────────────────────────────────────────────
Maps business terms to technical columns:
  - Business glossary (term → definition)
  - Term-to-column mapping (business name ↔ column name)
  - AI-driven semantic annotations (auto-tag columns)
  - Natural language disambiguation for LLM agents

This enables the LLM agents to understand business context,
not just raw column names.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field
from ados.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# BUSINESS GLOSSARY — defines business terms
# ═══════════════════════════════════════════════════════════════════════

class GlossaryTerm(BaseModel):
    """A business term with definition and related technical columns."""
    term: str                           # Business-friendly name
    definition: str                     # What it means in the business context
    synonyms: List[str] = Field(default_factory=list)   # Alternative names
    related_columns: List[str] = Field(default_factory=list)  # Technical columns
    domain: str = "general"             # Business domain
    category: str = "metric"            # metric | dimension | identifier | attribute


class ColumnAnnotation(BaseModel):
    """Semantic annotation attached to a technical column."""
    column_name: str
    business_name: str          # Human-readable name
    description: str            # What this column represents
    semantic_type: str          # identifier | dimension | measure | attribute | temporal
    business_domain: str = "general"
    sensitivity: str = "public"  # public | internal | restricted | confidential
    aggregation_hint: str = ""   # sum | avg | count | max | min | none
    format_hint: str = ""        # percentage | currency | count | category


# ═══════════════════════════════════════════════════════════════════════
# SEMANTIC LAYER
# ═══════════════════════════════════════════════════════════════════════

# Default semantic annotations for the Telco Churn dataset
TELCO_CHURN_ANNOTATIONS = {
    "customerID": ColumnAnnotation(
        column_name="customerID", business_name="Customer ID",
        description="Unique identifier for each customer",
        semantic_type="identifier", sensitivity="restricted",
    ),
    "gender": ColumnAnnotation(
        column_name="gender", business_name="Gender",
        description="Customer gender (Male/Female)",
        semantic_type="dimension", aggregation_hint="count",
    ),
    "SeniorCitizen": ColumnAnnotation(
        column_name="SeniorCitizen", business_name="Senior Citizen Status",
        description="Whether the customer is a senior citizen (1=Yes, 0=No)",
        semantic_type="dimension", format_hint="boolean",
    ),
    "Partner": ColumnAnnotation(
        column_name="Partner", business_name="Has Partner",
        description="Whether the customer has a partner",
        semantic_type="dimension",
    ),
    "Dependents": ColumnAnnotation(
        column_name="Dependents", business_name="Has Dependents",
        description="Whether the customer has dependents",
        semantic_type="dimension",
    ),
    "tenure": ColumnAnnotation(
        column_name="tenure", business_name="Customer Tenure",
        description="Number of months the customer has stayed with the company",
        semantic_type="measure", aggregation_hint="avg", format_hint="months",
    ),
    "PhoneService": ColumnAnnotation(
        column_name="PhoneService", business_name="Phone Service",
        description="Whether the customer has phone service",
        semantic_type="dimension",
    ),
    "MultipleLines": ColumnAnnotation(
        column_name="MultipleLines", business_name="Multiple Lines",
        description="Whether the customer has multiple phone lines",
        semantic_type="dimension",
    ),
    "InternetService": ColumnAnnotation(
        column_name="InternetService", business_name="Internet Service Type",
        description="Customer's internet service provider (DSL, Fiber optic, No)",
        semantic_type="dimension",
    ),
    "OnlineSecurity": ColumnAnnotation(
        column_name="OnlineSecurity", business_name="Online Security Service",
        description="Whether the customer has online security add-on",
        semantic_type="dimension",
    ),
    "OnlineBackup": ColumnAnnotation(
        column_name="OnlineBackup", business_name="Online Backup Service",
        description="Whether the customer has online backup add-on",
        semantic_type="dimension",
    ),
    "DeviceProtection": ColumnAnnotation(
        column_name="DeviceProtection", business_name="Device Protection Service",
        description="Whether the customer has device protection add-on",
        semantic_type="dimension",
    ),
    "TechSupport": ColumnAnnotation(
        column_name="TechSupport", business_name="Tech Support Service",
        description="Whether the customer has tech support add-on",
        semantic_type="dimension",
    ),
    "StreamingTV": ColumnAnnotation(
        column_name="StreamingTV", business_name="Streaming TV Service",
        description="Whether the customer has streaming TV add-on",
        semantic_type="dimension",
    ),
    "StreamingMovies": ColumnAnnotation(
        column_name="StreamingMovies", business_name="Streaming Movies Service",
        description="Whether the customer has streaming movies add-on",
        semantic_type="dimension",
    ),
    "Contract": ColumnAnnotation(
        column_name="Contract", business_name="Contract Type",
        description="Type of contract: Month-to-month, One year, Two year",
        semantic_type="dimension",
    ),
    "PaperlessBilling": ColumnAnnotation(
        column_name="PaperlessBilling", business_name="Paperless Billing",
        description="Whether the customer uses paperless billing",
        semantic_type="dimension",
    ),
    "PaymentMethod": ColumnAnnotation(
        column_name="PaymentMethod", business_name="Payment Method",
        description="How the customer pays: Electronic check, Mailed check, Bank transfer, Credit card",
        semantic_type="dimension",
    ),
    "MonthlyCharges": ColumnAnnotation(
        column_name="MonthlyCharges", business_name="Monthly Charges",
        description="Amount charged to the customer monthly",
        semantic_type="measure", aggregation_hint="avg", format_hint="currency",
    ),
    "TotalCharges": ColumnAnnotation(
        column_name="TotalCharges", business_name="Total Charges",
        description="Total amount charged to the customer (stored as text, cast to DOUBLE for calculations)",
        semantic_type="measure", aggregation_hint="sum", format_hint="currency",
    ),
    "Churn": ColumnAnnotation(
        column_name="Churn", business_name="Customer Churn",
        description="Whether the customer left the company (Yes/No). Key target variable.",
        semantic_type="measure", aggregation_hint="count", format_hint="category",
    ),
    "PromptInput": ColumnAnnotation(
        column_name="PromptInput", business_name="Prompt Input",
        description="Generated prompt input for AI analysis",
        semantic_type="attribute", sensitivity="internal",
    ),
    "CustomerFeedback": ColumnAnnotation(
        column_name="CustomerFeedback", business_name="Customer Feedback",
        description="Free-text customer feedback and comments",
        semantic_type="attribute", sensitivity="internal",
    ),
}

# Default business glossary for the Telco Churn domain
TELCO_CHURN_GLOSSARY = [
    GlossaryTerm(
        term="Churn Rate", definition="Percentage of customers who left the company",
        synonyms=["attrition rate", "taux de churn", "taux d'attrition"],
        related_columns=["Churn"], domain="customer", category="metric",
    ),
    GlossaryTerm(
        term="Customer Lifetime Value", definition="Total revenue generated by a customer",
        synonyms=["CLV", "CLTV", "valeur client"],
        related_columns=["TotalCharges", "tenure", "MonthlyCharges"],
        domain="revenue", category="metric",
    ),
    GlossaryTerm(
        term="ARPU", definition="Average Revenue Per User — monthly charges averaged across customers",
        synonyms=["revenu moyen par utilisateur", "average revenue"],
        related_columns=["MonthlyCharges"], domain="revenue", category="metric",
    ),
    GlossaryTerm(
        term="Contract Type", definition="The duration commitment of the customer's subscription",
        synonyms=["type de contrat", "subscription plan", "engagement"],
        related_columns=["Contract"], domain="product", category="dimension",
    ),
    GlossaryTerm(
        term="Service Bundle", definition="Combination of services a customer subscribes to",
        synonyms=["bouquet de services", "service package"],
        related_columns=["PhoneService", "InternetService", "OnlineSecurity",
                         "OnlineBackup", "DeviceProtection", "TechSupport",
                         "StreamingTV", "StreamingMovies"],
        domain="product", category="dimension",
    ),
    GlossaryTerm(
        term="Customer Tenure", definition="How long a customer has been with the company, in months",
        synonyms=["ancienneté", "durée d'abonnement", "customer age"],
        related_columns=["tenure"], domain="customer", category="measure",
    ),
    GlossaryTerm(
        term="Senior Customer", definition="A customer aged 65 or older",
        synonyms=["client senior", "senior citizen", "elderly customer"],
        related_columns=["SeniorCitizen"], domain="customer", category="dimension",
    ),
]


class SemanticLayer:
    """
    Semantic Layer — bridges the gap between business language and technical data.
    
    Key capabilities:
    1. Business glossary — define and resolve business terms
    2. Column annotations — rich metadata for each column
    3. Term resolution — map user queries to relevant columns
    4. Context enrichment — enhance LLM prompts with business semantics
    """

    def __init__(self):
        self._glossary: Dict[str, GlossaryTerm] = {}
        self._annotations: Dict[str, Dict[str, ColumnAnnotation]] = {}  # product → {col → annotation}
        self._term_index: Dict[str, Set[str]] = {}  # lowercase term/synonym → glossary key

    def load_defaults(self, product_name: str = "telco_churn_with_all_feedback") -> None:
        """Load default glossary and annotations for the Telco Churn dataset."""
        # Load glossary
        for term in TELCO_CHURN_GLOSSARY:
            self.add_glossary_term(term)

        # Load annotations
        for col_name, annotation in TELCO_CHURN_ANNOTATIONS.items():
            self.annotate_column(product_name, annotation)

        logger.info(
            f"SemanticLayer: loaded {len(self._glossary)} glossary terms, "
            f"{len(TELCO_CHURN_ANNOTATIONS)} annotations for '{product_name}'"
        )

    def add_glossary_term(self, term: GlossaryTerm) -> None:
        """Add a business term to the glossary."""
        self._glossary[term.term] = term
        # Index term and synonyms for fast lookup
        self._term_index.setdefault(term.term.lower(), set()).add(term.term)
        for syn in term.synonyms:
            self._term_index.setdefault(syn.lower(), set()).add(term.term)

    def annotate_column(self, product_name: str, annotation: ColumnAnnotation) -> None:
        """Add a semantic annotation to a column."""
        self._annotations.setdefault(product_name, {})[annotation.column_name] = annotation

    def resolve_term(self, user_input: str) -> List[GlossaryTerm]:
        """
        Resolve a user's business term to glossary entries.
        Supports exact match, synonym match, and partial match.
        """
        user_lower = user_input.lower()
        results = []

        # Exact match
        for key, terms in self._term_index.items():
            if key == user_lower:
                for t in terms:
                    results.append(self._glossary[t])

        # Partial match (if no exact)
        if not results:
            for key, terms in self._term_index.items():
                if user_lower in key or key in user_lower:
                    for t in terms:
                        if self._glossary[t] not in results:
                            results.append(self._glossary[t])

        return results

    def get_columns_for_term(self, term: str) -> List[str]:
        """Get all technical columns related to a business term."""
        resolved = self.resolve_term(term)
        columns = []
        for gt in resolved:
            columns.extend(gt.related_columns)
        return list(set(columns))

    def get_semantic_context(self, product_name: str) -> str:
        """
        Build a rich semantic context for LLM agents.
        This enriches the raw schema with business meaning.
        """
        lines = ["## Business Semantic Layer", ""]

        # Glossary
        lines.append("### Business Glossary:")
        for term in self._glossary.values():
            syns = f" (aka: {', '.join(term.synonyms[:3])})" if term.synonyms else ""
            lines.append(f"  - **{term.term}**{syns}: {term.definition}")
            if term.related_columns:
                lines.append(f"    → Columns: {', '.join(term.related_columns)}")

        # Column annotations
        annotations = self._annotations.get(product_name, {})
        if annotations:
            lines.append(f"\n### Column Semantics for '{product_name}':")
            for col, ann in annotations.items():
                lines.append(
                    f"  - **{col}** → \"{ann.business_name}\" ({ann.semantic_type}): "
                    f"{ann.description}"
                )
                if ann.aggregation_hint:
                    lines.append(f"    Aggregation: {ann.aggregation_hint}")

        return "\n".join(lines)

    def enrich_query_context(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze a user query and enrich it with semantic information.
        Returns resolved terms, suggested columns, and business context.
        """
        words = user_query.lower().split()
        resolved_terms = []
        suggested_columns = set()

        # Try multi-word combinations
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                phrase = " ".join(words[i:j])
                matches = self.resolve_term(phrase)
                for m in matches:
                    if m not in resolved_terms:
                        resolved_terms.append(m)
                        suggested_columns.update(m.related_columns)

        return {
            "resolved_terms": [t.term for t in resolved_terms],
            "suggested_columns": list(suggested_columns),
            "definitions": {t.term: t.definition for t in resolved_terms},
            "enriched": len(resolved_terms) > 0,
        }

    def get_annotation(self, product_name: str, column_name: str) -> Optional[ColumnAnnotation]:
        return self._annotations.get(product_name, {}).get(column_name)

    def summary(self) -> Dict[str, Any]:
        return {
            "glossary_terms": len(self._glossary),
            "annotated_products": len(self._annotations),
            "total_annotations": sum(len(a) for a in self._annotations.values()),
            "domains": list(set(t.domain for t in self._glossary.values())),
        }
