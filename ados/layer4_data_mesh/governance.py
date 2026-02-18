"""
Federated Governance — Data Mesh Principle #4
─────────────────────────────────────────────
Global policies + domain-specific rules.
Enforces access control, PII detection, compliance, and interoperability.
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import re
from pydantic import BaseModel, Field
from ados.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ACCESS CONTROL — Role-based data access
# ═══════════════════════════════════════════════════════════════════════

class AccessLevel(str, Enum):
    PUBLIC = "public"           # Anyone can access
    INTERNAL = "internal"       # Internal team members
    RESTRICTED = "restricted"   # Need explicit approval
    CONFIDENTIAL = "confidential"  # PII / sensitive data


class AccessPolicy(BaseModel):
    """Who can access what columns and at what level."""
    product_name: str
    access_level: AccessLevel = AccessLevel.INTERNAL
    allowed_roles: List[str] = Field(default_factory=lambda: ["analyst", "manager", "data_engineer"])
    restricted_columns: List[str] = Field(default_factory=list)  # Columns requiring higher access
    pii_columns: List[str] = Field(default_factory=list)          # Detected PII columns


# ═══════════════════════════════════════════════════════════════════════
# PII DETECTION — Automatic detection of personally identifiable info
# ═══════════════════════════════════════════════════════════════════════

# Patterns that suggest PII in column names
PII_PATTERNS = [
    r"(?i)(customer.*id|client.*id|user.*id)",
    r"(?i)(name|first.?name|last.?name|full.?name)",
    r"(?i)(email|e.?mail|mail)",
    r"(?i)(phone|telephone|mobile|cell)",
    r"(?i)(address|street|city|zip|postal)",
    r"(?i)(ssn|social.?security|national.?id|passport)",
    r"(?i)(birth|dob|date.?of.?birth|age)",
    r"(?i)(credit.?card|card.?number|cvv|expiry)",
    r"(?i)(salary|income|compensation|wage)",
]


def detect_pii_columns(column_names: List[str]) -> List[str]:
    """Auto-detect columns that likely contain PII data."""
    pii = []
    for col in column_names:
        for pattern in PII_PATTERNS:
            if re.search(pattern, col):
                pii.append(col)
                break
    return pii


# ═══════════════════════════════════════════════════════════════════════
# GOVERNANCE POLICIES — Global + domain-level rules
# ═══════════════════════════════════════════════════════════════════════

class GovernanceRule(BaseModel):
    """A single governance rule."""
    rule_id: str
    name: str
    description: str
    severity: str = "warning"  # info | warning | error
    scope: str = "global"      # global | domain-specific
    domain: Optional[str] = None


class ComplianceResult(BaseModel):
    """Result of a compliance check."""
    rule_id: str
    rule_name: str
    status: str  # pass | fail | warning
    message: str
    severity: str


class FederatedGovernance:
    """
    Federated Governance Engine — Data Mesh Principle #4.
    
    Combines:
    - Global policies (apply to ALL data products)
    - Domain-specific policies (per domain overrides)
    - PII auto-detection
    - Role-based access control
    - Interoperability standards
    """

    def __init__(self):
        self._global_rules: List[GovernanceRule] = self._default_global_rules()
        self._domain_rules: Dict[str, List[GovernanceRule]] = {}
        self._access_policies: Dict[str, AccessPolicy] = {}
        self._compliance_history: List[Dict[str, Any]] = []

    def _default_global_rules(self) -> List[GovernanceRule]:
        """Default global governance rules applied to all data products."""
        return [
            GovernanceRule(
                rule_id="G001",
                name="Data Contract Required",
                description="Every data product must have a data contract",
                severity="error",
                scope="global",
            ),
            GovernanceRule(
                rule_id="G002",
                name="PII Detection",
                description="PII columns must be identified and access-restricted",
                severity="warning",
                scope="global",
            ),
            GovernanceRule(
                rule_id="G003",
                name="Minimum Quality Score",
                description="Data products must have quality score >= 70/100",
                severity="warning",
                scope="global",
            ),
            GovernanceRule(
                rule_id="G004",
                name="Schema Documentation",
                description="All columns must have descriptions in the contract",
                severity="info",
                scope="global",
            ),
            GovernanceRule(
                rule_id="G005",
                name="Owner Assignment",
                description="Every data product must have a designated owner",
                severity="error",
                scope="global",
            ),
            GovernanceRule(
                rule_id="G006",
                name="No Dangerous Queries",
                description="Generated queries must not contain DROP, DELETE, UPDATE, INSERT, DETACH DELETE, SET, REMOVE",
                severity="error",
                scope="global",
            ),
        ]

    def add_domain_rule(self, domain: str, rule: GovernanceRule) -> None:
        """Add a domain-specific governance rule."""
        rule.scope = "domain"
        rule.domain = domain
        self._domain_rules.setdefault(domain, []).append(rule)
        logger.info(f"Governance: added domain rule '{rule.name}' for {domain}")

    def register_access_policy(self, product_name: str, columns: List[str]) -> AccessPolicy:
        """Auto-generate access policy with PII detection."""
        pii_cols = detect_pii_columns(columns)
        policy = AccessPolicy(
            product_name=product_name,
            access_level=AccessLevel.RESTRICTED if pii_cols else AccessLevel.INTERNAL,
            pii_columns=pii_cols,
            restricted_columns=pii_cols,
        )
        self._access_policies[product_name] = policy
        if pii_cols:
            logger.warning(
                f"Governance: PII detected in '{product_name}': {pii_cols} → RESTRICTED access"
            )
        else:
            logger.info(f"Governance: access policy set for '{product_name}': INTERNAL")
        return policy

    def check_access(self, product_name: str, user_role: str, columns: List[str]) -> Dict[str, Any]:
        """Check if a user role can access specific columns."""
        policy = self._access_policies.get(product_name)
        if not policy:
            return {"allowed": True, "reason": "No access policy defined"}

        if user_role not in policy.allowed_roles:
            return {
                "allowed": False,
                "reason": f"Role '{user_role}' not in allowed roles: {policy.allowed_roles}",
            }

        restricted = [c for c in columns if c in policy.restricted_columns]
        if restricted and user_role not in ["manager", "data_engineer"]:
            return {
                "allowed": False,
                "reason": f"Columns {restricted} are restricted (PII). "
                          f"Role '{user_role}' needs escalation.",
                "restricted_columns": restricted,
            }

        return {"allowed": True, "reason": "Access granted"}

    def validate_query(self, query: str) -> ComplianceResult:
        """Check generated query (SQL or Cypher) against governance rules."""
        dangerous_sql = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "GRANT"]
        dangerous_cypher = ["DETACH DELETE", "DELETE", "SET ", "REMOVE ", "CREATE ", "MERGE "]
        query_upper = query.upper()

        # Check SQL-style dangerous ops
        found = [d for d in dangerous_sql if d in query_upper]

        # Check Cypher-style mutation ops (but allow MATCH...CREATE in read-like contexts)
        # Only flag standalone mutation keywords
        if not found:
            for d in dangerous_cypher:
                # "CREATE INDEX" is safe for setup, skip it
                if d.strip() == "CREATE" and "CREATE INDEX" in query_upper:
                    continue
                if d.strip() in query_upper:
                    # Ensure it's not inside a RETURN clause
                    idx = query_upper.find(d.strip())
                    before = query_upper[:idx]
                    if "RETURN" not in before:  # mutation before RETURN = write query
                        found.append(d.strip())

        if found:
            return ComplianceResult(
                rule_id="G006",
                rule_name="No Dangerous Queries",
                status="fail",
                message=f"Query contains mutation operations: {found}",
                severity="error",
            )
        return ComplianceResult(
            rule_id="G006",
            rule_name="No Dangerous Queries",
            status="pass",
            message="Query is safe (read-only)",
            severity="info",
        )

    # Keep backward-compatible alias
    def validate_sql(self, sql: str) -> ComplianceResult:
        return self.validate_query(sql)

    def run_compliance_check(self, product, quality_score: Optional[float] = None) -> List[ComplianceResult]:
        """Run all governance rules against a data product."""
        results = []

        # G001: Contract required
        has_contract = hasattr(product, "contract") and product.contract is not None
        results.append(ComplianceResult(
            rule_id="G001", rule_name="Data Contract Required",
            status="pass" if has_contract else "fail",
            message="Data contract exists" if has_contract else "No data contract defined",
            severity="error",
        ))

        # G002: PII detection
        if has_contract:
            policy = self._access_policies.get(product.domain_name)
            pii = policy.pii_columns if policy else []
            results.append(ComplianceResult(
                rule_id="G002", rule_name="PII Detection",
                status="pass" if not pii else "warning",
                message=f"PII columns detected: {pii}" if pii else "No PII detected",
                severity="warning",
            ))

        # G003: Minimum quality
        if quality_score is not None:
            results.append(ComplianceResult(
                rule_id="G003", rule_name="Minimum Quality Score",
                status="pass" if quality_score >= 70 else "fail",
                message=f"Quality score: {quality_score:.1f}/100",
                severity="warning",
            ))

        # G004: Schema documentation
        if has_contract:
            documented = sum(
                1 for sc in product.contract.schema_contracts if sc.description
            )
            total = len(product.contract.schema_contracts)
            results.append(ComplianceResult(
                rule_id="G004", rule_name="Schema Documentation",
                status="pass" if documented == total else "warning",
                message=f"{documented}/{total} columns documented",
                severity="info",
            ))

        # G005: Owner assignment
        if has_contract:
            owner = product.contract.owner
            results.append(ComplianceResult(
                rule_id="G005", rule_name="Owner Assignment",
                status="pass" if owner and owner != "unknown" else "fail",
                message=f"Owner: {owner}" if owner != "unknown" else "No owner assigned",
                severity="error",
            ))

        # Store compliance history
        self._compliance_history.append({
            "product": product.domain_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": [r.model_dump() for r in results],
            "overall": "pass" if all(r.status != "fail" for r in results) else "fail",
        })

        logger.info(
            f"Governance: compliance check for '{product.domain_name}': "
            f"{sum(1 for r in results if r.status == 'pass')}/{len(results)} passed"
        )
        return results

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get overall governance compliance summary."""
        return {
            "total_rules": len(self._global_rules),
            "domain_rules": {d: len(r) for d, r in self._domain_rules.items()},
            "access_policies": len(self._access_policies),
            "pii_products": [
                name for name, p in self._access_policies.items() if p.pii_columns
            ],
            "compliance_checks": len(self._compliance_history),
            "latest_results": self._compliance_history[-1] if self._compliance_history else None,
        }
