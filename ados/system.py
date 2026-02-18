"""
ADOS System — Composition Root (DI Container)
──────────────────────────────────────────────
Wires: CSV Data Products → Contracts → Quality → Semantics → Governance
     → Neo4j KG → LLM Agents → LangGraph Pipeline → Grafana
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from ados.config import get_settings
from ados.layer4_data_mesh.data_product import DataProductRegistry
from ados.layer4_data_mesh.governance import FederatedGovernance
from ados.layer3_data_fabric.metadata_catalog import MetadataCatalog
from ados.layer3_data_fabric.lineage_service import DynamicLineageService
from ados.layer3_data_fabric.quality_engine import DataQualityEngine
from ados.layer3_data_fabric.semantic_layer import SemanticLayer
from ados.layer2_kernel.knowledge_graph import Neo4jKnowledgeGraph
from ados.layer2_kernel.orchestrator import LangGraphOrchestrator
from ados.logging_config import get_logger

logger = get_logger(__name__)


class ADOSSystem:
    """
    Composition root — wires all ADOS v2 components.

    Initialization (8 steps):
        1. Load CSV data products (no generation)
        2. Register in metadata catalog
        3. Run quality assessment (Data Fabric)
        4. Load semantic layer (Data Fabric)
        5. Enrich catalog with semantics + quality (Active Metadata)
        6. Run governance compliance checks (Data Mesh)
        7. Build Neo4j knowledge graph
        8. Wire LangGraph orchestrator with LLM agents

    Query:
        User query → Access check → Semantic enrichment → LangGraph pipeline → Certified result
    """

    def __init__(self):
        self._settings = get_settings()
        self.catalog = MetadataCatalog()
        self.lineage = DynamicLineageService()
        self.quality_engine = DataQualityEngine()
        self.semantic_layer = SemanticLayer()
        self.governance = FederatedGovernance()
        self.data_products: Dict[str, Any] = {}
        self.knowledge_graph: Neo4jKnowledgeGraph = None
        self.orchestrator: LangGraphOrchestrator = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        """
        Full initialization — 8 steps for complete Data Mesh + Data Fabric:
        """
        logger.info("═══ ADOS v2 System Initialization ═══")

        # Step 1: Load CSV data products (with auto-generated contracts)
        logger.info("▸ Step 1/8: Loading CSV data products with contracts...")
        registry = DataProductRegistry(self._settings.csv_dir)
        self.data_products = registry.discover_and_load()

        # Step 2: Register in catalog
        logger.info("▸ Step 2/8: Registering in metadata catalog...")
        for product in self.data_products.values():
            self.catalog.register_from_product(product)

        # Step 3: Data Quality Assessment (Data Fabric)
        logger.info("▸ Step 3/8: Running quality assessments...")
        for name, product in self.data_products.items():
            if product.dataframe is not None:
                report = self.quality_engine.assess(
                    name, product.dataframe, contract=product.contract
                )
                self.catalog.enrich_with_quality(name, report.composite_score, report.grade)

        # Step 4: Load Semantic Layer (Data Fabric)
        logger.info("▸ Step 4/8: Loading semantic layer...")
        self.semantic_layer.load_defaults()

        # Step 5: Enrich catalog with semantic annotations (Active Metadata)
        logger.info("▸ Step 5/8: Enriching catalog with semantics...")
        for name in self.data_products:
            self.catalog.enrich_with_semantics(name, self.semantic_layer)

        # Step 6: Governance compliance checks (Data Mesh)
        logger.info("▸ Step 6/8: Running governance compliance checks...")
        for name, product in self.data_products.items():
            # Register access policies (auto-detects PII)
            columns = list(product.schema.keys())
            self.governance.register_access_policy(name, columns)
            # Run compliance checks
            quality_score = self.quality_engine.get_report(name)
            score = quality_score.composite_score if quality_score else None
            self.governance.run_compliance_check(product, quality_score=score)

        # Step 7: Build Neo4j Knowledge Graph
        logger.info("▸ Step 7/8: Building Neo4j knowledge graph...")
        try:
            self.knowledge_graph = Neo4jKnowledgeGraph(
                uri=self._settings.neo4j.uri,
                user=self._settings.neo4j.user,
                password=self._settings.neo4j.password,
            )
            self.knowledge_graph.build_from_catalog(self.catalog, self.data_products)
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}. Continuing without KG.")
            self.knowledge_graph = _FallbackKG()

        # Step 8: Wire LangGraph orchestrator
        logger.info("▸ Step 8/8: Wiring LangGraph orchestrator...")
        self.orchestrator = LangGraphOrchestrator(
            catalog=self.catalog,
            knowledge_graph=self.knowledge_graph,
            lineage=self.lineage,
        )

        self._initialized = True
        logger.info("═══ ADOS v2 System Ready ═══")

    def query(self, user_query: str, user_role: str = "analyst") -> Dict[str, Any]:
        """Process a query through the LangGraph pipeline with governance checks."""
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")

        # Governance: check access
        for name in self.data_products:
            access = self.governance.check_access(name, user_role, list(self.data_products[name].schema.keys()))
            if not access["allowed"]:
                logger.warning(f"Access denied for '{user_role}' on '{name}': {access['reason']}")

        # Semantic enrichment
        enrichment = self.semantic_layer.enrich_query_context(user_query)
        if enrichment.get("enriched"):
            logger.info(f"Semantic enrichment: resolved terms={enrichment['resolved_terms']}")

        # Run pipeline
        result = self.orchestrator.process_query(user_query, user_role)

        # Governance: validate generated Cypher
        cypher = result.get("sql", "")
        if cypher:
            cypher_check = self.governance.validate_query(cypher)
            result["governance_query_check"] = cypher_check.status
            if cypher_check.status == "fail":
                logger.error(f"Governance: Cypher rejected — {cypher_check.message}")

        # Active metadata: record usage
        products_used = result.get("discovery", {}).get("relevant_products", [])
        columns_used = []
        for cols in result.get("discovery", {}).get("relevant_columns", {}).values():
            columns_used.extend(cols)
        for p in products_used:
            self.catalog.record_usage(p, user_query, columns_used, user_role)

        # Add quality + governance info to result
        result["semantic_enrichment"] = enrichment
        result["quality_scores"] = self.quality_engine.get_summary()
        result["governance_status"] = self.governance.get_compliance_summary()

        return result

    def print_status(self) -> str:
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════╗",
            "║           ADOS v2 — System Status                      ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Status: {'✔ READY' if self._initialized else '✘ NOT INITIALIZED'}",
            f"║  Architecture: LLM Agents + LangGraph + Neo4j + Grafana",
            f"║  LLM: Groq ({self._settings.llm.model_name})",
            f"║  API Key: {'✔ SET' if self._settings.llm.api_key else '✘ MISSING'}",
            f"║  Data Products: {len(self.data_products)}",
        ]
        for name, product in self.data_products.items():
            rows = len(product.dataframe) if product.dataframe is not None else 0
            cols = len(product.schema)
            owner = product.contract.owner if hasattr(product, 'contract') else "?"
            lines.append(f"║    📦 {name}: {rows} rows, {cols} columns, owner={owner}")

        # Quality summary
        quality = self.quality_engine.get_summary()
        if quality.get("total_assessed"):
            lines.append(f"║  Quality: avg={quality['average_score']:.1f}/100")
            for name, grade in quality.get("grades", {}).items():
                score = quality["scores"][name]
                lines.append(f"║    📊 {name}: {score:.1f}/100 (Grade {grade})")

        # Semantic layer
        sem = self.semantic_layer.summary()
        lines.append(f"║  Semantic Layer: {sem['glossary_terms']} terms, {sem['total_annotations']} annotations")

        # Governance
        gov = self.governance.get_compliance_summary()
        lines.append(f"║  Governance: {gov['total_rules']} rules, {gov['access_policies']} policies")
        if gov.get("pii_products"):
            lines.append(f"║    ⚠️ PII detected in: {gov['pii_products']}")

        if self.knowledge_graph:
            stats = self.knowledge_graph.summary()
            lines.append(f"║  Neo4j KG: {stats.get('nodes', 0)} nodes, {stats.get('relationships', 0)} rels")

        lines.append(f"║  Lineage Traces: {len(self.lineage.get_all_traces())}")
        lines.append(f"║  Grafana: {self._settings.grafana.url}")
        lines.append("╚══════════════════════════════════════════════════════════╝")
        report = "\n".join(lines)
        print(report)
        return report


class _FallbackKG:
    """Minimal fallback when Neo4j is unavailable."""
    def get_context_for_llm(self) -> str:
        return "(Knowledge graph unavailable — Neo4j not connected)"
    def query_cypher(self, cypher: str) -> list:
        return []
    def summary(self) -> dict:
        return {"nodes": 0, "relationships": 0}
    def render_ascii(self) -> str:
        return "╔═══ KG: Neo4j not connected ═══╗"
    def get_relationships(self) -> list:
        return []
    def get_schema_graph(self) -> list:
        return []
