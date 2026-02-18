"""
FastAPI Application — ADOS REST API + Grafana JSON Datasource
──────────────────────────────────────────────────────────────
Exposes:
  - /api/v1/query          — Process a natural language query via LLM pipeline
  - /api/v1/catalog        — Metadata catalog
  - /api/v1/kg             — Knowledge graph summary
  - /api/v1/lineage        — Lineage traces
  - /api/v1/health         — Health check

Grafana JSON Datasource (SimpleJSON):
  - /grafana/              — Test endpoint
  - /grafana/search        — Available metrics
  - /grafana/query         — Query data for Grafana panels
  - /grafana/annotations   — Annotations
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ados.config import get_settings
from ados.logging_config import get_logger, set_correlation_id

logger = get_logger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    user_role: str = "analyst"


class QueryResponse(BaseModel):
    status: str
    user_query: str
    intent: Dict[str, Any] = {}
    discovery: Dict[str, Any] = {}
    cypher: str = ""
    trust: Dict[str, Any] = {}
    analysis: Dict[str, Any] = {}
    result_count: int = 0
    result_data: Optional[List[Dict[str, Any]]] = None
    lineage_trace_id: str = ""
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    governance_query_check: str = ""
    semantic_enrichment: Dict[str, Any] = {}
    quality_scores: Dict[str, Any] = {}
    error: Optional[str] = None


# ── Grafana models ──────────────────────────────────────────────────
class GrafanaTarget(BaseModel):
    target: str
    type: str = "table"


class GrafanaQuery(BaseModel):
    targets: List[GrafanaTarget]
    range: Optional[Dict[str, str]] = None


def create_api_app(ados_system=None) -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ADOS v2 — AI-Native Data OS",
        description="LLM + LangGraph + Neo4j + Grafana",
        version=settings.version,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )
    app.state.ados = ados_system

    # ── Core API ────────────────────────────────────────────────────

    @app.get("/api/v1/health")
    async def health():
        return {"status": "healthy", "version": settings.version}

    @app.post("/api/v1/query", response_model=QueryResponse)
    async def process_query(req: QueryRequest):
        system = app.state.ados
        if not system:
            raise HTTPException(503, "System not initialized")

        result = system.query(req.query, req.user_role)

        return QueryResponse(
            status=result.get("status", "unknown"),
            user_query=req.query,
            intent=result.get("intent", {}),
            discovery=result.get("discovery", {}),
            cypher=result.get("sql", ""),
            trust=result.get("trust", {}),
            analysis=result.get("analysis", {}),
            result_count=len(result.get("result_data", [])),
            result_data=result.get("result_data", [])[:50],
            lineage_trace_id=result.get("lineage_trace_id", ""),
            steps=result.get("steps", []),
            total_duration_ms=result.get("total_duration_ms", 0),
            governance_query_check=result.get("governance_query_check", ""),
            semantic_enrichment=result.get("semantic_enrichment", {}),
            quality_scores=result.get("quality_scores", {}),
            error=result.get("error"),
        )

    @app.get("/api/v1/catalog")
    async def get_catalog():
        system = app.state.ados
        if not system:
            raise HTTPException(503, "Not initialized")
        return system.catalog.summary()

    @app.get("/api/v1/kg")
    async def get_kg():
        system = app.state.ados
        if not system:
            raise HTTPException(503, "Not initialized")
        return {
            "summary": system.knowledge_graph.summary(),
            "ascii": system.knowledge_graph.render_ascii(),
            "relationships": system.knowledge_graph.get_relationships(),
        }

    @app.get("/api/v1/lineage")
    async def get_lineage():
        system = app.state.ados
        if not system:
            raise HTTPException(503, "Not initialized")
        traces = system.lineage.get_all_traces()
        return {
            "total": len(traces),
            "traces": [t.model_dump(mode="json") for t in traces],
        }

    # ── Quality, Governance, Semantic endpoints ─────────────────────

    @app.get("/api/v1/quality")
    async def get_quality():
        """Get quality assessment reports for all data products."""
        system = app.state.ados
        if not system:
            raise HTTPException(503, "Not initialized")
        summary = system.quality_engine.get_summary()
        reports = {}
        for name, report in system.quality_engine.get_all_reports().items():
            reports[name] = {
                "composite_score": report.composite_score,
                "grade": report.grade,
                "total_issues": report.total_issues,
                "critical_issues": report.critical_issues,
                "dimensions": [
                    {
                        "dimension": d.dimension,
                        "score": d.score,
                        "weight": d.weight,
                        "issues_count": len(d.issues),
                    }
                    for d in report.dimensions
                ],
            }
        return {"summary": summary, "reports": reports}

    @app.get("/api/v1/quality/{product_name}")
    async def get_quality_report(product_name: str):
        """Get quality report for a specific data product."""
        system = app.state.ados
        if not system:
            raise HTTPException(503, "Not initialized")
        report = system.quality_engine.get_report(product_name)
        if not report:
            raise HTTPException(404, f"No quality report for '{product_name}'")
        return report.model_dump(mode="json")

    @app.get("/api/v1/governance")
    async def get_governance():
        """Get governance compliance summary."""
        system = app.state.ados
        if not system:
            raise HTTPException(503, "Not initialized")
        return system.governance.get_compliance_summary()

    @app.get("/api/v1/semantic")
    async def get_semantic():
        """Get semantic layer summary (glossary + annotations)."""
        system = app.state.ados
        if not system:
            raise HTTPException(503, "Not initialized")
        summary = system.semantic_layer.summary()
        # Include glossary terms
        glossary = []
        for term in system.semantic_layer._glossary.values():
            glossary.append({
                "term": term.term,
                "definition": term.definition,
                "synonyms": term.synonyms,
                "related_columns": term.related_columns,
                "domain": term.domain,
                "category": term.category,
            })
        return {"summary": summary, "glossary": glossary}

    @app.get("/api/v1/recommendations/{product_name}")
    async def get_recommendations(product_name: str):
        """Get AI-driven recommendations for a data product."""
        system = app.state.ados
        if not system:
            raise HTTPException(503, "Not initialized")
        recs = system.catalog.get_recommendations(product_name)
        return {"product": product_name, "recommendations": recs}

    @app.get("/api/v1/usage")
    async def get_usage():
        """Get data usage analytics (active metadata)."""
        system = app.state.ados
        if not system:
            raise HTTPException(503, "Not initialized")
        return system.catalog.get_usage_stats()

    # ── Grafana SimpleJSON Datasource ───────────────────────────────

    @app.get("/grafana/")
    @app.get("/grafana")
    async def grafana_test():
        return {"status": "ok"}

    @app.post("/grafana/search")
    async def grafana_search():
        """Return available metric names for Grafana."""
        system = app.state.ados
        if not system:
            return []
        metrics = []
        for name in system.catalog.list_products():
            entry = system.catalog.get_product(name)
            if entry:
                metrics.append(name)
                for col in entry.columns:
                    metrics.append(f"{name}.{col.name}")
        metrics.extend(["pipeline_steps", "trust_scores", "churn_analysis"])
        return metrics

    @app.post("/grafana/query")
    async def grafana_query(query: GrafanaQuery):
        """Serve data to Grafana panels."""
        system = app.state.ados
        if not system:
            return []

        results = []
        for target in query.targets:
            t = target.target

            if t == "churn_analysis":
                # Serve churn distribution data
                data = _get_churn_data(system)
                results.append({"target": t, "type": "table", **data})

            elif t == "trust_scores":
                # Serve trust score history
                data = _get_trust_data(system)
                results.append({"target": t, "type": "table", **data})

            elif t == "pipeline_steps":
                data = _get_pipeline_data(system)
                results.append({"target": t, "type": "table", **data})

            elif "." in t:
                # Column-level data: product.column
                parts = t.split(".", 1)
                data = _get_column_data(system, parts[0], parts[1])
                results.append({"target": t, "type": "table", **data})

            else:
                # Product-level summary
                entry = system.catalog.get_product(t)
                if entry:
                    results.append({
                        "target": t, "type": "table",
                        "columns": [{"text": "Column"}, {"text": "Type"}, {"text": "Unique"}],
                        "rows": [[c.name, c.data_type, c.nunique] for c in entry.columns],
                    })

        return results

    @app.post("/grafana/annotations")
    async def grafana_annotations():
        return []

    return app


def _get_churn_data(system):
    """Build churn analysis data for Grafana."""
    import pandas as pd
    products = system.data_products
    for name, p in products.items():
        if p.dataframe is not None and "Churn" in p.dataframe.columns:
            df = p.dataframe
            churn_counts = df["Churn"].value_counts().to_dict()
            by_contract = df.groupby(["Contract", "Churn"]).size().reset_index(name="count")
            rows = [[r["Contract"], r["Churn"], int(r["count"])] for _, r in by_contract.iterrows()]
            return {
                "columns": [{"text": "Contract"}, {"text": "Churn"}, {"text": "Count"}],
                "rows": rows,
            }
    return {"columns": [], "rows": []}


def _get_trust_data(system):
    traces = system.lineage.get_all_traces()
    rows = [[t.trace_id, str(t.created_at), len(t.nodes)] for t in traces]
    return {
        "columns": [{"text": "Trace ID"}, {"text": "Timestamp"}, {"text": "Steps"}],
        "rows": rows,
    }


def _get_pipeline_data(system):
    return {
        "columns": [{"text": "Metric"}, {"text": "Value"}],
        "rows": [
            ["Products Loaded", len(system.data_products)],
            ["KG Nodes", system.knowledge_graph.summary().get("nodes", 0)],
            ["KG Relationships", system.knowledge_graph.summary().get("relationships", 0)],
            ["Lineage Traces", len(system.lineage.get_all_traces())],
        ],
    }


def _get_column_data(system, product_name: str, col_name: str):
    import pandas as pd
    product = system.data_products.get(product_name)
    if product and product.dataframe is not None and col_name in product.dataframe.columns:
        series = product.dataframe[col_name]
        vc = series.value_counts().head(20)
        rows = [[str(k), int(v)] for k, v in vc.items()]
        return {
            "columns": [{"text": col_name}, {"text": "Count"}],
            "rows": rows,
        }
    return {"columns": [], "rows": []}
