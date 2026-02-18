"""
ADOS v2 — Streamlit Control Panel
──────────────────────────────────
Lightweight UI: Query interface → LangGraph pipeline.
Visualizations → Grafana (embedded iframes).
"""
import streamlit as st
import time
import uuid

st.set_page_config(
    page_title="ADOS v2 — Autonomous Data OS",
    page_icon="🧠",
    layout="wide",
)


def get_system():
    """Initialize ADOS system once (stored in session_state, clearable)."""
    if "ados_system" not in st.session_state or st.session_state.ados_system is None:
        from ados.config import reset_settings  # Force reload .env
        settings = reset_settings()  # Always reload from .env when reinitializing
        if not settings.llm.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set! "
                "Create a .env file with: GROQ_API_KEY=your_key"
            )
        from ados.system import ADOSSystem
        system = ADOSSystem()
        system.initialize()
        st.session_state.ados_system = system
    return st.session_state.ados_system


def main():
    st.title("🧠 ADOS v2 — Autonomous Data Operating System")
    st.caption("LLM Agents + LangGraph + Neo4j + Grafana")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ System")
        if st.button("🔄 Réinitialiser le système"):
            st.session_state.pop("ados_system", None)
            st.rerun()
        try:
            system = get_system()
            st.success("System initialized")

            # Show data products
            st.subheader("📦 Data Products")
            for name, product in system.data_products.items():
                rows = len(product.dataframe)
                cols = len(product.schema)
                st.metric(name, f"{rows} rows × {cols} cols")

            # Show KG stats
            st.subheader("🔗 Knowledge Graph")
            kg_stats = system.knowledge_graph.summary()
            col1, col2 = st.columns(2)
            col1.metric("Nodes", kg_stats.get("nodes", 0))
            col2.metric("Relationships", kg_stats.get("relationships", 0))

            # LLM info
            from ados.config import get_settings
            settings = get_settings()
            st.subheader("🤖 LLM")
            st.info(f"{settings.llm.provider} / {settings.llm.model_name}")
            if not settings.llm.api_key:
                st.warning("⚠️ GROQ_API_KEY not set!")

        except Exception as e:
            st.error(f"Initialization failed: {e}")
            st.stop()

        st.divider()
        st.subheader("🔗 Links")
        st.markdown("- [Grafana Dashboard](http://localhost:3001)")
        st.markdown("- [API Docs](http://localhost:8001/docs)")
        st.markdown("- [Neo4j Browser](http://localhost:7475)")

    # Main content
    tab_query, tab_grafana, tab_catalog, tab_quality, tab_governance, tab_lineage = st.tabs([
        "🔍 Query", "📊 Grafana", "📋 Catalog", "✅ Quality", "🏛️ Governance", "🔗 Lineage"
    ])

    # ── TAB: Query ──────────────────────────────────────────
    with tab_query:
        st.subheader("Natural Language Query → LLM Pipeline")

        # Predefined queries
        examples = [
            "Quel est le taux de churn par type de contrat ?",
            "Quels sont les clients seniors avec fibre optique qui ont churné ?",
            "Quel est le montant moyen des charges mensuelles par méthode de paiement ?",
            "Combien de clients ont le support technique activé ?",
            "Analyse les feedbacks des clients qui ont churné",
        ]

        selected = st.selectbox("📝 Exemples de requêtes", [""] + examples)
        user_query = st.text_area(
            "Votre requête en langage naturel",
            value=selected,
            height=80,
            placeholder="Ex: Quel est le taux de churn des clients senior ?"
        )

        col_run, col_role = st.columns([1, 3])
        with col_run:
            run_button = st.button("🚀 Exécuter", type="primary", use_container_width=True)
        with col_role:
            user_role = st.selectbox("Rôle", ["analyst", "manager", "data_engineer"])

        if run_button and user_query.strip():
            with st.spinner("🧠 Pipeline LangGraph en cours..."):
                start = time.time()
                try:
                    result = system.query(user_query, user_role)
                    elapsed = time.time() - start

                    # Check if pipeline actually succeeded
                    cypher = result.get("sql", "")
                    data = result.get("result_data", [])
                    error = result.get("error")

                    if error:
                        # Show user-friendly message for common errors
                        if "429" in str(error) or "rate_limit" in str(error).lower():
                            st.error("⏳ Limite de requêtes Groq atteinte (rate limit). Attendez quelques minutes puis réessayez.")
                            st.info("💡 Le plan gratuit Groq est limité à 100K tokens/jour. Vous pouvez upgrader sur https://console.groq.com/settings/billing")
                        else:
                            st.error(f"❌ Pipeline error: {error}")
                        # Show agent steps for debugging
                        steps = result.get("steps", [])
                        if steps:
                            with st.expander("🔍 Détails des étapes (debug)"):
                                for step in steps:
                                    status_icon = "✅" if step.get("status") == "success" else "❌"
                                    st.text(f"{status_icon} {step.get('agent', '?')}: {step.get('message', '')[:200]}")
                    elif not cypher or not cypher.strip():
                        st.error("❌ Le LLM n'a pas généré de Cypher.")
                        steps = result.get("steps", [])
                        if steps:
                            with st.expander("🔍 Détails des étapes (debug)"):
                                for step in steps:
                                    status_icon = "✅" if step.get("status") == "success" else "❌"
                                    st.text(f"{status_icon} {step.get('agent', '?')}: {step.get('message', '')[:200]}")
                        st.warning("💡 Cliquez 🔄 Réinitialiser dans la sidebar pour relancer le système.")
                    else:
                        st.success(f"✅ Pipeline terminé en {elapsed:.2f}s")

                    # Results layout
                    col_left, col_right = st.columns([2, 1])

                    with col_left:
                        # Cypher
                        st.subheader("📝 Cypher généré par LLM")
                        st.code(result.get("sql", "N/A"), language="cypher")

                        # Data
                        data = result.get("result_data", [])
                        st.subheader(f"📊 Résultats ({len(data)} lignes)")
                        if data:
                            import pandas as pd
                            df = pd.DataFrame(data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("Aucun résultat")

                    with col_right:
                        # Intent
                        st.subheader("🎯 Intent")
                        intent = result.get("intent", {})
                        st.json(intent)

                        # Trust
                        st.subheader("🛡️ Trust Score")
                        trust = result.get("trust", {})
                        score = trust.get("trust_score", 0)
                        st.metric("Score", f"{score}/100")
                        if trust.get("approved"):
                            st.success("Validé ✅")
                        else:
                            st.warning("Non validé ⚠️")
                            if trust.get("issues"):
                                for issue in trust["issues"]:
                                    st.warning(f"  - {issue}")

                        # Governance
                        gov_check = result.get("governance_query_check", "")
                        if gov_check == "pass":
                            st.success("🏛️ Governance: ✅ Pass")
                        elif gov_check == "fail":
                            st.error("🏛️ Governance: ❌ Fail")

                    # Analysis
                    analysis = result.get("analysis", {})
                    if analysis:
                        st.subheader("🧠 Analyse LLM")
                        if analysis.get("summary"):
                            st.markdown(analysis["summary"])
                        insights = analysis.get("key_insights") or analysis.get("insights", [])
                        if insights:
                            st.markdown("**Insights:**")
                            for ins in insights:
                                st.markdown(f"- {ins}")
                        if analysis.get("recommendations"):
                            st.markdown("**Recommandations:**")
                            for rec in analysis["recommendations"]:
                                st.markdown(f"- {rec}")
                        if analysis.get("visualization_suggestions"):
                            st.markdown("**Visualisations suggérées:**")
                            for vis in analysis["visualization_suggestions"]:
                                st.markdown(f"- 📊 {vis}")

                    # Lineage
                    if result.get("lineage_trace_id"):
                        with st.expander("🔗 Lineage"):
                            trace = system.lineage.get_trace(result["lineage_trace_id"])
                            if trace:
                                st.text(system.lineage.render_ascii(trace))
                            else:
                                st.text(f"Trace ID: {result['lineage_trace_id']}")

                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
                    st.exception(e)

    # ── TAB: Grafana ────────────────────────────────────────
    with tab_grafana:
        st.subheader("📊 Grafana Dashboards")
        st.info("Grafana is available at http://localhost:3001 (admin/admin)")

        grafana_url = "http://localhost:3001"

        st.markdown(f"""
        ### Configuration Grafana
        1. **Ajouter le datasource** : Settings → Data Sources → Add → SimpleJSON
           - URL : `http://ados-api:8000/grafana` (Docker) ou `http://localhost:8001/grafana` (local)
        2. **Créer un dashboard** avec les métriques disponibles :
           - `churn_analysis` — Analyse du churn par contrat
           - `telco_churn_with_all_feedback.Contract` — Distribution des contrats
           - `telco_churn_with_all_feedback.Churn` — Distribution du churn
           - `telco_churn_with_all_feedback.InternetService` — Distribution Internet
           - `pipeline_steps` — Étapes du pipeline
           - `trust_scores` — Scores de confiance
        """)

        try:
            st.components.v1.iframe(grafana_url, height=600, scrolling=True)
        except Exception:
            st.warning("Grafana iframe not available. Access directly at http://localhost:3001")

    # ── TAB: Catalog ────────────────────────────────────────
    with tab_catalog:
        st.subheader("📋 Metadata Catalog (Active Metadata)")

        for pname in system.catalog.list_products():
            entry = system.catalog.get_product(pname)
            if not entry:
                continue

            # Product header with quality badge
            quality_badge = ""
            if entry.quality_score is not None:
                quality_badge = f" — Quality: {entry.quality_score:.0f}/100 (Grade {entry.quality_grade})"
            owner_badge = f" | Owner: {entry.owner}" if entry.owner != "unknown" else ""

            with st.expander(f"📦 {entry.domain_name} — {entry.row_count} rows{quality_badge}{owner_badge}"):
                import pandas as pd

                # Column details with semantic enrichments
                cols_data = []
                for c in entry.columns:
                    row_data = {
                        "Column": c.name,
                        "Business Name": c.business_name or "—",
                        "Type": c.data_type,
                        "Semantic": c.semantic_type or "—",
                        "Sensitivity": c.sensitivity,
                        "Unique": c.nunique,
                        "Nulls": c.null_count,
                        "Description": c.description[:60] if c.description else "—",
                    }
                    cols_data.append(row_data)
                st.dataframe(pd.DataFrame(cols_data), use_container_width=True)

                # Recommendations
                recs = system.catalog.get_recommendations(pname)
                if recs:
                    st.markdown("**📌 Recommendations:**")
                    for rec in recs:
                        st.markdown(f"- {rec}")

        # Usage stats
        usage = system.catalog.get_usage_stats()
        if usage.get("total_queries", 0) > 0:
            st.subheader("📈 Usage Analytics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", usage["total_queries"])
            with col2:
                if usage.get("most_used_columns"):
                    top_cols = ", ".join([c for c, _ in usage["most_used_columns"][:5]])
                    st.info(f"Top columns: {top_cols}")

        # Active alerts
        alerts = system.catalog.get_alerts()
        if alerts:
            st.subheader("🚨 Active Alerts")
            for alert in alerts:
                icon = "🔴" if alert.severity == "critical" else "🟡" if alert.severity == "warning" else "ℹ️"
                st.warning(f"{icon} [{alert.alert_type}] {alert.message}")

    # ── TAB: Quality ────────────────────────────────────────
    with tab_quality:
        st.subheader("✅ Data Quality Reports")

        quality_summary = system.quality_engine.get_summary()
        if quality_summary.get("total_assessed", 0) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Products Assessed", quality_summary["total_assessed"])
            with col2:
                st.metric("Average Score", f"{quality_summary['average_score']:.1f}/100")
            with col3:
                st.metric("Total Issues", quality_summary.get("total_issues", 0))

            st.divider()

            for pname, report in system.quality_engine.get_all_reports().items():
                grade_color = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}
                icon = grade_color.get(report.grade, "⚪")
                with st.expander(f"{icon} {pname} — Score: {report.composite_score:.1f}/100 (Grade {report.grade})"):
                    import pandas as pd

                    # Dimension scores
                    st.markdown("**Quality Dimensions:**")
                    dim_data = []
                    for dim in report.dimensions:
                        weight_pct = dim.weight * 100 if dim.weight < 1 else dim.weight
                        dim_data.append({
                            "Dimension": dim.dimension.capitalize(),
                            "Score": f"{dim.score:.1f}",
                            "Weight": f"{weight_pct:.0f}%",
                            "Issues": len(dim.issues),
                        })
                    st.dataframe(pd.DataFrame(dim_data), use_container_width=True)

                    # Issues
                    if report.critical_issues:
                        st.error("**Critical Issues:**")
                        for issue in report.critical_issues:
                            st.markdown(f"- 🔴 {issue}")

                    all_issues = []
                    for dim in report.dimensions:
                        all_issues.extend(dim.issues)
                    non_critical = [i for i in all_issues if "CRITICAL" not in i.upper()]
                    if non_critical:
                        st.warning("**Warnings:**")
                        for issue in non_critical[:10]:
                            st.markdown(f"- 🟡 {issue}")

                    # Column-level quality
                    if report.column_scores:
                        st.markdown("**Per-Column Quality:**")
                        col_data = [{
                            "Column": cs.column_name,
                            "Completeness": f"{cs.completeness:.1f}%",
                            "Uniqueness": f"{cs.uniqueness:.1f}%",
                            "Validity": f"{cs.validity:.1f}%",
                        } for cs in report.column_scores]
                        st.dataframe(pd.DataFrame(col_data), use_container_width=True)
        else:
            st.info("No quality assessments available.")

    # ── TAB: Governance ─────────────────────────────────────
    with tab_governance:
        st.subheader("🏛️ Federated Governance")

        gov_summary = system.governance.get_compliance_summary()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Global Rules", gov_summary.get("total_rules", 0))
        with col2:
            st.metric("Access Policies", gov_summary.get("access_policies", 0))
        with col3:
            pii_count = len(gov_summary.get("pii_products", []))
            st.metric("PII Products", pii_count)

        # PII alert
        pii_products = gov_summary.get("pii_products", [])
        if pii_products:
            st.warning(f"⚠️ PII detected in: {', '.join(pii_products)}")

        # Latest compliance results
        latest = gov_summary.get("latest_results")
        if latest:
            st.divider()
            st.markdown(f"**Latest Compliance Check:** `{latest['product']}`")
            overall = latest.get("overall", "unknown")
            if overall == "pass":
                st.success("Overall: ✅ PASS")
            else:
                st.error("Overall: ❌ FAIL")

            import pandas as pd
            results_data = []
            for r in latest.get("results", []):
                status_icon = "✅" if r["status"] == "pass" else "❌" if r["status"] == "fail" else "⚠️"
                results_data.append({
                    "Rule": r["rule_name"],
                    "Status": f"{status_icon} {r['status']}",
                    "Message": r["message"],
                    "Severity": r["severity"],
                })
            st.dataframe(pd.DataFrame(results_data), use_container_width=True)

        # Semantic layer summary
        st.divider()
        st.subheader("📚 Semantic Layer")
        sem = system.semantic_layer.summary()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Glossary Terms", sem.get("glossary_terms", 0))
        with col2:
            st.metric("Column Annotations", sem.get("total_annotations", 0))

    # ── TAB: Lineage ────────────────────────────────────────
    with tab_lineage:
        st.subheader("🔗 Pipeline Lineage")
        traces = system.lineage.get_all_traces()
        if traces:
            for trace in traces:
                with st.expander(f"Trace: {trace.trace_id}"):
                    st.text(system.lineage.render_ascii(trace))
        else:
            st.info("No lineage traces yet. Run a query first.")


if __name__ == "__main__":
    main()
