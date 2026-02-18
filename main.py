"""
ADOS v2 — Entry Point
─────────────────────
Usage:
  python main.py              → Run demo queries through LangGraph pipeline
  python main.py --api        → Launch FastAPI server (+ Grafana datasource)
  python main.py --streamlit  → Launch Streamlit control panel
"""
from __future__ import annotations
import sys
import json
from ados.system import ADOSSystem
from ados.logging_config import get_logger, set_correlation_id
import uuid

logger = get_logger(__name__)


DEMO_QUERIES = [
    "Quel est le taux de churn par type de contrat ?",
]


def run_demo():
    """Run example queries through the LangGraph pipeline."""
    system = ADOSSystem()
    system.initialize()
    system.print_status()

    print("\n" + "=" * 70)
    print("  ADOS v2 — Demo: LLM + LangGraph Pipeline")
    print("=" * 70)

    for i, query in enumerate(DEMO_QUERIES, 1):
        set_correlation_id(str(uuid.uuid4())[:8])
        print(f"\n{'─' * 70}")
        print(f"  Query {i}/{len(DEMO_QUERIES)}: {query}")
        print(f"{'─' * 70}")

        try:
            result = system.query(query)

            # Display results
            print(f"\n  📋 Intent:     {result.get('intent', {}).get('action', 'N/A')}")
            print(f"  🔍 Discovery:  {result.get('discovery', {}).get('relevant_products', [])}")
            print(f"  📝 Cypher:     {result.get('sql', 'N/A')[:100]}")

            # Show result data
            data = result.get("result_data", [])
            if data:
                print(f"  📊 Results:    {len(data)} rows")
                for row in data[:5]:
                    print(f"     {row}")
                if len(data) > 5:
                    print(f"     ... ({len(data) - 5} more rows)")
            else:
                print(f"  📊 Results:    No data returned")

            # Trust & Analysis
            trust = result.get("trust", {})
            print(f"  🛡️ Trust:      {trust.get('trust_score', 'N/A')}/100")
            print(f"  ✅ Approved:   {trust.get('approved', False)}")

            analysis = result.get("analysis", {})
            if analysis.get("summary"):
                print(f"  🧠 Analysis:   {analysis['summary'][:200]}")

            # Quality & Governance
            quality = result.get("quality_scores", {})
            if quality.get("average_score"):
                print(f"  📊 Quality:    {quality['average_score']:.1f}/100")

            gov_check = result.get("governance_query_check", "")
            if gov_check:
                print(f"  🏛️ Governance: {gov_check}")

            # Lineage
            trace_id = result.get("lineage_trace_id", "")
            if trace_id:
                trace = system.lineage.get_trace(trace_id)
                if trace:
                    print(f"\n  {system.lineage.render_ascii(trace)}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            logger.error(f"Demo query failed: {e}", exc_info=True)


def run_api():
    """Launch FastAPI server with Grafana datasource."""
    import uvicorn
    from ados.layer1_interface.api import create_api_app

    system = ADOSSystem()
    system.initialize()
    system.print_status()
    app = create_api_app(ados_system=system)
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║  ADOS v2 API — FastAPI + Grafana SimpleJSON          ║")
    print("║  API:     http://localhost:8000/docs                  ║")
    print("║  Grafana: http://localhost:3000                       ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def run_streamlit():
    """Launch Streamlit control panel."""
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.headless", "true",
    ])


def main():
    if "--api" in sys.argv:
        run_api()
    elif "--streamlit" in sys.argv:
        run_streamlit()
    else:
        run_demo()


if __name__ == "__main__":
    main()
