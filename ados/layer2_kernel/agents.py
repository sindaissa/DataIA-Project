"""
LLM-Powered Agents — All intelligence via LangChain + Groq
───────────────────────────────────────────────────────────────────
No if/else keyword matching. Every decision goes through the LLM.

Rate-limit resilience:
  • Detects daily-token-limit (TPD) errors and skips straight to fallbacks
  • Tries a configurable list of fallback models (LLM_FALLBACK_MODELS)
  • Caches identical LLM calls for LLM_CACHE_TTL seconds
"""
from __future__ import annotations
import hashlib
import json
import time
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ados.config import get_settings
from ados.logging_config import get_logger

logger = get_logger(__name__)


class AgentResult(BaseModel):
    agent_name: str
    status: str = "success"
    data: Dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    execution_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Simple TTL cache for LLM responses ──────────────────────────────
_llm_cache: Dict[str, tuple] = {}  # key → (timestamp, response)


def _cache_key(params: dict) -> str:
    """Deterministic hash of the invoke parameters."""
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str, ttl: int):
    """Return cached value if it exists and is fresh, else None."""
    if ttl <= 0:
        return None
    entry = _llm_cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.time() - ts > ttl:
        _llm_cache.pop(key, None)
        return None
    return value


def _cache_put(key: str, value, ttl: int):
    if ttl > 0:
        _llm_cache[key] = (time.time(), value)


# ── Helpers to classify errors ──────────────────────────────────────

def _is_rate_limit(err: Exception) -> bool:
    s = str(err).lower()
    return "429" in s or "rate_limit" in s or "rate limit" in s


def _is_daily_limit(err: Exception) -> bool:
    """True when the error is a *daily* token-per-day (TPD) limit."""
    s = str(err).lower()
    return _is_rate_limit(err) and ("per day" in s or "tpd" in s)


def _is_model_unavailable(err: Exception) -> bool:
    s = str(err).lower()
    return any(kw in s for kw in ("decommission", "not found", "does not exist",
                                   "model_not_active", "invalid model"))


def _get_fallback_models(settings) -> List[str]:
    """Return the ordered list of fallback model names from settings."""
    raw = getattr(settings.llm, "fallback_models", "llama-3.1-8b-instant")
    return [m.strip() for m in raw.split(",") if m.strip()]


# ── LLM constructor ────────────────────────────────────────────────

def get_llm(settings, model_override: str | None = None):
    """Create the LangChain LLM based on settings — Groq (ultra-fast)."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model_override or settings.llm.model_name,
        api_key=settings.llm.api_key,
        temperature=settings.llm.temperature,
    )


# ── Resilient invoke with retry + fallback + cache ─────────────────

def _invoke_with_retry(chain, params: dict, settings, max_retries: int = 3):
    """
    Invoke an LLM chain with:
      1. TTL cache lookup (skip LLM call entirely for repeated queries)
      2. Retry on transient 429 (per-minute) errors with exponential back-off
      3. Skip retries on daily-token-limit (TPD) errors → go to fallbacks
      4. Walk through fallback models one by one
    Returns the raw string response.  Raises on persistent failure.
    """
    import time as _time
    from langchain_groq import ChatGroq
    from langchain_core.output_parsers import StrOutputParser

    cache_ttl = getattr(settings.llm, "cache_ttl_seconds", 300)

    # ── 0. Cache check ──────────────────────────────────────────────
    ckey = _cache_key(params)
    cached = _cache_get(ckey, cache_ttl)
    if cached is not None:
        logger.info("LLM cache hit — skipping API call")
        return cached

    last_error: Exception | None = None

    # ── 1. Try the primary model ────────────────────────────────────
    for attempt in range(max_retries):
        try:
            result = chain.invoke(params)
            _cache_put(ckey, result, cache_ttl)
            return result
        except Exception as e:
            last_error = e
            if _is_daily_limit(e):
                # Daily limit — retrying the same model won't help
                logger.warning(
                    f"Daily token limit (TPD) hit on primary model, "
                    f"switching to fallback models immediately."
                )
                break
            elif _is_rate_limit(e):
                wait = 2 ** attempt  # 1 s, 2 s, 4 s
                logger.warning(
                    f"Rate-limited (attempt {attempt+1}/{max_retries}), "
                    f"retrying in {wait}s…"
                )
                _time.sleep(wait)
            else:
                raise  # non-rate-limit error, propagate immediately

    # ── 2. Walk fallback models ─────────────────────────────────────
    prompt = chain.first  # the ChatPromptTemplate
    for fb_model in _get_fallback_models(settings):
        try:
            logger.info(f"Trying fallback model: {fb_model}")
            fb_llm = ChatGroq(
                model=fb_model,
                api_key=settings.llm.api_key,
                temperature=settings.llm.temperature,
            )
            fb_chain = prompt | fb_llm | StrOutputParser()
            result = fb_chain.invoke(params)
            _cache_put(ckey, result, cache_ttl)
            logger.info(f"Fallback model {fb_model} succeeded ✔")
            return result
        except Exception as e:
            last_error = e
            if _is_rate_limit(e) or _is_model_unavailable(e):
                logger.warning(
                    f"Fallback {fb_model} unavailable ({str(e)[:80]}), "
                    f"trying next…"
                )
                _time.sleep(1)  # small courtesy pause between fallbacks
                continue
            else:
                raise

    # ── 3. All models exhausted ─────────────────────────────────────
    raise last_error  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# INTENT COMPILER AGENT — LLM parses natural language into structured intent
# ═══════════════════════════════════════════════════════════════════════

INTENT_PROMPT = ChatPromptTemplate.from_template("""
You are an AI data analyst. Parse the following user query into a structured JSON intent.

Available data schema:
{schema_context}

User query: "{query}"

Return ONLY valid JSON with this exact structure:
{{
    "action": "analyze|list|aggregate|compare|predict",
    "description": "what the user wants to achieve",
    "relevant_columns": ["col1", "col2"],
    "filters": {{"column_name": "condition"}},
    "metrics": ["columns to measure/aggregate"],
    "groupby": ["columns to group by"],
    "complexity": "simple|filtered|aggregation|cross-analysis",
    "confidence": 0.0 to 1.0
}}

JSON:
""")


def run_intent_agent(llm, query: str, schema_context: str) -> AgentResult:
    """LLM-powered intent compilation — no regex, no keywords."""
    start = time.time()
    chain = INTENT_PROMPT | llm | StrOutputParser()
    settings = get_settings()

    try:
        raw = _invoke_with_retry(chain, {"query": query, "schema_context": schema_context}, settings)
        # Extract JSON from response
        intent = _extract_json(raw)
        elapsed = (time.time() - start) * 1000

        logger.info(f"IntentAgent: parsed intent in {elapsed:.0f}ms")
        return AgentResult(
            agent_name="intent_agent",
            data={"intent": intent, "raw_response": raw},
            message=f"Intent: {intent.get('action', 'unknown')} — {intent.get('description', '')}",
            execution_time_ms=elapsed,
        )
    except Exception as e:
        logger.error(f"IntentAgent error: {e}")
        return AgentResult(
            agent_name="intent_agent", status="error",
            data={"error": str(e)},
            message=str(e),
            execution_time_ms=(time.time() - start) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════
# DISCOVERY AGENT — LLM decides which data products and columns are relevant
# ═══════════════════════════════════════════════════════════════════════

DISCOVERY_PROMPT = ChatPromptTemplate.from_template("""
You are a data discovery agent. Given the user's intent and available data products,
determine which data sources and columns are needed.

Available data products and their schemas:
{schema_context}

Knowledge graph relationships:
{kg_context}

User intent:
{intent_json}

Return ONLY valid JSON:
{{
    "relevant_products": ["product_name1"],
    "relevant_columns": {{"product_name": ["col1", "col2"]}},
    "join_strategy": "description of how to join data if cross-product",
    "reasoning": "why these products/columns are relevant"
}}

JSON:
""")


def run_discovery_agent(llm, intent: dict, schema_context: str, kg_context: str) -> AgentResult:
    """LLM discovers relevant data sources — no hardcoded keyword maps."""
    start = time.time()
    chain = DISCOVERY_PROMPT | llm | StrOutputParser()
    settings = get_settings()

    try:
        raw = _invoke_with_retry(chain, {
            "schema_context": schema_context,
            "kg_context": kg_context,
            "intent_json": json.dumps(intent, ensure_ascii=False),
        }, settings)
        discovery = _extract_json(raw)
        elapsed = (time.time() - start) * 1000

        logger.info(
            f"DiscoveryAgent: found {len(discovery.get('relevant_products', []))} products "
            f"in {elapsed:.0f}ms"
        )
        return AgentResult(
            agent_name="discovery_agent",
            data=discovery,
            message=discovery.get("reasoning", ""),
            execution_time_ms=elapsed,
        )
    except Exception as e:
        logger.error(f"DiscoveryAgent error: {e}")
        return AgentResult(
            agent_name="discovery_agent", status="error",
            data={"error": str(e)}, message=str(e),
            execution_time_ms=(time.time() - start) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════
# QUERY BUILDER AGENT — LLM generates Neo4j Cypher from intent + graph schema
# ═══════════════════════════════════════════════════════════════════════

QUERY_PROMPT = ChatPromptTemplate.from_template("""
You are a Neo4j Cypher expert. Generate a Cypher query to answer the user's question.

The data is stored in a Neo4j graph database. Here is the graph schema:

{schema_context}

User's intent:
{intent_json}

Discovery results (relevant entities/relationships):
{discovery_json}

CRITICAL RULES — READ CAREFULLY:
1. ONLY use node labels and properties listed in the schema above. NEVER invent properties.
2. Customer nodes do NOT have a 'label' or 'dataset' property. Access ALL customers with: MATCH (c:Customer)
3. Use relationships to navigate the graph:
   - Contract type: MATCH (c:Customer)-[:HAS_CONTRACT]->(con:Contract) — access via con.type
   - Internet service: MATCH (c:Customer)-[:USES_INTERNET]->(i:InternetService) — access via i.type
   - Payment method: MATCH (c:Customer)-[:PAYS_BY]->(p:PaymentMethod) — access via p.method
   - Churn status: MATCH (c:Customer)-[:HAS_CHURN_STATUS]->(s:ChurnStatus) — access via s.status
4. Customer scalar properties (on the Customer node directly): tenure, MonthlyCharges, TotalCharges, gender, SeniorCitizen, Partner, Dependents, etc.
5. For churn rate: count customers where s.status = 'Yes' divided by total count
6. Aggregations: count(), avg(), sum(), max(), min() — no GROUP BY keyword needed
7. LIMIT results to 200 rows max
8. Return ONLY the Cypher query, nothing else — no explanation, no markdown

CORRECT EXAMPLE QUERIES:

- Churn rate by contract type:
  MATCH (c:Customer)-[:HAS_CONTRACT]->(con:Contract)
  MATCH (c)-[:HAS_CHURN_STATUS]->(s:ChurnStatus)
  RETURN con.type AS contract_type, 
         count(c) AS total_customers,
         sum(CASE WHEN s.status = 'Yes' THEN 1 ELSE 0 END) AS churned,
         round(toFloat(sum(CASE WHEN s.status = 'Yes' THEN 1 ELSE 0 END)) / count(c) * 100, 2) AS churn_rate_pct

- Average monthly charges by internet service:
  MATCH (c:Customer)-[:USES_INTERNET]->(i:InternetService)
  RETURN i.type AS internet_service, round(avg(c.MonthlyCharges), 2) AS avg_charges

- Senior citizens who churned:
  MATCH (c:Customer)-[:HAS_CHURN_STATUS]->(s:ChurnStatus)
  WHERE c.SeniorCitizen = 1 AND s.status = 'Yes'
  RETURN count(c) AS senior_churned

Cypher:
""")


def run_query_agent(llm, intent: dict, discovery: dict, schema_context: str) -> AgentResult:
    """LLM generates Cypher — no template-based query building."""
    start = time.time()
    chain = QUERY_PROMPT | llm | StrOutputParser()
    settings = get_settings()

    try:
        raw = _invoke_with_retry(chain, {
            "schema_context": schema_context,
            "intent_json": json.dumps(intent, ensure_ascii=False),
            "discovery_json": json.dumps(discovery, ensure_ascii=False),
        }, settings)
        # Clean up the Cypher (same extraction logic works for Cypher)
        cypher = _extract_sql(raw)  # Works for Cypher too
        elapsed = (time.time() - start) * 1000

        logger.info(f"QueryAgent: Cypher generated in {elapsed:.0f}ms")
        return AgentResult(
            agent_name="query_agent",
            data={"sql": cypher, "raw_response": raw},  # Keep "sql" key for compatibility
            message=f"Cypher query generated ({len(cypher)} chars)",
            execution_time_ms=elapsed,
        )
    except Exception as e:
        logger.error(f"QueryAgent error: {e}")
        return AgentResult(
            agent_name="query_agent", status="error",
            data={"error": str(e)}, message=str(e),
            execution_time_ms=(time.time() - start) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════
# TRUST JUDGE AGENT — LLM validates the query and results
# ═══════════════════════════════════════════════════════════════════════

TRUST_PROMPT = ChatPromptTemplate.from_template("""
You are an LLM-as-a-Judge for data queries. Your role is to evaluate the generated
Cypher query and its execution results for correctness, safety, and relevance.

Original user question: "{user_query}"

Generated Cypher query:
{sql}

Query result summary:
- Rows returned: {row_count}
- Columns: {columns}
- Sample data (first 3 rows): {sample_data}

Evaluate on these 5 criteria (score each 0-20, total = trust_score 0-100):
1. **Correctness** — Does the Cypher correctly answer the user's question?
2. **Safety** — Any dangerous mutation operations (DELETE, SET, REMOVE, CREATE)?
3. **Coherence** — Are the results logical and meaningful (no absurd values)?
4. **Data Quality** — Missing values, nulls, inconsistencies in results?
5. **PII Exposure** — Does the result expose sensitive data (names, IDs, emails)?

Return ONLY valid JSON:
{{
    "trust_score": 0 to 100,
    "approved": true/false,
    "criteria": {{
        "correctness": {{"score": 0-20, "comment": "..."}},
        "safety": {{"score": 0-20, "comment": "..."}},
        "coherence": {{"score": 0-20, "comment": "..."}},
        "data_quality": {{"score": 0-20, "comment": "..."}},
        "pii_exposure": {{"score": 0-20, "comment": "..."}}
    }},
    "assessment": "overall assessment in 2-3 sentences",
    "issues": ["issue1", "issue2"],
    "warnings": ["warning1"],
    "recommendations": ["recommendation1"]
}}

JSON:
""")


def run_trust_agent(llm, user_query: str, sql: str, result_data: list) -> AgentResult:
    """LLM-powered trust validation — no rule-based checking."""
    start = time.time()
    chain = TRUST_PROMPT | llm | StrOutputParser()
    settings = get_settings()

    columns = list(result_data[0].keys()) if result_data else []
    sample = result_data[:3] if result_data else []

    try:
        raw = _invoke_with_retry(chain, {
            "user_query": user_query,
            "sql": sql,
            "row_count": len(result_data),
            "columns": json.dumps(columns),
            "sample_data": json.dumps(sample, ensure_ascii=False, default=str),
        }, settings)
        trust = _extract_json(raw)
        elapsed = (time.time() - start) * 1000

        score = trust.get("trust_score", 75)
        logger.info(f"TrustJudge: score={score}/100 in {elapsed:.0f}ms")
        return AgentResult(
            agent_name="trust_judge",
            data=trust,
            message=f"Trust Score: {score}/100 — {'✔ APPROVED' if trust.get('approved', True) else '✘ REJECTED'}",
            execution_time_ms=elapsed,
        )
    except Exception as e:
        logger.error(f"TrustJudge error: {e}")
        return AgentResult(
            agent_name="trust_judge", status="error",
            data={"trust_score": 50, "approved": True, "error": str(e)},
            message=str(e),
            execution_time_ms=(time.time() - start) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════
# ANALYST AGENT — LLM provides intelligent analysis of results
# ═══════════════════════════════════════════════════════════════════════

ANALYST_PROMPT = ChatPromptTemplate.from_template("""
You are a senior data analyst. Analyze the query results and provide intelligent insights.

User question: "{user_query}"
Cypher executed: {sql}

Results summary:
- Total rows: {row_count}
- Columns: {columns}
- Data sample: {sample_data}

{numeric_summary}

Provide a concise but insightful analysis in JSON:
{{
    "summary": "2-3 sentence summary of key findings",
    "key_insights": ["insight1", "insight2", "insight3"],
    "trends": ["trend1"],
    "anomalies": ["anomaly1 if any"],
    "recommendations": ["actionable recommendation"],
    "visualization_suggestions": ["chart_type: description"]
}}

JSON:
""")


def run_analyst_agent(llm, user_query: str, sql: str, result_data: list) -> AgentResult:
    """LLM-powered data analysis — real intelligence, not formatted tables."""
    start = time.time()
    chain = ANALYST_PROMPT | llm | StrOutputParser()
    settings = get_settings()

    import pandas as pd
    df = pd.DataFrame(result_data) if result_data else pd.DataFrame()
    columns = list(df.columns) if not df.empty else []

    # Build numeric summary
    numeric_summary = ""
    if not df.empty:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            numeric_summary = "Numeric column statistics:\n"
            for col in numeric_cols:
                numeric_summary += f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}\n"

    try:
        raw = _invoke_with_retry(chain, {
            "user_query": user_query,
            "sql": sql,
            "row_count": len(result_data),
            "columns": json.dumps(columns),
            "sample_data": json.dumps(result_data[:5], ensure_ascii=False, default=str),
            "numeric_summary": numeric_summary,
        }, settings)
        analysis = _extract_json(raw)
        elapsed = (time.time() - start) * 1000

        logger.info(f"AnalystAgent: analysis complete in {elapsed:.0f}ms")
        return AgentResult(
            agent_name="analyst_agent",
            data=analysis,
            message=analysis.get("summary", "Analysis complete"),
            execution_time_ms=elapsed,
        )
    except Exception as e:
        logger.error(f"AnalystAgent error: {e}")
        return AgentResult(
            agent_name="analyst_agent", status="error",
            data={"error": str(e)}, message=str(e),
            execution_time_ms=(time.time() - start) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    import re
    # Try to find JSON in code blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1)

    # Try to find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try the whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"raw": text, "parse_error": True}


def _extract_sql(text: str) -> str:
    """Extract SQL or Cypher query from LLM response."""
    import re
    # Try code blocks first (```sql, ```cypher, or unmarked)
    match = re.search(r"```(?:sql|cypher)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find MATCH statement (Cypher)
    match = re.search(r"(MATCH\s+.+)", text, re.DOTALL | re.IGNORECASE)
    if match:
        query = match.group(1).strip()
        # Remove trailing non-query text
        for stop in ["\n\n", "Note:", "This query", "Explanation", "This Cypher"]:
            idx = query.find(stop)
            if idx > 0:
                query = query[:idx].strip()
        return query

    # Try to find SELECT statement (legacy SQL fallback)
    match = re.search(r"(SELECT\s+.+)", text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        for stop in ["\n\n", "Note:", "This query", "Explanation"]:
            idx = sql.find(stop)
            if idx > 0:
                sql = sql[:idx].strip()
        return sql

    return text.strip()
