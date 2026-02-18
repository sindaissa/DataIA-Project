"""
Living Knowledge Graph — Neo4j Backend (Graph-Native Architecture)
───────────────────────────────────────────────────────────────────
REPLACES CSV+DuckDB with pure Neo4j graph storage.

Architecture change:
  OLD: CSV → DuckDB → SQL query
  NEW: CSV → Neo4j Graph → Cypher query

The entire dataset is loaded as nodes + relationships:
  - Customer nodes (with all properties)
  - Dimension nodes (Contract, InternetService, PaymentMethod, etc.)
  - Relationships: HAS_CONTRACT, USES_INTERNET, PAYS_BY, CHURNED, etc.

The LLM generates Cypher, not SQL.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd
from ados.logging_config import get_logger

logger = get_logger(__name__)


class Neo4jKnowledgeGraph:
    """
    Graph-native data storage — the CSV data lives in Neo4j as a graph.
    
    Model for Telco Churn:
      (:Customer {customerID, gender, SeniorCitizen, tenure, MonthlyCharges, TotalCharges, ...})
      (:Contract {type})
      (:InternetService {type})
      (:PaymentMethod {method})
      (:ChurnStatus {status: 'Yes'|'No'})
      
      (:Customer)-[:HAS_CONTRACT]->(:Contract)
      (:Customer)-[:USES_INTERNET]->(:InternetService)
      (:Customer)-[:PAYS_BY]->(:PaymentMethod)
      (:Customer)-[:HAS_CHURN_STATUS]->(:ChurnStatus)
    """

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._uri = uri
        self._node_count = 0
        self._relationship_count = 0
        logger.info(f"Neo4j KG: connecting to {uri}")

    def close(self):
        self._driver.close()

    def clear(self):
        """Wipe the entire graph for re-initialization."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j KG: graph cleared")

    # ── Build from CSV data ─────────────────────────────────────────

    def load_csv_as_graph(self, product_name: str, df: pd.DataFrame) -> None:
        """
        Load the entire CSV dataframe into Neo4j as a property graph.
        
        For Telco Churn dataset:
        - Each row becomes a Customer node
        - Categorical values become dimension nodes
        - Relationships connect customers to dimensions
        """
        self.clear()
        
        logger.info(f"Neo4j KG: loading '{product_name}' as graph ({len(df)} rows)...")
        
        with self._driver.session() as session:
            # Create indexes for performance
            session.run("CREATE INDEX customer_id IF NOT EXISTS FOR (c:Customer) ON (c.customerID)")
            session.run("CREATE INDEX contract_type IF NOT EXISTS FOR (c:Contract) ON (c.type)")
            session.run("CREATE INDEX internet_type IF NOT EXISTS FOR (i:InternetService) ON (i.type)")
            session.run("CREATE INDEX payment_method IF NOT EXISTS FOR (p:PaymentMethod) ON (p.method)")
            session.run("CREATE INDEX churn_status IF NOT EXISTS FOR (s:ChurnStatus) ON (s.status)")
            
            # Create dimension nodes (Contract, InternetService, PaymentMethod, Churn)
            contracts = df['Contract'].dropna().unique() if 'Contract' in df.columns else []
            for contract in contracts:
                session.run("MERGE (c:Contract {type: $type})", type=str(contract))
            
            internet_services = df['InternetService'].dropna().unique() if 'InternetService' in df.columns else []
            for inet in internet_services:
                session.run("MERGE (i:InternetService {type: $type})", type=str(inet))
            
            payment_methods = df['PaymentMethod'].dropna().unique() if 'PaymentMethod' in df.columns else []
            for pm in payment_methods:
                session.run("MERGE (p:PaymentMethod {method: $method})", method=str(pm))
            
            churn_statuses = df['Churn'].dropna().unique() if 'Churn' in df.columns else []
            for status in churn_statuses:
                session.run("MERGE (s:ChurnStatus {status: $status})", status=str(status))
            
            logger.info(f"Neo4j KG: created dimension nodes (contracts={len(contracts)}, internet={len(internet_services)}, payments={len(payment_methods)})")
            
            # Load customers in batches (performance)
            batch_size = 500
            total_customers = 0
            
            for start_idx in range(0, len(df), batch_size):
                batch = df.iloc[start_idx:start_idx + batch_size]
                
                for _, row in batch.iterrows():
                    # Convert row to dict, handle NaN
                    customer_props = {}
                    for col, val in row.items():
                        if pd.notna(val):
                            # Store as appropriate type
                            if isinstance(val, (int, float, bool)):
                                customer_props[col] = val
                            else:
                                customer_props[col] = str(val)
                    
                    customer_id = customer_props.get('customerID', f'customer_{start_idx}')
                    
                    # Create Customer node
                    session.run(
                        """
                        CREATE (c:Customer $props)
                        """,
                        props=customer_props
                    )
                    
                    # Create relationships to dimension nodes
                    contract = customer_props.get('Contract')
                    if contract:
                        session.run(
                            """
                            MATCH (c:Customer {customerID: $cid})
                            MATCH (con:Contract {type: $type})
                            CREATE (c)-[:HAS_CONTRACT]->(con)
                            """,
                            cid=customer_id, type=contract
                        )
                    
                    internet = customer_props.get('InternetService')
                    if internet:
                        session.run(
                            """
                            MATCH (c:Customer {customerID: $cid})
                            MATCH (i:InternetService {type: $type})
                            CREATE (c)-[:USES_INTERNET]->(i)
                            """,
                            cid=customer_id, type=internet
                        )
                    
                    payment = customer_props.get('PaymentMethod')
                    if payment:
                        session.run(
                            """
                            MATCH (c:Customer {customerID: $cid})
                            MATCH (p:PaymentMethod {method: $method})
                            CREATE (c)-[:PAYS_BY]->(p)
                            """,
                            cid=customer_id, method=payment
                        )
                    
                    churn = customer_props.get('Churn')
                    if churn:
                        session.run(
                            """
                            MATCH (c:Customer {customerID: $cid})
                            MATCH (s:ChurnStatus {status: $status})
                            CREATE (c)-[:HAS_CHURN_STATUS]->(s)
                            """,
                            cid=customer_id, status=churn
                        )
                    
                    total_customers += 1
                
                logger.info(f"Neo4j KG: loaded batch {start_idx // batch_size + 1} ({total_customers} customers so far...)")
        
        stats = self.summary()
        self._node_count = stats['nodes']
        self._relationship_count = stats['relationships']
        logger.info(
            f"Neo4j KG: graph built — {stats['nodes']} nodes, "
            f"{stats['relationships']} relationships"
        )

    def _graph_already_loaded(self, expected_rows: int) -> bool:
        """Check if the graph has the CORRECT schema and enough data."""
        try:
            with self._driver.session() as session:
                # Check that expected labels exist
                required_labels = ["Customer", "Contract", "ChurnStatus"]
                for label in required_labels:
                    result = session.run(
                        f"MATCH (n:{label}) RETURN count(n) AS c"
                    ).single()
                    if not result or result["c"] == 0:
                        logger.info(f"Neo4j KG: label :{label} missing or empty — will reload")
                        return False
                # Check customer count
                cust_count = session.run(
                    "MATCH (c:Customer) RETURN count(c) AS c"
                ).single()["c"]
                if cust_count < expected_rows * 0.9:
                    logger.info(f"Neo4j KG: only {cust_count} customers vs {expected_rows} expected — will reload")
                    return False
            return True
        except Exception:
            return False

    def build_from_catalog(self, catalog, products: dict) -> None:
        """
        Load actual CSV data as graph, not just schema.
        Skips reload if the graph is already populated.
        """
        for name, product in products.items():
            if product.dataframe is not None:
                if self._graph_already_loaded(len(product.dataframe)):
                    stats = self.summary()
                    self._node_count = stats["nodes"]
                    self._relationship_count = stats["relationships"]
                    logger.info(
                        f"Neo4j KG: graph already loaded — {stats['nodes']} nodes, "
                        f"{stats['relationships']} relationships (skipping reload)"
                    )
                else:
                    self.load_csv_as_graph(name, product.dataframe)
                break  # Only load the first product for now

    # ── Query methods ───────────────────────────────────────────────

    def query_cypher(self, cypher: str) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query (LLM-generated) and return results.
        This replaces DuckDB SQL execution.
        """
        with self._driver.session() as session:
            result = session.run(cypher)
            return [dict(r) for r in result]

    def get_schema_graph(self) -> List[Dict[str, Any]]:
        """Return schema information for the graph."""
        with self._driver.session() as session:
            # Get node labels and counts
            result = session.run(
                """
                CALL db.labels() YIELD label
                CALL {
                  WITH label
                  MATCH (n)
                  WHERE label IN labels(n)
                  RETURN count(n) AS count
                }
                RETURN label, count
                """
            )
            return [dict(r) for r in result]

    def get_relationships(self) -> List[Dict[str, Any]]:
        """Return all relationship types in the graph."""
        with self._driver.session() as session:
            result = session.run(
                """
                CALL db.relationshipTypes() YIELD relationshipType
                CALL {
                  WITH relationshipType
                  MATCH ()-[r]->()
                  WHERE type(r) = relationshipType
                  RETURN count(r) AS count
                }
                RETURN relationshipType, count
                """
            )
            return [dict(r) for r in result]

    def get_context_for_llm(self) -> str:
        """
        Build a text description of the graph schema for LLM Cypher generation.
        This is CRITICAL — the LLM needs to know the graph structure.
        Dynamically inspects the actual graph to avoid stale info.
        """
        lines = ["## Neo4j Graph Schema", ""]
        lines.append("IMPORTANT: Only use the labels, properties, and relationships listed below.")
        lines.append("Customer nodes do NOT have a 'label' or 'dataset' property.")
        lines.append("")

        with self._driver.session() as session:
            # Node labels
            lines.append("### Node Labels and Their Properties:")
            schema = self.get_schema_graph()
            for row in schema:
                label = row['label']
                count = row['count']
                lines.append(f"\n  :{label} ({count} nodes)")
                # Show actual properties and sample values for each label
                sample = session.run(
                    f"MATCH (n:{label}) RETURN n LIMIT 1"
                ).single()
                if sample:
                    node_dict = dict(sample['n'])
                    for key in sorted(node_dict.keys()):
                        val = node_dict[key]
                        lines.append(f"    - {key}: {type(val).__name__} (e.g. {repr(val)})")

            # Show distinct values for small dimension nodes
            lines.append("\n### Dimension Node Distinct Values:")
            dimension_labels = ["Contract", "InternetService", "PaymentMethod", "ChurnStatus"]
            for label in dimension_labels:
                check = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
                if check and check["c"] > 0:
                    vals = session.run(
                        f"MATCH (n:{label}) RETURN properties(n) AS props"
                    ).data()
                    val_strs = [str(v["props"]) for v in vals]
                    lines.append(f"  :{label} values: {', '.join(val_strs)}")

        # Relationships
        lines.append("\n### Relationship Types:")
        rels = self.get_relationships()
        for row in rels:
            lines.append(f"  - {row['relationshipType']} ({row['count']} relationships)")

        # Relationship patterns (source -> target)
        lines.append("\n### Relationship Patterns (source)-[rel]->(target):")
        with self._driver.session() as session:
            patterns = session.run("""
                MATCH (a)-[r]->(b)
                WITH labels(a)[0] AS src, type(r) AS rel, labels(b)[0] AS tgt, count(*) AS cnt
                RETURN src, rel, tgt, cnt ORDER BY cnt DESC
            """).data()
            for p in patterns:
                lines.append(f"  (:{p['src']})-[:{p['rel']}]->(:{p['tgt']})  [{p['cnt']} rels]")

        # Example Cypher queries
        lines.append("\n### Correct Example Cypher Patterns:")
        lines.append("  - Get all customers: MATCH (c:Customer) RETURN c LIMIT 10")
        lines.append("  - Customers by contract: MATCH (c:Customer)-[:HAS_CONTRACT]->(con:Contract) RETURN con.type, count(c) AS total")
        lines.append("  - Churn rate by contract: MATCH (c:Customer)-[:HAS_CONTRACT]->(con:Contract), (c)-[:HAS_CHURN_STATUS]->(s:ChurnStatus) RETURN con.type, s.status, count(c) AS count")
        lines.append("  - Aggregate: Use count(), avg(), sum(), max(), min()")
        lines.append("  - Filter: WHERE c.tenure > 12, WHERE s.status = 'Yes'")

        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        with self._driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        return {"nodes": nodes, "relationships": rels}

    def render_ascii(self) -> str:
        lines = ["╔═══ Neo4j Graph-Native Data Store ═══"]
        
        schema = self.get_schema_graph()
        rels = self.get_relationships()
        
        lines.append("║  Node Types:")
        for row in schema:
            lines.append(f"║    {row['label']}: {row['count']} nodes")
        
        lines.append("║  Relationship Types:")
        for row in rels:
            lines.append(f"║    {row['relationshipType']}: {row['count']} relationships")
        
        lines.append("╚═══════════════════════════════════════")
        return "\n".join(lines)
