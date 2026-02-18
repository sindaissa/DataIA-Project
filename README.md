# 🧠 ADOS v2 — Autonomous Data Operating System

> **Plateforme de données intelligente qui charge un CSV dans un graphe Neo4j, transforme des questions en langage naturel en requêtes Cypher via des agents LLM, et retourne des résultats certifiés — avec gouvernance Data Mesh et assurance qualité Data Fabric.**

---

## 📑 Table des matières

1. [Présentation du projet](#-présentation-du-projet)
2. [Architecture globale](#-architecture-globale)
3. [Stack technique](#-stack-technique)
4. [Structure du projet](#-structure-du-projet)
5. [Description détaillée des couches](#-description-détaillée-des-couches)
   - [Layer 4 — Data Mesh](#layer-4--data-mesh)
   - [Layer 3 — Data Fabric](#layer-3--data-fabric)
   - [Layer 2 — Kernel (LLM + Neo4j)](#layer-2--kernel-llm--neo4j)
   - [Layer 1 — Interface](#layer-1--interface)
6. [Pipeline LangGraph (flux de traitement)](#-pipeline-langgraph-flux-de-traitement)
7. [Modèle de graphe Neo4j](#-modèle-de-graphe-neo4j)
8. [Mécanisme de résilience LLM](#-mécanisme-de-résilience-llm)
9. [Initialisation du système (8 étapes)](#-initialisation-du-système-8-étapes)
10. [Installation et démarrage rapide](#-installation-et-démarrage-rapide)
11. [Utilisation](#-utilisation)
12. [API REST (FastAPI)](#-api-rest-fastapi)
13. [Configuration](#-configuration)
14. [Dataset](#-dataset)
15. [Logging et observabilité](#-logging-et-observabilité)
16. [Diagrammes de séquence](#-diagrammes-de-séquence)
17. [Dépannage](#-dépannage)
18. [Licence](#-licence)

---

## 🎯 Présentation du projet

ADOS v2 (**A**utonomous **D**ata **O**perating **S**ystem) est une plateforme de données intelligente qui combine :

- **5 agents LLM** (aucune logique codée en dur) pour analyser une question utilisateur et générer des requêtes Cypher
- **LangGraph** pour l'orchestration multi-agent sous forme de graphe d'état
- **Neo4j** comme base de données graphe native (les données CSV vivent dans le graphe)
- **Data Mesh** — produits de données, contrats, SLA, gouvernance fédérée, détection PII
- **Data Fabric** — qualité (5 dimensions), couche sémantique, métadonnées actives, lignage
- **Grafana** — dashboards de monitoring et visualisation
- **Streamlit** — interface utilisateur interactive (6 onglets)
- **FastAPI** — API REST avec datasource Grafana SimpleJSON

### Ce que fait le système :

```
Question en langage naturel (français/anglais)
        ↓
   [Agent Intent]     → Parse l'intention en JSON structuré
        ↓
   [Agent Discovery]  → Identifie les produits de données pertinents
        ↓
   [Agent Cypher]     → Génère une requête Cypher Neo4j
        ↓
   [Neo4j Execute]    → Exécute la requête sur le graphe propriété
        ↓
   [Agent Trust]      → Valide la requête et les résultats (score 0–100)
        ↓
   [Agent Analyst]    → Produit un résumé, des insights et recommandations
        ↓
   Résultat certifié avec lignage complet
```

---

## 🏗️ Architecture globale

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        ADOS v2 — Architecture                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │               Layer 1 — Interface                             │        │
│  │  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │        │
│  │  │  Streamlit   │   │  FastAPI +   │   │  Grafana         │  │        │
│  │  │  Control     │   │  REST API    │   │  SimpleJSON      │  │        │
│  │  │  Panel       │   │  /query      │   │  Datasource      │  │        │
│  │  │  :8502       │   │  :8001       │   │  :3001           │  │        │
│  │  └──────┬───────┘   └──────┬───────┘   └──────────────────┘  │        │
│  └─────────┼──────────────────┼─────────────────────────────────┘        │
│            │                  │                                           │
│            ▼                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │               Layer 2 — Kernel (LangGraph + Neo4j)            │        │
│  │                                                                │        │
│  │   ┌─────────┐   ┌───────────┐   ┌─────────┐   ┌──────────┐  │        │
│  │   │ Intent  │──▶│ Discovery │──▶│ Cypher  │──▶│ Execute  │  │        │
│  │   │ Agent   │   │  Agent    │   │ Builder │   │ (Neo4j)  │  │        │
│  │   │  (LLM)  │   │  (LLM)   │   │  (LLM)  │   │          │  │        │
│  │   └─────────┘   └───────────┘   └─────────┘   └────┬─────┘  │        │
│  │                                                      │        │        │
│  │                  ┌───────────┐   ┌──────────┐        │        │        │
│  │                  │ Analyst   │◀──│  Trust   │◀───────┘        │        │
│  │                  │  Agent    │   │  Judge   │                 │        │
│  │                  │  (LLM)   │   │  (LLM)   │                 │        │
│  │                  └───────────┘   └──────────┘                 │        │
│  └──────────────────────────────────────────────────────────────┘        │
│            │                  │                                           │
│            ▼                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │               Layer 3 — Data Fabric                           │        │
│  │  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │        │
│  │  │  Active     │  │ Quality  │  │ Semantic │  │ Lineage  │  │        │
│  │  │  Metadata   │  │ Engine   │  │  Layer   │  │ Service  │  │        │
│  │  │  Catalog    │  │ 5 dims   │  │ Glossary │  │ DAG      │  │        │
│  │  └─────────────┘  └──────────┘  └──────────┘  └──────────┘  │        │
│  └──────────────────────────────────────────────────────────────┘        │
│            │                                                             │
│            ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │               Layer 4 — Data Mesh                             │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │        │
│  │  │ Data Product │  │  Data        │  │   Federated        │  │        │
│  │  │ + Contract   │  │  Product     │  │   Governance       │  │        │
│  │  │ + SLA        │  │  Registry    │  │   PII + Access     │  │        │
│  │  └──────────────┘  └──────────────┘  └────────────────────┘  │        │
│  └──────────────────────────────────────────────────────────────┘        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Stack technique

| Composant          | Technologie                       | Rôle                                   |
|--------------------|-----------------------------------|----------------------------------------|
| **LLM**            | Groq (`llama-3.3-70b-versatile`)  | Tous les agents IA (intent, Cypher, trust, analyst) |
| **Fallback LLM**   | `llama-3.1-8b-instant`            | Modèle de secours (rate limit)         |
| **Orchestration**  | LangGraph (`StateGraph`)          | Pipeline multi-agent avec état typé    |
| **Framework LLM**  | LangChain + `langchain-groq`      | Templates de prompts, chaînes, parsers |
| **Base graphe**    | Neo4j 5 Community (Docker)        | Stockage graphe natif + exécution Cypher |
| **API**            | FastAPI + Uvicorn                 | Endpoints REST + datasource Grafana    |
| **UI**             | Streamlit                         | Panel interactif (6 onglets)           |
| **Visualisation**  | Grafana + SimpleJSON plugin       | Dashboards et monitoring               |
| **Conteneurs**     | Docker Compose                    | Déploiement 4 services                 |
| **Langage**        | Python 3.10                       | Tout le backend                        |
| **Validation**     | Pydantic v2                       | Modèles de données, config, contrats   |

---

## 🗂️ Structure du projet

```
chall2tal/
├── main.py                              # Point d'entrée (demo | --api | --streamlit)
├── streamlit_app.py                     # Interface Streamlit (6 onglets)
├── docker-compose.yml                   # 4 services : Neo4j, Grafana, API, UI
├── Dockerfile                           # Image Python 3.10-slim
├── requirements.txt                     # Dépendances Python
├── .env                                 # Variables d'environnement (GROQ_API_KEY, NEO4J_*, ...)
├── telco_churn_with_all_feedback.csv    # Dataset réel (7 043 lignes × 23 colonnes)
│
├── ados/                                # Package principal ADOS
│   ├── __init__.py                      # Exportation ADOSSystem
│   ├── config.py                        # Configuration centralisée (Groq, Neo4j, Grafana)
│   ├── system.py                        # ADOSSystem — racine de composition (8 étapes init)
│   ├── logging_config.py                # Logging JSON structuré + IDs de corrélation
│   │
│   ├── layer1_interface/                # Couche d'interface utilisateur
│   │   └── api.py                       # FastAPI : /query, /quality, /governance, /grafana/*
│   │
│   ├── layer2_kernel/                   # Couche d'orchestration IA
│   │   ├── agents.py                    # 5 agents LLM + retry + fallback + cache TTL
│   │   ├── orchestrator.py              # LangGraph StateGraph (6 nœuds) — exécution Neo4j
│   │   └── knowledge_graph.py           # Neo4jKnowledgeGraph (CSV → graphe propriété)
│   │
│   ├── layer3_data_fabric/              # Intelligence Data Fabric
│   │   ├── metadata_catalog.py          # Métadonnées actives : suivi d'usage, alertes
│   │   ├── quality_engine.py            # Scoring qualité 5 dimensions (0–100, grades A–F)
│   │   ├── semantic_layer.py            # Glossaire métier + annotations colonnes
│   │   └── lineage_service.py           # Lignage DAG + visualisation ASCII
│   │
│   └── layer4_data_mesh/                # Principes Data Mesh
│       ├── data_product.py              # DataProduct + DataContract + SLA + SchemaContract
│       └── governance.py                # FederatedGovernance + détection PII + AccessPolicy
│
├── grafana/provisioning/                # Auto-provisioning Grafana
│   └── datasources/ados.yml
│
├── data/parquet/                        # Exports Parquet (optionnel)
└── logs/                                # Logs JSON structurés (ados.log)
```

---

## 📚 Description détaillée des couches

### Layer 4 — Data Mesh

**Fichiers** : `ados/layer4_data_mesh/data_product.py`, `ados/layer4_data_mesh/governance.py`

Le Data Mesh implémente les 4 principes fondamentaux :

#### 1. Domain-Oriented Ownership (Propriété par domaine)

Chaque `DataProduct` a un propriétaire (`owner`), une équipe (`team`) et un domaine (`domain`). Le registre auto-détecte ces attributs lors de l'enregistrement.

```python
class DataProduct:
    name: str
    domain: str          # Ex: "telecom_analytics"
    owner: str           # Ex: "data_engineering_team"
    contract: DataContract  # Contrat + SLA
    dataframe: pd.DataFrame
    schema: dict
```

#### 2. Data as a Product (Données comme Produit)

Chaque produit de données possède un **contrat** formel :

```python
class DataContract:
    version: str         # Ex: "1.0.0"
    owner: str           # Propriétaire responsable
    sla: SLA             # Accord de niveau de service
    schema_contracts: List[SchemaContract]  # Contrat par colonne
    quality_expectations: dict

class SLA:
    freshness_hours: int = 24      # Données fraîches sous 24h
    availability_pct: float = 99.0 # Disponibilité 99%
    max_query_time_ms: int = 5000  # Temps de réponse max

class SchemaContract:
    column_name: str
    data_type: str       # "string", "int64", "float64"
    nullable: bool
    unique: bool
    allowed_values: list # Valeurs autorisées (si applicable)
```

#### 3. Self-Serve Platform (Plateforme en libre-service)

- `DataProductRegistry` auto-découvre les fichiers CSV et génère les contrats
- `discover_and_load()` scanne le dossier, charge les CSV, crée les `DataProduct` avec contrats
- `publish_product()` permet aux équipes domaines d'enregistrer leurs données

#### 4. Federated Governance (Gouvernance fédérée)

**6 règles globales** appliquées automatiquement :

| Règle | Description |
|-------|------------|
| `contract_required` | Chaque produit doit avoir un contrat |
| `pii_detection` | Détection automatique des données personnelles |
| `min_quality_score` | Score qualité minimum = 70/100 |
| `schema_validation` | Le schéma doit correspondre au contrat |
| `owner_required` | Un propriétaire doit être assigné |
| `sla_compliance` | Vérification des SLA (fraîcheur, disponibilité) |

**Détection PII** — patterns regex détectés automatiquement :
- Email, téléphone, numéro de sécurité sociale, carte bancaire, adresse IP
- Colonnes sensibles : tout ce qui contient "name", "email", "phone", "address", "ssn"

**Contrôle d'accès basé sur les rôles** :

| Rôle | Accès |
|------|-------|
| `analyst` | Lecture seule, pas de PII |
| `manager` | Lecture + PII masqué |
| `data_engineer` | Accès complet |
| `admin` | Accès complet + écriture |

**Validation des requêtes** — bloque les opérations dangereuses :
`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, `SET`, `REMOVE`, `CREATE`, `MERGE`

---

### Layer 3 — Data Fabric

**Fichiers** : `ados/layer3_data_fabric/`

#### Moteur de qualité (`quality_engine.py`)

Évaluation automatique sur **5 dimensions** :

| Dimension      | Poids | Ce qu'elle mesure |
|----------------|-------|-------------------|
| **Completeness** | 30%  | % de valeurs non-null par colonne |
| **Uniqueness**   | 15%  | Détection de lignes dupliquées |
| **Validity**     | 25%  | Valeurs conformes au type/contraintes |
| **Consistency**  | 20%  | Détection d'outliers (méthode IQR) |
| **Timeliness**   | 10%  | Évaluation de la fraîcheur des données |

- Score composite : 0–100
- Grades : A (≥90), B (≥80), C (≥70), D (≥60), F (<60)
- Rapport détaillé par colonne : completeness, uniqueness, validity

```python
class QualityReport:
    product_name: str
    composite_score: float      # 0–100
    grade: str                  # "A"–"F"
    dimensions: List[DimensionScore]
    column_scores: List[ColumnQuality]
    critical_issues: List[str]
    assessed_at: datetime
```

#### Couche sémantique (`semantic_layer.py`)

- **Glossaire métier** : 7 termes prédéfinis (Churn Rate, CLV, ARPU, Contract Type, Service Bundle, Tenure, Senior Customer)
- **Annotations de colonnes** : 23 colonnes annotées avec nom métier, description, type sémantique, hints d'agrégation
- **Résolution de termes** : Mappe les requêtes utilisateur (français/anglais) vers les colonnes techniques

```python
# Exemple d'enrichissement d'une requête
semantic_layer.enrich_query_context("analyse du taux de churn")
# → {"enriched": True, "resolved_terms": [{"term": "Churn Rate", "columns": ["Churn"]}]}
```

#### Catalogue de métadonnées actives (`metadata_catalog.py`)

- **Suivi d'usage** : chaque requête est journalisée (produit, colonnes, rôle, timestamp)
- **Alertes** : baisse de qualité, changements de schéma, pics d'usage
- **Recommandations IA** : propriétaires manquants, imputation de nulls, pré-agrégation suggérée

```python
class CatalogEntry:
    domain_name: str
    row_count: int
    columns: List[ColumnMetadata]
    quality_score: float
    quality_grade: str
    owner: str
    created_at: datetime
    usage_count: int
    recommendations: List[str]
```

#### Service de lignage (`lineage_service.py`)

Suivi du lignage complet de chaque requête sous forme de **DAG** (graphe acyclique dirigé) :

```
Source: telco_churn_with_all_feedback (csv)
    │
    └──[read]──▶ LLM-Generated Cypher (transform)
                    │
                    └──[analyze]──▶ LLM Analysis (transform)
                                      │
                                      └──[certify]──▶ Certified Result (sink)
```

- Traces identifiées par UUID
- Chaque nœud a un type (`source`, `transform`, `sink`) et des métadonnées
- Visualisation ASCII dans le terminal et Streamlit

---

### Layer 2 — Kernel (LLM + Neo4j)

**Fichiers** : `ados/layer2_kernel/`

#### 5 Agents LLM (`agents.py`)

Tous les agents utilisent **Groq** via **LangChain** — aucune logique if/else codée en dur.

| Agent | Rôle | Entrée | Sortie |
|-------|------|--------|--------|
| **Intent Agent** | Parse la question utilisateur | Query + Schema | JSON structuré (action, colonnes, filtres, groupby) |
| **Discovery Agent** | Identifie les produits pertinents | Intent + Schema + KG context | Produits et colonnes à utiliser |
| **Query Agent** | Génère une requête Cypher | Intent + Discovery + Schema | Requête Cypher Neo4j |
| **Trust Judge** | Valide la requête et résultats | Query + Cypher + Results | Score 0–100, approve/reject, 5 critères |
| **Analyst Agent** | Analyse intelligente des résultats | Query + Cypher + Results | Summary, insights, recommandations |

**Trust Judge — 5 critères d'évaluation (chacun noté sur 20)** :

| Critère | Ce qu'il évalue |
|---------|----------------|
| Correctness | La requête Cypher répond-elle bien à la question ? |
| Safety | Opérations dangereuses (DELETE, SET, REMOVE) ? |
| Coherence | Résultats logiques et cohérents ? |
| Data Quality | Valeurs nulles, incohérences dans les résultats ? |
| PII Exposure | Données sensibles exposées ? |

#### Orchestrateur LangGraph (`orchestrator.py`)

Le pipeline est un **StateGraph** LangGraph avec 6 nœuds et des transitions conditionnelles :

```python
class PipelineState(TypedDict):
    user_query: str          # Question utilisateur
    user_role: str           # Rôle (analyst, manager, etc.)
    schema_context: str      # Schéma des données pour le LLM
    kg_context: str          # Description du graphe pour le LLM
    knowledge_graph: Any     # Instance Neo4jKnowledgeGraph
    intent: dict             # Résultat de l'agent Intent
    discovery: dict          # Résultat de l'agent Discovery
    sql: str                 # Requête Cypher générée
    result_data: list        # Données retournées par Neo4j
    trust: dict              # Évaluation du Trust Judge
    analysis: dict           # Analyse de l'agent Analyst
    steps: list              # Trace de chaque étape
    error: str | None        # Erreur éventuelle
    status: str              # État du pipeline
    lineage_trace_id: str    # ID de trace de lignage
```

**Graphe d'exécution** :
```
intent → discovery → query_build → execute → [error? → END] → trust → analyze → END
```

#### Knowledge Graph Neo4j (`knowledge_graph.py`)

La classe `Neo4jKnowledgeGraph` est responsable de :

1. **Charger le CSV dans Neo4j** : `load_csv_as_graph(product_name, df)`
2. **Vérifier le graphe existant** : `_graph_already_loaded(expected_rows)` — évite le rechargement
3. **Exécuter des requêtes Cypher** : `query_cypher(cypher)` — interface pour les requêtes LLM
4. **Fournir le contexte au LLM** : `get_context_for_llm()` — inspection dynamique du schéma réel

---

### Layer 1 — Interface

**Fichiers** : `ados/layer1_interface/api.py`, `streamlit_app.py`

#### Streamlit UI (`streamlit_app.py`)

Interface web avec **6 onglets** :

| Onglet | Fonctionnalité |
|--------|---------------|
| 🔍 **Query** | Requêtes en langage naturel → pipeline LLM → résultats + Cypher + Trust + Analyse |
| 📊 **Grafana** | Iframe Grafana intégré + guide de configuration |
| 📋 **Catalog** | Catalogue de métadonnées avec annotations, qualité, recommandations |
| ✅ **Quality** | Rapports qualité détaillés (5 dimensions, par colonne) |
| 🏛️ **Governance** | Règles de gouvernance, politiques d'accès, PII, compliance |
| 🔗 **Lineage** | Traces de lignage de toutes les requêtes exécutées |

**Fonctionnalités supplémentaires** :
- Bouton 🔄 **Réinitialiser** pour forcer un redémarrage propre du système
- 5 requêtes d'exemple prédéfinies (clic rapide)
- Affichage des étapes du pipeline en cas d'erreur (debug)
- Display du Cypher généré, résultats, Trust Score, analyse LLM, lignage

#### FastAPI (`api.py`)

REST API complète avec documentation Swagger automatique :

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/query` | Exécuter une requête langage naturel |
| GET | `/api/v1/catalog` | Résumé du catalogue |
| GET | `/api/v1/kg` | Statistiques du graphe Neo4j |
| GET | `/api/v1/lineage` | Traces de lignage |
| GET | `/api/v1/quality` | Résumé qualité global |
| GET | `/api/v1/quality/{name}` | Rapport qualité d'un produit |
| GET | `/api/v1/governance` | Compliance gouvernance |
| GET | `/api/v1/semantic` | Glossaire + couche sémantique |
| GET | `/api/v1/recommendations/{name}` | Recommandations IA |
| GET | `/api/v1/usage` | Analytique d'usage |
| POST | `/grafana/query` | Datasource SimpleJSON Grafana |
| POST | `/grafana/search` | Search endpoint Grafana |

---

## 🔄 Pipeline LangGraph (flux de traitement)

```
Question utilisateur (langage naturel)
       │
       ▼
 ┌───────────┐
 │  Intent   │  LLM parse la question → JSON structuré
 │  Agent    │  {action, description, relevant_columns, filters, groupby, complexity}
 └─────┬─────┘
       ▼
 ┌───────────┐
 │ Discovery │  LLM sélectionne les data products pertinents
 │  Agent    │  + colonnes nécessaires, stratégie de jointure
 └─────┬─────┘
       ▼
 ┌───────────┐
 │  Cypher   │  LLM génère une requête Cypher Neo4j
 │ Builder   │  (utilise le schéma dynamique du graphe réel)
 └─────┬─────┘
       ▼
 ┌───────────┐
 │  Execute  │  Neo4j exécute le Cypher contre le graphe
 │  (Neo4j)  │  propriété → retourne les résultats
 └─────┬─────┘
       │
   [erreur?] ──── oui ───▶ FIN (erreur propagée)
       │
      non
       ▼
 ┌───────────┐
 │  Trust    │  LLM valide le Cypher + les résultats
 │  Judge    │  trust_score 0–100 (5 critères × 20 pts)
 └─────┬─────┘
       ▼
 ┌───────────┐
 │  Analyst  │  LLM génère un résumé, des key insights,
 │  Agent    │  tendances, anomalies, recommandations,
 └─────┬─────┘  suggestions de visualisation
       ▼
  Résultat final (JSON) + Lignage enregistré
```

---

## 🕸️ Modèle de graphe Neo4j

Le CSV est chargé dans Neo4j sous forme de **graphe propriété** :

```
(:Customer {customerID, gender, SeniorCitizen, Partner, Dependents,
            tenure, PhoneService, MultipleLines, OnlineSecurity,
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges, ...})
    │
    ├──[:HAS_CONTRACT]──▶ (:Contract {type: "Month-to-month"|"One year"|"Two year"})
    │
    ├──[:USES_INTERNET]──▶ (:InternetService {type: "DSL"|"Fiber optic"|"No"})
    │
    ├──[:PAYS_BY]──▶ (:PaymentMethod {method: "Electronic check"|"Mailed check"|...})
    │
    └──[:HAS_CHURN_STATUS]──▶ (:ChurnStatus {status: "Yes"|"No"})
```

**Statistiques du graphe** :
- **~7 055 nœuds** : 7 043 Customer + 3 Contract + 3 InternetService + 4 PaymentMethod + 2 ChurnStatus
- **~28 172 relations** : 7 043 HAS_CONTRACT + 7 043 USES_INTERNET + 7 043 PAYS_BY + 7 043 HAS_CHURN_STATUS

**Exemples de requêtes Cypher (générées par le LLM)** :

```cypher
-- Taux de churn par type de contrat
MATCH (c:Customer)-[:HAS_CONTRACT]->(con:Contract)
MATCH (c)-[:HAS_CHURN_STATUS]->(s:ChurnStatus)
RETURN con.type AS contract_type,
       count(c) AS total_customers,
       sum(CASE WHEN s.status = 'Yes' THEN 1 ELSE 0 END) AS churned,
       round(toFloat(sum(CASE WHEN s.status = 'Yes' THEN 1 ELSE 0 END)) / count(c) * 100, 2) AS churn_rate_pct

-- Charges mensuelles moyennes par service internet
MATCH (c:Customer)-[:USES_INTERNET]->(i:InternetService)
RETURN i.type AS internet_service, round(avg(c.MonthlyCharges), 2) AS avg_charges

-- Clients seniors ayant churné
MATCH (c:Customer)-[:HAS_CHURN_STATUS]->(s:ChurnStatus)
WHERE c.SeniorCitizen = 1 AND s.status = 'Yes'
RETURN count(c) AS senior_churned

-- Taux de churn par méthode de paiement
MATCH (c:Customer)-[:PAYS_BY]->(p:PaymentMethod)
MATCH (c)-[:HAS_CHURN_STATUS]->(s:ChurnStatus)
RETURN p.method AS payment_method,
       round(toFloat(sum(CASE WHEN s.status = 'Yes' THEN 1 ELSE 0 END)) / count(c) * 100, 2) AS churn_rate_pct
```

**Pour visualiser le graphe dans Neo4j Browser** (http://localhost:7475) :

```cypher
-- Aperçu du graphe (vue Graph)
MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 200

-- Schéma complet
CALL db.schema.visualization()

-- Voisinage d'un client
MATCH p=(c:Customer {customerID: 'XXXX-XXXXX'})-[*1..2]-(n) RETURN p
```

---

## 🔄 Mécanisme de résilience LLM

Le système implémente un mécanisme de résilience multi-niveaux dans `_invoke_with_retry()` :

```
┌─ 1. Cache TTL (300s par défaut) ──────────────────────────────────┐
│  Si la même requête a déjà été faite → retourner le cache         │
└──────────────────────────────────────────────────────────────────┘
           │ (pas en cache)
           ▼
┌─ 2. Modèle principal (llama-3.3-70b-versatile) ──────────────────┐
│  3 tentatives avec backoff exponentiel (1s, 2s, 4s)              │
│  Si erreur 429 quotidienne (TPD) → passer directement aux        │
│  fallbacks sans retries supplémentaires                           │
└──────────────────────────────────────────────────────────────────┘
           │ (rate limit persistant)
           ▼
┌─ 3. Modèles de fallback ─────────────────────────────────────────┐
│  Essai séquentiel de chaque modèle configuré                     │
│  (défaut : llama-3.1-8b-instant)                                 │
│  Les modèles décommissionnés (erreur 400) sont sautés            │
└──────────────────────────────────────────────────────────────────┘
           │ (tous épuisés)
           ▼
┌─ 4. Erreur propagée ─────────────────────────────────────────────┐
│  Le message d'erreur réel (rate limit, etc.) est remonté vers    │
│  l'UI Streamlit avec un message clair et un lien Groq            │
└──────────────────────────────────────────────────────────────────┘
```

**Classification des erreurs** :
- `_is_rate_limit(err)` — détecte les erreurs 429 / "rate_limit"
- `_is_daily_limit(err)` — détecte les limites quotidiennes TPD (tokens-per-day)
- `_is_model_unavailable(err)` — détecte les modèles décommissionnés (erreur 400)

**Configuration** (via `.env` ou variables d'environnement) :
```bash
LLM_FALLBACK_MODELS=llama-3.1-8b-instant,gemma2-9b-it    # Modèles de secours
LLM_CACHE_TTL=300                                          # Cache 5 minutes (0 = désactivé)
```

---

## 📊 Initialisation du système (8 étapes)

Lors du démarrage, `ADOSSystem.initialize()` exécute 8 étapes séquentielles :

| Étape | Action | Détail |
|-------|--------|--------|
| **1/8** | Charger les CSV comme Data Products | `DataProductRegistry` scanne le dossier, charge les CSV, génère `DataContract` + `SLA` + `SchemaContract` par colonne |
| **2/8** | Enregistrer dans le catalogue | `MetadataCatalog` crée une `CatalogEntry` avec schéma, types, statistiques par colonne |
| **3/8** | Évaluation de la qualité | `DataQualityEngine` évalue les 5 dimensions, produit un score composite et un grade (A–F) |
| **4/8** | Charger la couche sémantique | `SemanticLayer` charge le glossaire métier (7 termes) et les annotations de colonnes (23 colonnes) |
| **5/8** | Enrichir le catalogue | Active Metadata : chaque colonne reçoit un nom métier, une description et un type sémantique |
| **6/8** | Contrôles de compliance | `FederatedGovernance` applique les 6 règles globales, détecte les PII, crée les politiques d'accès |
| **7/8** | Construire le graphe Neo4j | `Neo4jKnowledgeGraph` charge 7 043 clients dans Neo4j avec ~28 172 relations (skip si déjà chargé) |
| **8/8** | Configurer l'orchestrateur | `LangGraphOrchestrator` compile le `StateGraph` avec 6 nœuds d'agents LLM connectés |

---

## 🚀 Installation et démarrage rapide

### Prérequis

- **Python 3.10+** (avec `venv`)
- **Docker Desktop** (v20+) avec **Docker Compose** (v2+)
- **Clé API Groq** — gratuite sur [console.groq.com](https://console.groq.com)

### Option 1 : Docker Compose (recommandé)

```bash
# 1. Cloner le repo
git clone <repo-url>
cd chall2tal

# 2. Configurer la clé API
echo "GROQ_API_KEY=gsk_votre_clé_ici" > .env

# 3. Lancer les 4 services
docker compose up --build -d

# 4. Vérifier que Neo4j est healthy
docker compose ps
```

**Services démarrés** :

| Service | Conteneur | Port | Description |
|---------|-----------|------|-------------|
| Neo4j | `ados-neo4j` | `7475` (browser), `7688` (bolt) | Base de données graphe |
| Grafana | `ados-grafana` | `3001` | Dashboards (admin/admin) |
| ADOS API | `ados-api` | `8001` | FastAPI + datasource Grafana |
| ADOS UI | `ados-ui` | `8502` | Streamlit control panel |

### Option 2 : Développement local

```bash
# 1. Créer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer la clé API
echo "GROQ_API_KEY=gsk_votre_clé_ici" > .env

# 4. Démarrer Neo4j (obligatoire)
docker compose up neo4j -d

# 5. Attendre que Neo4j soit healthy
docker compose ps   # Neo4j doit être "healthy"

# 6. Lancer l'application
python main.py                          # Mode demo (terminal)
python main.py --api                    # FastAPI sur :8000
streamlit run streamlit_app.py          # Streamlit sur :8501
```

---

## 💻 Utilisation

### Mode Demo (terminal)

```bash
python main.py
```

Exécute une requête de démonstration et affiche :
- Intent parsé, produits découverts, Cypher généré
- Résultats, Trust Score, analyse LLM
- Lignage ASCII complet

### Mode API (FastAPI)

```bash
python main.py --api
# → API disponible sur http://localhost:8000/docs
```

### Mode UI (Streamlit)

```bash
python main.py --streamlit
# → http://localhost:8501

# Ou directement :
streamlit run streamlit_app.py --server.port 8502 --server.headless true
# → http://localhost:8502
```

### Exemple de requête API

```bash
curl -X POST http://localhost:8001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Quel est le taux de churn par type de contrat ?", "user_role": "analyst"}'
```

**Réponse** :
```json
{
  "status": "completed",
  "user_query": "Quel est le taux de churn par type de contrat ?",
  "sql": "MATCH (c:Customer)-[:HAS_CONTRACT]->(con:Contract) ...",
  "result_data": [
    {"contract_type": "Month-to-month", "total_customers": 3875, "churned": 1655, "churn_rate_pct": 42.71},
    {"contract_type": "One year", "total_customers": 1473, "churned": 166, "churn_rate_pct": 11.27},
    {"contract_type": "Two year", "total_customers": 1695, "churned": 48, "churn_rate_pct": 2.83}
  ],
  "trust": {"trust_score": 96, "approved": true},
  "analysis": {
    "summary": "L'analyse montre que les contrats mensuels ont un taux de churn 15x supérieur aux contrats de 2 ans.",
    "key_insights": ["Le contrat Month-to-month a un churn de 42.7%", "Les contrats longs fidélisent"],
    "recommendations": ["Inciter la migration vers des contrats annuels"]
  },
  "governance_query_check": "pass",
  "lineage_trace_id": "abc123"
}
```

---

## 📡 API REST (FastAPI)

Documentation Swagger auto-générée : `http://localhost:8001/docs`

### Tous les endpoints

| Méthode | Endpoint | Description | Body |
|---------|----------|-------------|------|
| `GET` | `/api/v1/health` | Vérification santé | — |
| `POST` | `/api/v1/query` | Requête langage naturel | `{"query":"...", "user_role":"analyst"}` |
| `GET` | `/api/v1/catalog` | Catalogue de métadonnées | — |
| `GET` | `/api/v1/kg` | Stats du graphe Neo4j | — |
| `GET` | `/api/v1/lineage` | Traces de lignage | — |
| `GET` | `/api/v1/quality` | Résumé qualité global | — |
| `GET` | `/api/v1/quality/{name}` | Rapport qualité détaillé | — |
| `GET` | `/api/v1/governance` | Compliance gouvernance | — |
| `GET` | `/api/v1/semantic` | Glossaire + couche sémantique | — |
| `GET` | `/api/v1/recommendations/{name}` | Recommandations IA | — |
| `GET` | `/api/v1/usage` | Analytique d'usage | — |
| `GET` | `/grafana/` | Health check Grafana | — |
| `POST` | `/grafana/search` | Liste des métriques | — |
| `POST` | `/grafana/query` | Données métriques | `{"targets":[...]}` |

---

## ⚙️ Configuration

### Variables d'environnement (`.env`)

```bash
# === Obligatoire ===
GROQ_API_KEY=gsk_votre_clé_api           # Clé API Groq (gratuite sur console.groq.com)

# === Neo4j (optionnel — valeurs par défaut) ===
NEO4J_URI=bolt://localhost:7688           # URI de connexion Neo4j
NEO4J_USER=neo4j                          # Utilisateur Neo4j
NEO4J_PASSWORD=ados_secret                # Mot de passe Neo4j

# === LLM (optionnel) ===
LLM_MODEL=llama-3.3-70b-versatile        # Modèle primaire Groq
LLM_FALLBACK_MODELS=llama-3.1-8b-instant # Modèles de fallback (virgule-séparés)
LLM_CACHE_TTL=300                         # Cache LLM en secondes (0 = désactivé)

# === Grafana (optionnel) ===
GRAFANA_URL=http://localhost:3001         # URL Grafana
```

### Limites du tier gratuit Groq

| Modèle | Tokens/min | Tokens/jour | Requêtes/min |
|--------|-----------|-------------|--------------|
| llama-3.3-70b-versatile | 6 000 | 100 000 | 30 |
| llama-3.1-8b-instant | 20 000 | 500 000 | 30 |

> **Note** : Le mécanisme de fallback + cache aide à rester dans les limites du tier gratuit. Pour un usage intensif, passez au [plan payant Groq](https://console.groq.com).

---

## 📁 Dataset

**Fichier** : `telco_churn_with_all_feedback.csv`

| Propriété | Valeur |
|-----------|--------|
| Lignes | 7 043 clients |
| Colonnes | 23 |
| Domaine | Télécommunications — analyse du churn |
| Source | IBM Telco Customer Churn dataset (enrichi) |

**Colonnes principales** :

| Colonne | Type | Description |
|---------|------|-------------|
| `customerID` | string | Identifiant unique client |
| `gender` | string | Genre (Male/Female) |
| `SeniorCitizen` | int | Senior (1) ou non (0) |
| `Partner` | string | A un partenaire (Yes/No) |
| `Dependents` | string | A des dépendants (Yes/No) |
| `tenure` | int | Nombre de mois comme client |
| `PhoneService` | string | Service téléphonique (Yes/No) |
| `MultipleLines` | string | Lignes multiples |
| `InternetService` | string | Type d'internet (DSL/Fiber optic/No) |
| `OnlineSecurity` | string | Sécurité en ligne |
| `OnlineBackup` | string | Sauvegarde en ligne |
| `DeviceProtection` | string | Protection appareil |
| `TechSupport` | string | Support technique |
| `StreamingTV` | string | TV en streaming |
| `StreamingMovies` | string | Films en streaming |
| `Contract` | string | Type de contrat |
| `PaperlessBilling` | string | Facturation dématérialisée |
| `PaymentMethod` | string | Méthode de paiement |
| `MonthlyCharges` | float | Charges mensuelles ($) |
| `TotalCharges` | float | Charges totales ($) |
| `Churn` | string | A churné (Yes/No) |

---

## 📋 Logging et observabilité

### Format de log (JSON structuré)

```json
{
  "ts": "2026-02-18T10:15:00.000000+00:00",
  "level": "INFO",
  "logger": "ados.layer2_kernel.agents",
  "cid": "a1b2c3d4",
  "msg": "QueryAgent: Cypher generated in 464ms"
}
```

| Champ | Description |
|-------|-------------|
| `ts` | Timestamp UTC ISO 8601 |
| `level` | Niveau : DEBUG, INFO, WARNING, ERROR |
| `logger` | Module Python source |
| `cid` | Correlation ID (trace unique par requête) |
| `msg` | Message du log |

### Destinations
- **Console** (stdout) — affichage en temps réel
- **Fichier** — `logs/ados.log` (persistant)

### Correlation IDs

Chaque requête reçoit un ID de corrélation unique (`set_correlation_id()`), permettant de tracer toutes les étapes d'une même requête :

```
cid=a1b2c3d4 IntentAgent: parsed intent in 579ms
cid=a1b2c3d4 DiscoveryAgent: found 1 products in 712ms
cid=a1b2c3d4 QueryAgent: Cypher generated in 464ms
cid=a1b2c3d4 Neo4j: Cypher returned 3 rows in 42ms
cid=a1b2c3d4 TrustJudge: score=96/100 in 820ms
cid=a1b2c3d4 Pipeline Complete: completed, trust=96, rows=3 in 4217ms
```

---

## 📐 Diagrammes de séquence

### Requête utilisateur complète

```
Utilisateur          Streamlit         ADOSSystem        Orchestrator       Neo4j
    │                    │                  │                  │               │
    │─── "taux de ──────▶│                  │                  │               │
    │    churn par        │──── query() ───▶│                  │               │
    │    contrat ?"       │                  │── check_access() │               │
    │                    │                  │── enrich_query() │               │
    │                    │                  │── process_query()▶│               │
    │                    │                  │                  │─ Intent LLM   │
    │                    │                  │                  │─ Discovery LLM│
    │                    │                  │                  │─ Cypher LLM   │
    │                    │                  │                  │─ execute() ──▶│
    │                    │                  │                  │◀── 3 rows ────│
    │                    │                  │                  │─ Trust LLM    │
    │                    │                  │                  │─ Analyst LLM  │
    │                    │                  │                  │─ lineage()    │
    │                    │                  │◀── final_state ──│               │
    │                    │◀── result ───────│                  │               │
    │◀── Cypher + ───────│                  │                  │               │
    │   résultats +      │                  │                  │               │
    │   trust + analyse  │                  │                  │               │
```

### Mécanisme de fallback LLM

```
Agent               _invoke_with_retry          Groq API
  │                       │                        │
  │── invoke(params) ────▶│                        │
  │                       │── cache lookup          │
  │                       │   (hit → return)        │
  │                       │── try primary model ──▶│
  │                       │◀── 429 TPD ────────────│
  │                       │── skip remaining retries│
  │                       │── try llama-3.1-8b ──▶│
  │                       │◀── 200 OK ─────────────│
  │                       │── cache store           │
  │◀── result ────────────│                        │
```

---

## 🔧 Dépannage

### Erreur 429 — Rate Limit Groq

**Symptôme** : "❌ Pipeline error: No Cypher query was generated" ou "rate_limit_exceeded"

**Cause** : Le tier gratuit Groq a une limite de 100 000 tokens/jour.

**Solutions** :
1. **Attendre** le reset quotidien (minuit UTC)
2. **Vérifier le cache** — les requêtes identiques sont servies depuis le cache (5 min TTL)
3. **Changer de clé** : mettre à jour `GROQ_API_KEY` dans `.env`
4. **Passer au plan payant** : [console.groq.com](https://console.groq.com)

### Streamlit affiche d'anciennes erreurs

**Symptôme** : L'application Streamlit montre des erreurs même après correction.

**Solutions** :
1. Cliquer sur le bouton **🔄 Réinitialiser le système** dans l'UI
2. Supprimer le cache bytecode :
   ```bash
   Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
   ```
3. Relancer Streamlit :
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

### Neo4j non accessible

**Symptôme** : Erreur de connexion à Neo4j.

**Solutions** :
1. Vérifier que le conteneur est en cours d'exécution : `docker compose ps`
2. Attendre le healthcheck (30s après démarrage)
3. Vérifier les ports : `7688` (bolt) et `7475` (browser)
4. Tester la connexion : ouvrir http://localhost:7475 dans le navigateur

### Le graphe n'est pas chargé

**Symptôme** : Requêtes Cypher retournent 0 résultats.

**Solutions** :
1. Vérifier dans Neo4j Browser : `MATCH (n) RETURN count(n)` — doit retourner ~7 055
2. Si 0, relancer l'application : le système recharge automatiquement le CSV
3. Forcer le rechargement : supprimer les nœuds existants puis relancer :
   ```cypher
   MATCH (n) DETACH DELETE n
   ```

---

## 📝 Licence

Ce projet est à des fins éducatives et de démonstration.

---

<p align="center">
  <b>ADOS v2</b> — Built with 🧠 LLM Agents • 🔗 LangGraph • 🕸️ Neo4j Graph • 📦 Data Mesh • 🧬 Data Fabric • 📊 Grafana
</p>
#   c h l l 2  
 