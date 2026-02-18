"""
Dynamic Lineage Service — tracks data flow per query execution.
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ados.logging_config import get_logger

logger = get_logger(__name__)


class LineageNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_type: str  # source | transform | sink
    label: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LineageEdge(BaseModel):
    from_node: str
    to_node: str
    operation: str


class LineageGraph(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    nodes: List[LineageNode] = Field(default_factory=list)
    edges: List[LineageEdge] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DynamicLineageService:
    def __init__(self):
        self._traces: List[LineageGraph] = []

    def create_trace(self) -> LineageGraph:
        g = LineageGraph()
        logger.info(f"Lineage: new trace '{g.trace_id}'")
        return g

    def add_node(self, graph: LineageGraph, ntype: str, label: str,
                 meta: Optional[Dict] = None) -> str:
        node = LineageNode(node_type=ntype, label=label, metadata=meta or {})
        graph.nodes.append(node)
        return node.node_id

    def add_edge(self, graph: LineageGraph, from_id: str, to_id: str, op: str):
        graph.edges.append(LineageEdge(from_node=from_id, to_node=to_id, operation=op))

    def commit(self, graph: LineageGraph):
        self._traces.append(graph)
        logger.info(f"Lineage: committed '{graph.trace_id}' ({len(graph.nodes)} nodes)")

    def get_all_traces(self) -> List[LineageGraph]:
        return self._traces

    def get_trace(self, tid: str) -> Optional[LineageGraph]:
        return next((t for t in self._traces if t.trace_id == tid), None)

    def render_ascii(self, graph: LineageGraph) -> str:
        node_map = {n.node_id: n for n in graph.nodes}
        lines = [f"╔═ Lineage: {graph.trace_id} ═══"]
        for e in graph.edges:
            s = node_map.get(e.from_node)
            d = node_map.get(e.to_node)
            lines.append(f"║  [{s.label if s else '?'}] ──({e.operation})──▶ [{d.label if d else '?'}]")
        lines.append("╚═══════════════════════════════════")
        return "\n".join(lines)
