import networkx as nx
import logging
from typing import List, Any, Optional, Dict

from data_structures import MemeNodeData, PropagationEvent
from .graph_dynamics import GraphDynamicsStrategy, NullDynamicsStrategy

logger = logging.getLogger(__name__)

class GraphManager:
    """Manages the in-memory NetworkX graph, node data, and propagation history."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.propagation_history: List[PropagationEvent] = []
        self.dynamics_strategy: GraphDynamicsStrategy = NullDynamicsStrategy()
        self.next_node_id = 0
        if self.graph.nodes():
            self.next_node_id = max(list(self.graph.nodes())) + 1

    def get_graph(self) -> nx.DiGraph:
        """Returns the raw NetworkX DiGraph object."""
        return self.graph

    def set_dynamics_strategy(self, strategy: GraphDynamicsStrategy):
        """Sets the strategy for how the graph topology evolves."""
        self.dynamics_strategy = strategy
        logger.info(f"Graph dynamics strategy set to: {type(strategy).__name__}")

    def update_topology(self, generation: int):
        """Delegates the task of updating graph topology to the current strategy."""
        self.dynamics_strategy.execute(self, generation)

    def get_next_node_id(self) -> int:
        """Returns a unique node ID for creating new nodes."""
        new_id = self.next_node_id
        self.next_node_id += 1
        return new_id

    def get_node_data(self, node_id: Any) -> Optional[MemeNodeData]:
        if node_id in self.graph:
            return self.graph.nodes[node_id].get('data')
        return None

    def get_all_node_ids(self) -> List[Any]:
        return list(self.graph.nodes)

    def get_all_nodes_data(self) -> Dict[Any, MemeNodeData]:
        return {node_id: attrs.get('data') for node_id, attrs in self.graph.nodes(data=True) if attrs.get('data')}

    def get_neighbors(self, node_id: Any) -> List[Any]:
        return list(self.graph.successors(node_id)) if node_id in self.graph else []

    def get_predecessors(self, node_id: Any) -> List[Any]:
        return list(self.graph.predecessors(node_id)) if node_id in self.graph else []

    def get_edge_weight(self, u: Any, v: Any) -> float:
        return self.graph[u][v].get('weight', 0.0) if self.graph.has_edge(u, v) else 0.0

    def update_node_meme(self, node_id: Any, new_meme: str, new_score: Optional[float], generation: int):
        node_data = self.get_node_data(node_id)
        if node_data:
            node_data.current_meme = new_meme
            node_data.current_meme_score = new_score
            node_data.history.append(new_meme)
            if len(node_data.history_scores) < len(node_data.history) - 1:
                padding = [None] * (len(node_data.history) - 1 - len(node_data.history_scores))
                node_data.history_scores.extend(padding)
            node_data.history_scores.append(new_score if new_score is not None else None)

    def update_node_score(self, node_id: Any, score: float, history_index: int = -1):
        node_data = self.get_node_data(node_id)
        if node_data:
            if history_index == -1 or history_index >= len(node_data.history_scores):
                node_data.current_meme_score = score
                if node_data.history_scores:
                    node_data.history_scores[-1] = score
            else:
                node_data.history_scores[history_index] = score

    def add_received_meme(self, sender_node_id: Any, node_id: Any, meme: str, weight: float):
        node_data = self.get_node_data(node_id)
        if node_data:
            node_data.received_memes.append((sender_node_id, meme, weight))

    def clear_received_memes(self):
        for node_id in self.get_all_node_ids():
            node_data = self.get_node_data(node_id)
            if node_data:
                node_data.received_memes = []

    def record_propagation(self, generation: int, source: Any, target: Any, meme: str):
        event = PropagationEvent(generation=generation, source_node=source, target_node=target, meme=meme)
        self.propagation_history.append(event)

    def get_propagation_history(self) -> List[PropagationEvent]:
        return self.propagation_history
    
    def set_propagation_history(self, history: List[PropagationEvent]):
        self.propagation_history = history