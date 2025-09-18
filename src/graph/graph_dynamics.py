import networkx as nx
import random
import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, List, TYPE_CHECKING

from data_structures import MemeNodeData
from embeddings_utils import EmbeddingManager
from llm_service import LLMServiceInterface

if TYPE_CHECKING:
    from .graph_manager import GraphManager

logger = logging.getLogger(__name__)

# --- Base Strategy and Action Interfaces ---

class GraphDynamicsStrategy(ABC):
    """Abstract base class for defining graph topology evolution rules."""
    @abstractmethod
    def execute(self, graph_manager: 'GraphManager', generation: int):
        pass

class GraphAction(ABC):
    """Represents a single, atomic rule for graph modification."""
    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def execute(self, graph_manager: 'GraphManager', generation: int):
        pass

# --- Concrete Action Implementations ---

class NodeFusionAction(GraphAction):
    """An action that fuses highly similar, connected nodes."""
    def __init__(self, config: Dict, embedding_manager: EmbeddingManager):
        super().__init__(config)
        self.embedding_manager = embedding_manager

    def execute(self, graph_manager: 'GraphManager', generation: int):
        threshold = self.config.get('similarity_threshold', 0.95)
        graph = graph_manager.get_graph()
        
        potential_fusions = []
        all_nodes_data = graph_manager.get_all_nodes_data()
        
        for u, v in graph.edges():
            data_u, data_v = all_nodes_data.get(u), all_nodes_data.get(v)
            if data_u and data_v:
                similarity = self.embedding_manager.get_similarity(data_u.current_meme, data_v.current_meme)
                if similarity > threshold:
                    potential_fusions.append((similarity, u, v))
        
        potential_fusions.sort(key=lambda x: x[0], reverse=True)
        
        fused_nodes = set()
        for _, u, v in potential_fusions:
            if u not in fused_nodes and v not in fused_nodes:
                self._fuse_nodes(graph_manager, u, v)
                fused_nodes.update([u, v])

    def _fuse_nodes(self, graph_manager: 'GraphManager', u: Any, v: Any):
        data_u = graph_manager.get_node_data(u)
        data_v = graph_manager.get_node_data(v)
        if not data_u or not data_v: return

        score_u, score_v = data_u.current_meme_score or 0, data_v.current_meme_score or 0
        parent1_id, parent2_id = (u, v) if score_u >= score_v else (v, u)
        parent1_data = graph_manager.get_node_data(parent1_id)
        
        new_id = graph_manager.get_next_node_id()
        new_data = MemeNodeData(node_id=new_id, current_meme=parent1_data.current_meme,
                                current_meme_score=parent1_data.current_meme_score,
                                group=parent1_data.group, parents=[parent1_id, parent2_id])
        
        graph = graph_manager.get_graph()
        graph.add_node(new_id, data=new_data)
        
        in_edges, out_edges = {}, {}
        for pred, _, data in graph.in_edges([u, v], data=True): in_edges[pred] = data
        for _, succ, data in graph.out_edges([u, v], data=True): out_edges[succ] = data

        for pred, data in in_edges.items():
            if pred not in (u, v): graph.add_edge(pred, new_id, **data)
        for succ, data in out_edges.items():
            if succ not in (u, v): graph.add_edge(new_id, succ, **data)
        
        graph.remove_nodes_from([u, v])
        logger.info(f"Fused nodes {u} and {v} into new node {new_id}.")

class NodeDivisionAction(GraphAction):
    """An action that splits high-fitness nodes into two distinct new nodes."""
    def __init__(self, config: Dict, llm_service: LLMServiceInterface):
        super().__init__(config)
        self.llm_service = llm_service

    def execute(self, graph_manager: 'GraphManager', generation: int):
        max_nodes = self.config.get('max_graph_nodes', 250)
        graph = graph_manager.get_graph()
        if graph.number_of_nodes() >= max_nodes: return

        top_percent = self.config.get('fitness_top_percent', 0.05)
        num_candidates = max(1, math.ceil(graph.number_of_nodes() * top_percent))

        nodes_with_scores = sorted(
            [(nid, data.current_meme_score or 0) for nid, data in graph_manager.get_all_nodes_data().items()],
            key=lambda x: x[1], reverse=True
        )
        
        candidates = [
            nid for nid, score in nodes_with_scores[:num_candidates]
            if graph.in_degree(nid) >= self.config.get('min_in_degree', 2) and
               graph.out_degree(nid) >= self.config.get('min_out_degree', 2)
        ]
        
        for node_id in candidates:
            if graph.number_of_nodes() >= max_nodes - 1: break
            self._divide_node(graph_manager, node_id)

    def _divide_node(self, graph_manager: 'GraphManager', node_id: Any):
        original_data = graph_manager.get_node_data(node_id)
        if not original_data: return

        mutated = self.llm_service.mutate([original_data.current_meme] * 2)
        if not mutated or len(mutated) < 2 or mutated[0] == mutated[1]:
            logger.warning(f"Division of {node_id} failed: LLM did not produce two distinct memes.")
            return
        
        graph = graph_manager.get_graph()
        weights = [d['weight'] for _, _, d in graph.edges(node_id, data=True) if 'weight' in d] + \
                  [d['weight'] for _, _, d in graph.in_edges(node_id, data=True) if 'weight' in d]
        avg_weight = sum(weights) / len(weights) if weights else 0.5

        new_ids = [graph_manager.get_next_node_id(), graph_manager.get_next_node_id()]
        for i, new_id in enumerate(new_ids):
            new_data = MemeNodeData(node_id=new_id, current_meme=mutated[i],
                                    group=original_data.group, parents=[node_id])
            graph.add_node(new_id, data=new_data)

        ratio = self.config.get('connection_subset_ratio', 0.5)
        in_conn, out_conn = list(graph.in_edges(node_id, data=True)), list(graph.out_edges(node_id, data=True))
        
        self._assign_split_connections(graph, in_conn, ratio, new_ids, 'in')
        self._assign_split_connections(graph, out_conn, ratio, new_ids, 'out')

        graph.add_edge(new_ids[0], new_ids[1], weight=avg_weight)
        graph.add_edge(new_ids[1], new_ids[0], weight=avg_weight)
        graph.remove_node(node_id)
        logger.info(f"Divided node {node_id} into new nodes {new_ids[0]} and {new_ids[1]}.")

    def _assign_split_connections(self, graph: nx.DiGraph, connections: list, ratio: float, new_ids: list, direction: str):
        if not connections: return
        total, num_per_node = len(connections), min(len(connections), math.ceil(len(connections) * ratio))
        num_shared, num_unique = max(0, 2 * num_per_node - total), num_per_node - max(0, 2 * num_per_node - total)
        
        shuffled = random.sample(connections, k=total)
        shared_set = shuffled[:num_shared] + shuffled[num_shared + 2 * num_unique:]
        unique_set1, unique_set2 = shuffled[num_shared:num_shared+num_unique], shuffled[num_shared+num_unique:num_shared+2*num_unique]
        conn1, conn2 = shared_set + unique_set1, shared_set + unique_set2

        if direction == 'in':
            for pred, _, data in conn1: graph.add_edge(pred, new_ids[0], **data)
            for pred, _, data in conn2: graph.add_edge(pred, new_ids[1], **data)
        elif direction == 'out':
            for _, succ, data in conn1: graph.add_edge(new_ids[0], succ, **data)
            for _, succ, data in conn2: graph.add_edge(new_ids[1], succ, **data)

# --- Composite and Null Strategies ---

class NullDynamicsStrategy(GraphDynamicsStrategy):
    """A strategy that makes no changes to the graph topology (static graph)."""
    def execute(self, graph_manager: 'GraphManager', generation: int):
        pass

class CompositeDynamicsStrategy(GraphDynamicsStrategy):
    """An orchestrator that executes a list of individual GraphAction objects."""
    def __init__(self, actions: List[GraphAction], config: Dict):
        self.actions = [action for action in actions if action.config.get('enabled', False)]
        self.dyn_config = config.get('dynamic_graph', {})

    def execute(self, graph_manager: 'GraphManager', generation: int):
        if not self.dyn_config.get('enabled', False) or generation < self.dyn_config.get('initial_static_generations', 0):
            return
        
        logger.info(f"Generation {generation}: Checking for graph dynamic updates.")
        for action in self.actions:
            if generation % action.config.get('check_every_n_generations', 1) == 0:
                action.execute(graph_manager, generation)