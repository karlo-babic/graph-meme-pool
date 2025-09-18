import networkx as nx
import random
import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING

from data_structures import MemeNodeData
from embeddings_utils import EmbeddingManager
from llm_service import LLMServiceInterface

if TYPE_CHECKING:
    from .graph_manager import GraphManager

logger = logging.getLogger(__name__)

class GraphDynamicsStrategy(ABC):
    """Abstract base class for defining graph topology evolution rules."""
    @abstractmethod
    def execute(self, graph_manager: 'GraphManager', generation: int):
        """Applies topological changes to the graph for a given generation."""
        pass

class NullDynamicsStrategy(GraphDynamicsStrategy):
    """A strategy that makes no changes to the graph topology (static graph)."""
    def execute(self, graph_manager: 'GraphManager', generation: int):
        pass

class FusionDivisionStrategy(GraphDynamicsStrategy):
    """A strategy that implements node fusion (merging) and division (splitting)."""

    def __init__(self, config: Dict, embedding_manager: EmbeddingManager, llm_service: LLMServiceInterface):
        self.dyn_config = config.get('dynamic_graph', {})
        self.embedding_manager = embedding_manager
        self.llm_service = llm_service

    def execute(self, graph_manager: 'GraphManager', generation: int):
        if not self.dyn_config.get('enabled', False):
            return

        initial_static_gens = self.dyn_config.get('initial_static_generations', 0)
        if generation < initial_static_gens:
            return

        if generation % self.dyn_config.get('check_every_n_generations', 1) != 0:
            return

        logger.info(f"Generation {generation}: Checking for graph dynamic updates.")
        if self.dyn_config.get('fusion', {}).get('enabled', False):
            self._perform_fusion_check(graph_manager)
        if self.dyn_config.get('division', {}).get('enabled', False):
            self._perform_division_check(graph_manager)

    def _perform_fusion_check(self, graph_manager: 'GraphManager'):
        fusion_config = self.dyn_config.get('fusion', {})
        threshold = fusion_config.get('similarity_threshold', 0.95)
        graph = graph_manager.get_graph()
        
        potential_fusions = []
        all_nodes_data = graph_manager.get_all_nodes_data()
        
        for u, v in graph.edges():
            data_u = all_nodes_data.get(u)
            data_v = all_nodes_data.get(v)
            if data_u and data_v:
                similarity = self.embedding_manager.get_similarity(data_u.current_meme, data_v.current_meme)
                if similarity > threshold:
                    potential_fusions.append((similarity, u, v))
        
        potential_fusions.sort(key=lambda x: x[0], reverse=True)
        
        fused_nodes = set()
        for similarity, u, v in potential_fusions:
            if u not in fused_nodes and v not in fused_nodes:
                self._fuse_nodes(graph_manager, u, v)
                fused_nodes.add(u)
                fused_nodes.add(v)

    def _fuse_nodes(self, graph_manager: 'GraphManager', u: Any, v: Any):
        data_u = graph_manager.get_node_data(u)
        data_v = graph_manager.get_node_data(v)
        if not data_u or not data_v: return

        score_u = data_u.current_meme_score or 0
        score_v = data_v.current_meme_score or 0

        parent1_id, parent2_id = (u, v) if score_u >= score_v else (v, u)
        parent1_data = graph_manager.get_node_data(parent1_id)
        
        new_id = graph_manager.get_next_node_id()
        new_data = MemeNodeData(
            node_id=new_id,
            current_meme=parent1_data.current_meme,
            current_meme_score=parent1_data.current_meme_score,
            group=parent1_data.group,
            parents=[parent1_id, parent2_id]
        )
        graph = graph_manager.get_graph()
        graph.add_node(new_id, data=new_data)
        
        in_edges_map = {}
        for pred, _, data in graph.in_edges(u, data=True): in_edges_map[pred] = data
        for pred, _, data in graph.in_edges(v, data=True): in_edges_map[pred] = data

        out_edges_map = {}
        for _, succ, data in graph.out_edges(u, data=True): out_edges_map[succ] = data
        for _, succ, data in graph.out_edges(v, data=True): out_edges_map[succ] = data

        for pred, data in in_edges_map.items():
            if pred not in (u, v): graph.add_edge(pred, new_id, **data)
        
        for succ, data in out_edges_map.items():
            if succ not in (u, v): graph.add_edge(new_id, succ, **data)
        
        graph.remove_node(u)
        graph.remove_node(v)
        logger.info(f"Fused nodes {u} and {v} into new node {new_id}.")

    def _perform_division_check(self, graph_manager: 'GraphManager'):
        div_config = self.dyn_config.get('division', {})
        max_nodes = div_config.get('max_graph_nodes', 250)
        graph = graph_manager.get_graph()
        
        if graph.number_of_nodes() >= max_nodes:
            return

        top_percent = div_config.get('fitness_top_percent', 0.05)
        num_candidates = math.ceil(graph.number_of_nodes() * top_percent)
        num_candidates = max(1, num_candidates)

        nodes_with_scores = [
            (node_id, data.current_meme_score or 0)
            for node_id, data in graph_manager.get_all_nodes_data().items()
        ]
        nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        candidates = []
        for node_id, score in nodes_with_scores[:num_candidates]:
            if graph.in_degree(node_id) >= div_config.get('min_in_degree', 2) and \
               graph.out_degree(node_id) >= div_config.get('min_out_degree', 2):
                candidates.append(node_id)
        
        for node_id in candidates:
            if graph.number_of_nodes() >= max_nodes -1:
                break
            self._divide_node(graph_manager, node_id)

    def _divide_node(self, graph_manager: 'GraphManager', node_id: Any):
        original_data = graph_manager.get_node_data(node_id)
        if not original_data: return

        mutated_memes = self.llm_service.mutate([original_data.current_meme] * 2)
        if not mutated_memes or len(mutated_memes) < 2 or mutated_memes[0] == mutated_memes[1]:
            logger.warning(f"Division of node {node_id} failed: LLM mutation did not produce two distinct memes.")
            return
        
        graph = graph_manager.get_graph()
        all_weights = [d['weight'] for _, _, d in graph.in_edges(node_id, data=True) if 'weight' in d] + \
                      [d['weight'] for _, _, d in graph.out_edges(node_id, data=True) if 'weight' in d]
        avg_weight = sum(all_weights) / len(all_weights) if all_weights else 0.5

        new_ids = [graph_manager.get_next_node_id(), graph_manager.get_next_node_id()]
        for i, new_id in enumerate(new_ids):
            new_data = MemeNodeData(
                node_id=new_id,
                current_meme=mutated_memes[i],
                group=original_data.group,
                parents=[node_id]
            )
            graph.add_node(new_id, data=new_data)

        ratio = self.dyn_config.get('division', {}).get('connection_subset_ratio')
        
        in_connections = list(graph.in_edges(node_id, data=True))
        out_connections = list(graph.out_edges(node_id, data=True))
        
        self._assign_split_connections(graph, in_connections, ratio, new_ids, direction='in')
        self._assign_split_connections(graph, out_connections, ratio, new_ids, direction='out')

        graph.add_edge(new_ids[0], new_ids[1], weight=avg_weight)
        graph.add_edge(new_ids[1], new_ids[0], weight=avg_weight)

        graph.remove_node(node_id)
        logger.info(f"Divided node {node_id} into new nodes {new_ids[0]} and {new_ids[1]}.")

    def _assign_split_connections(self, graph: nx.DiGraph, connections: list, ratio: float, new_ids: list, direction: str):
        if not connections: return

        total_connections = len(connections)
        num_per_node = math.ceil(total_connections * ratio)
        num_per_node = min(total_connections, num_per_node)
        num_shared = max(0, (2 * num_per_node) - total_connections)
        num_unique = num_per_node - num_shared
        
        shuffled_connections = random.sample(connections, k=total_connections)
        shared_set = shuffled_connections[:num_shared]
        unique_set_1 = shuffled_connections[num_shared : num_shared + num_unique]
        unique_set_2 = shuffled_connections[num_shared + num_unique : num_shared + num_unique + num_unique]
        remaining_set = shuffled_connections[num_shared + (2 * num_unique):]
        shared_set.extend(remaining_set)
        
        connections_for_node1 = shared_set + unique_set_1
        connections_for_node2 = shared_set + unique_set_2

        if direction == 'in':
            for pred, _, data in connections_for_node1: graph.add_edge(pred, new_ids[0], **data)
            for pred, _, data in connections_for_node2: graph.add_edge(pred, new_ids[1], **data)
        elif direction == 'out':
            for _, succ, data in connections_for_node1: graph.add_edge(new_ids[0], succ, **data)
            for _, succ, data in connections_for_node2: graph.add_edge(new_ids[1], succ, **data)