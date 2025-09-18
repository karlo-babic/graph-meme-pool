import networkx as nx
import random
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

from data_structures import MemeNodeData

logger = logging.getLogger(__name__)

class GraphInitializer(ABC):
    """Abstract base class for creating graph topologies."""
    def __init__(self, config: Dict):
        self.config = config
        self.random_seed = config['seed']
        random.seed(self.random_seed)

    @abstractmethod
    def create(self, initial_memes: List[str]) -> nx.DiGraph:
        """Creates and returns a NetworkX DiGraph with initialized nodes."""
        pass

    def _load_initial_memes(self) -> List[str]:
        meme_file = self.config['paths']['init_memes']
        try:
            with open(meme_file, "r", encoding="utf-8") as f:
                memes = [line.strip() for line in f if line.strip()]
            if not memes:
                logger.error(f"No memes found in {meme_file}. Cannot initialize graph.")
                raise ValueError(f"Meme file {meme_file} is empty or invalid.")
            logger.info(f"Loaded {len(memes)} initial memes from {meme_file}")
            return memes
        except FileNotFoundError:
            logger.error(f"Initial meme file not found: {meme_file}")
            raise

class ExampleGraphInitializer(GraphInitializer):
    """Creates a small, simple example graph for testing."""
    def create(self, initial_memes: List[str]) -> nx.DiGraph:
        G = nx.DiGraph()
        n_nodes = 5
        for i in range(n_nodes):
            meme = initial_memes[i % len(initial_memes)]
            node_id = i + 1
            data = MemeNodeData(node_id=node_id, current_meme=meme)
            G.add_node(node_id, data=data)

        G.add_edge(1, 2, weight=0.8); G.add_edge(2, 1, weight=0.2)
        G.add_edge(2, 3, weight=0.5); G.add_edge(3, 1, weight=0.4)
        G.add_edge(4, 3, weight=0.8); G.add_edge(3, 5, weight=0.6)
        if 4 not in G: G.add_node(4, data=MemeNodeData(node_id=4, current_meme=initial_memes[3 % len(initial_memes)]))
        if 5 not in G: G.add_node(5, data=MemeNodeData(node_id=5, current_meme=initial_memes[4 % len(initial_memes)]))
        return G

class SmallWorldsInitializer(GraphInitializer):
    """Creates a graph with one or more small-world network groups."""
    def create(self, initial_memes: List[str]) -> nx.DiGraph:
        G = nx.DiGraph()
        params = self.config['graph_generation']['params']
        n, k, p, b, g, inter_p = params['n'], params['k'], params['p'], params['b'], params['g'], params['inter_p']
        num_initial_memes = len(initial_memes)
        if num_initial_memes == 0:
            logger.critical("No initial memes loaded. Cannot create graph nodes properly.")
            return G

        total_nodes = n * g
        meme_ids = self._calculate_initial_meme_ids(total_nodes, num_initial_memes, g, n)
        
        for group_idx in range(g):
            group_seed = self.random_seed + group_idx
            undirected_G = nx.watts_strogatz_graph(n, k, p, seed=group_seed)
            node_offset = group_idx * n
            for i in range(n):
                global_id = node_offset + i
                meme_index = meme_ids[global_id]
                meme = self._get_meme_from_index(initial_memes, meme_index)
                data = MemeNodeData(node_id=global_id, current_meme=meme, group=group_idx)
                G.add_node(global_id, data=data)
            
            for u_local, v_local in sorted(list(undirected_G.edges())):
                u_global, v_global = node_offset + u_local, node_offset + v_local
                weight = round(random.uniform(0.3, 0.7), 2)
                G.add_edge(u_global, v_global, weight=weight)
                if random.random() < b:
                    G.add_edge(v_global, u_global, weight=round(random.uniform(0.3, 0.7), 2))
        
        for i in range(g):
            for j in range(i + 1, g):
                nodes_i, nodes_j = list(range(i * n, (i + 1) * n)), list(range(j * n, (j + 1) * n))
                for u in nodes_i:
                    if random.random() < inter_p: G.add_edge(u, random.choice(nodes_j), weight=round(random.uniform(0.02, 0.1), 2))
                for v in nodes_j:
                    if random.random() < inter_p: G.add_edge(v, random.choice(nodes_i), weight=round(random.uniform(0.02, 0.1), 2))
        return G

    def _get_meme_from_index(self, memes: List[str], meme_index: int) -> Optional[str]:
        if not memes: return None
        return memes[meme_index] if 0 <= meme_index < len(memes) else memes[0]

    def _calculate_initial_meme_ids(self, num_nodes: int, num_memes: int, num_groups: int, nodes_per_group: int) -> List[int]:
        strategy = self.config['graph_generation']['initial_meme_assignment']
        if strategy == 'random':
            balanced_ids = [i % num_memes for i in range(num_nodes)]
            local_random = random.Random(self.random_seed)
            local_random.shuffle(balanced_ids)
            return balanced_ids
        elif strategy == 'structured':
            return [(i // nodes_per_group) % num_memes for i in range(num_nodes)]
        else:
            return [i % num_memes for i in range(num_nodes)]