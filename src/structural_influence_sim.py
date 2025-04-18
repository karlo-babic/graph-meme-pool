import networkx as nx
import random
import json
import yaml
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Any, Optional, Dict, Set
from networkx.readwrite import json_graph

# --- Configuration Loading ---

DEFAULT_CONFIG = {
    'paths': {
        'graph_save_dir': "saved",
        'graph_basename': "structural_influence_small_worlds",
        'log_file': "simulation.log"
    },
    'graph_generation': {
        'type': 'small_worlds',
        'params': {'n': 20, 'k': 4, 'p': 0.3, 'b': 0.3, 'g': 2, 'inter_p': 0.1}
    },
    'simulation': {
        'generations': 50,
    },
    'seed': 1,
    'logging': {
        'level': "INFO"
    }
}

def _recursive_update(d, u):
    """Recursively update dictionary d with values from u."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_config(config_path="config.yaml"):
    """Loads configuration from YAML file, applying defaults for missing keys."""
    config = DEFAULT_CONFIG.copy() # Start with defaults
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config:
            # Selectively update only the relevant sections
            relevant_keys = ['paths', 'graph_generation', 'simulation', 'seed', 'logging']
            relevant_yaml_config = {k: v for k, v in yaml_config.items() if k in relevant_keys}
            config = _recursive_update(config, relevant_yaml_config)
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using default structural configuration.")
    except Exception as e:
        print(f"Error loading config file '{config_path}': {e}. Using default structural configuration.")

    # Ensure directories exist (using the structural paths)
    Path(config['paths']['graph_save_dir']).mkdir(parents=True, exist_ok=True)

    return config

# --- Data Structure ---

@dataclass
class InfluenceNodeData:
    """Holds the influence counter data for a node."""
    node_id: Any
    current_value: int = 0
    history: List[int] = field(default_factory=lambda: [0]) # Start history with initial value 0
    group: Optional[int] = None # Group identifier, if applicable

# --- Graph Manager (Simplified for Structural Influence) ---

class GraphManagerStructural:
    """Manages the NetworkX graph, node data, persistence for structural influence."""

    def __init__(self, config: Dict):
        self.config = config
        self.graph = nx.DiGraph()
        self.random_seed = config['seed']
        random.seed(self.random_seed)
        self.loaded_last_generation: int = -1
        self.log = logging.getLogger(__name__) # Renamed logger instance variable

    def create_graph(self):
        """Creates a graph based on the configuration."""
        gen_type = self.config['graph_generation']['type']
        params = self.config['graph_generation']['params']

        self.log.info(f"Creating graph of type '{gen_type}' with params: {params}")

        if gen_type == 'example': # Simple example for testing
             self._create_example_graph()
        elif gen_type == 'small_world':
             self._create_small_world_graph(**params)
        elif gen_type == 'small_worlds':
             self._create_multiple_small_worlds_graph(**params)
        else:
            raise ValueError(f"Unknown graph generation type: {gen_type}")

        self.log.info(f"Graph created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        # Initialize history for all nodes after creation
        for node_id in self.graph.nodes():
            data = self.get_node_data(node_id)
            if data and not data.history: # Should ideally be initialized correctly
                data.history = [data.current_value]


    def _create_example_graph(self):
        G = self.graph # Use the class's graph attribute
        n_nodes = 5
        for i in range(n_nodes):
            node_id = i + 1
            data = InfluenceNodeData(node_id=node_id, current_value=0, history=[0])
            G.add_node(node_id, data=data)

        G.add_edge(1, 2, weight=0.8)
        G.add_edge(2, 1, weight=0.2)
        G.add_edge(2, 3, weight=0.5)
        G.add_edge(3, 1, weight=0.4)
        G.add_edge(4, 3, weight=0.8)
        G.add_edge(3, 5, weight=0.6)
        if 4 not in G: G.add_node(4, data=InfluenceNodeData(node_id=4, current_value=0, history=[0]))
        if 5 not in G: G.add_node(5, data=InfluenceNodeData(node_id=5, current_value=0, history=[0]))

    def _create_small_world_graph(self, n: int, k: int, p: float, b: float, **kwargs):
        G = self.graph
        # Create graph topology
        undirected_G = nx.watts_strogatz_graph(n, k, p, seed=self.random_seed)

        # Add nodes
        for i in range(n):
            data = InfluenceNodeData(node_id=i, current_value=0, history=[0])
            G.add_node(i, data=data)

        # Add edges
        for u, v in undirected_G.edges():
            weight = round(random.uniform(0.1, 1.0), 2)
            G.add_edge(u, v, weight=weight)
            if random.random() < b:
                bidir_weight = round(random.uniform(0.1, 1.0), 2)
                G.add_edge(v, u, weight=bidir_weight)

    def _create_multiple_small_worlds_graph(self, n: int, k: int, p: float, b: float, g: int, inter_p: float, **kwargs):
        G = self.graph
        total_nodes = n * g

        # Create nodes and intra-group edges
        for group_idx in range(g):
            undirected_G = nx.watts_strogatz_graph(n, k, p, seed=self.random_seed)
            node_offset = group_idx * n

            for i in range(n):
                global_id = node_offset + i
                data = InfluenceNodeData(node_id=global_id, current_value=0, history=[0], group=group_idx)
                G.add_node(global_id, data=data)

            # Add intra-group edges
            for u_local, v_local in undirected_G.edges():
                 u_global = node_offset + u_local
                 v_global = node_offset + v_local
                 if u_global in G and v_global in G:
                      weight = round(random.uniform(0.2, 0.5), 2)
                      G.add_edge(u_global, v_global, weight=weight)
                      if random.random() < b:
                          bidir_weight = round(random.uniform(0.2, 0.5), 2)
                          G.add_edge(v_global, u_global, weight=bidir_weight)

        # Add inter-group edges
        for i in range(g):
            for j in range(i + 1, g):
                nodes_i = list(range(i * n, (i + 1) * n))
                nodes_j = list(range(j * n, (j + 1) * n))

                valid_nodes_i = [node for node in nodes_i if node in G]
                valid_nodes_j = [node for node in nodes_j if node in G]
                if not valid_nodes_i or not valid_nodes_j: continue

                for u in valid_nodes_i:
                    if random.random() < inter_p:
                        v = random.choice(valid_nodes_j)
                        weight = round(random.uniform(0.1, 0.5), 2)
                        G.add_edge(u, v, weight=weight)

                for v in valid_nodes_j:
                    if random.random() < inter_p:
                        u = random.choice(valid_nodes_i)
                        weight = round(random.uniform(0.1, 0.5), 2)
                        G.add_edge(v, u, weight=weight)

    def save_graph(self, filename: Optional[str] = None, last_completed_generation: int = -1):
        """Saves the graph state (with InfluenceNodeData) to a JSON file."""
        if filename is None:
            filename = self.config['paths']['graph_basename']
        save_dir = Path(self.config['paths']['graph_save_dir'])
        filepath = save_dir / f"{filename}.json"

        try:
            self.graph.graph['last_completed_generation'] = last_completed_generation
            self.log.info(f"Saving graph state completed up to generation index: {last_completed_generation}")

            serializable_data = json_graph.node_link_data(self.graph)

            # Convert InfluenceNodeData to dict for serialization
            for node_dict in serializable_data.get('nodes', []):
                if 'data' in node_dict and isinstance(node_dict['data'], InfluenceNodeData):
                    node_dict['data'] = asdict(node_dict['data'])

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=2)
            self.log.info(f"Graph saved successfully to {filepath}")

        except Exception as e:
            self.log.error(f"Failed to save graph to {filepath}: {e}", exc_info=True)

    def load_graph(self, filename: Optional[str] = None):
        """Loads the graph state (with InfluenceNodeData) from a JSON file."""
        self.loaded_last_generation = -1
        base_filename = filename if filename is not None else self.config['paths']['graph_basename']
        load_dir = Path(self.config['paths']['graph_save_dir'])
        graph_filepath = load_dir / f"{base_filename}.json"

        try:
            with open(graph_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert dictionaries back to InfluenceNodeData objects
            for node_data_dict in data.get('nodes', []):
                if 'data' in node_data_dict and isinstance(node_data_dict['data'], dict):
                    # Recreate InfluenceNodeData
                    node_id = node_data_dict.get('id')
                    if node_id is None:
                         self.log.warning("Node dictionary missing 'id' during load.")
                         continue

                    # Use get with defaults for InfluenceNodeData fields
                    influence_data_args = {
                        'node_id': node_id,
                        'current_value': node_data_dict['data'].get('current_value', 0),
                        'history': node_data_dict['data'].get('history', [0]), # Ensure history exists
                        'group': node_data_dict['data'].get('group', None)
                    }
                    # Ensure history is not empty if loaded
                    if not influence_data_args['history']:
                        influence_data_args['history'] = [influence_data_args['current_value']]

                    node_data_dict['data'] = InfluenceNodeData(**influence_data_args)

            self.graph = json_graph.node_link_graph(data, directed=True, multigraph=False)
            self.loaded_last_generation = self.graph.graph.get('last_completed_generation', -1)
            self.log.info(f"Graph loaded successfully from {graph_filepath}")
            self.log.info(f"Loaded graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
            self.log.info(f"Loaded graph metadata indicates completion up to generation index: {self.loaded_last_generation}")

        except FileNotFoundError:
            self.log.error(f"Graph file not found: {graph_filepath}")
            raise
        except json.JSONDecodeError:
            self.log.error(f"Error decoding JSON from graph file: {graph_filepath}")
            raise
        except Exception as e:
            self.log.error(f"Failed to load graph from {graph_filepath}: {e}")
            raise

    # --- Graph Data Access Methods ---

    def get_node_data(self, node_id: Any) -> Optional[InfluenceNodeData]:
        """Retrieves the InfluenceNodeData for a given node."""
        if node_id in self.graph:
            return self.graph.nodes[node_id].get('data')
        self.log.warning(f"Attempted to get data for non-existent node: {node_id}")
        return None

    def get_all_node_ids(self) -> List[Any]:
        """Returns a list of all node IDs."""
        return list(self.graph.nodes)

    def get_neighbors(self, node_id: Any) -> List[Any]:
        """Returns the list of successor nodes (neighbors this node influences)."""
        if node_id in self.graph:
            return list(self.graph.successors(node_id))
        return []

    def get_edge_weight(self, u: Any, v: Any) -> float:
        """Returns the weight of the edge from u to v."""
        if self.graph.has_edge(u, v):
            return self.graph[u][v].get('weight', 0.0)
        return 0.0

    def update_node_history(self, node_id: Any):
        """Appends the current value of the node to its history."""
        node_data = self.get_node_data(node_id)
        if node_data:
            # Ensure history is initialized if somehow missed
            if not node_data.history:
                node_data.history = [node_data.current_value]
            else:
                node_data.history.append(node_data.current_value)
        else:
            self.log.warning(f"Attempted to update history for non-existent node: {node_id}")

    def increment_node_value(self, node_id: Any):
        """Increments the current value of the node by 1."""
        node_data = self.get_node_data(node_id)
        if node_data:
            node_data.current_value += 1
        else:
            self.log.warning(f"Attempted to increment value for non-existent node: {node_id}")


# --- Simulation Runner ---

class SimulationRunner:
    """Runs the structural influence simulation."""

    def __init__(self, graph_manager: GraphManagerStructural, config: Dict):
        self.graph_manager = graph_manager
        self.config = config['simulation']
        random.seed(config['seed'])
        self.log = logging.getLogger(__name__) # Renamed logger instance variable

    def step(self, generation_index: int):
        """Performs one generation step: propagate influence and update counters."""
        self.log.debug(f"Generation {generation_index}: Propagating influence...")
        influenced_nodes_this_generation: Set[Any] = set()
        propagation_attempts = 0
        successful_propagations = 0

        # Propagation phase
        for source_node_id in self.graph_manager.get_all_node_ids():
            neighbors = self.graph_manager.get_neighbors(source_node_id)
            for target_node_id in neighbors:
                propagation_attempts += 1
                weight = self.graph_manager.get_edge_weight(source_node_id, target_node_id)
                if random.random() < weight:
                    influenced_nodes_this_generation.add(target_node_id)
                    successful_propagations += 1

        self.log.debug(f"Generation {generation_index}: {successful_propagations}/{propagation_attempts} successful propagations influencing {len(influenced_nodes_this_generation)} unique nodes.")

        # Update phase: Increment counter only once if influenced
        for node_id in influenced_nodes_this_generation:
            self.graph_manager.increment_node_value(node_id)

        # History update phase: Record the value *after* all increments for this generation
        for node_id in self.graph_manager.get_all_node_ids():
            self.graph_manager.update_node_history(node_id)

        self.log.debug(f"Generation {generation_index}: Values and history updated.")

    def run_simulation(self, start_generation_index: int = 0):
        """Runs the simulation loop."""
        num_generations_to_run = self.config['generations']
        end_generation_index = start_generation_index + num_generations_to_run
        self.log.info(f"Starting structural influence simulation run from generation {start_generation_index} up to {end_generation_index-1}.")

        last_completed_generation_index = start_generation_index - 1
        for current_gen_index in range(start_generation_index, end_generation_index):
            self.log.info(f"--- Starting Generation {current_gen_index} ---")
            try:
                self.step(current_gen_index)
                last_completed_generation_index = current_gen_index
                self.log.info(f"--- Finished Generation {current_gen_index} ---")
                yield last_completed_generation_index # Yield for potential intermediate processing/saving

            except Exception as e:
                self.log.error(f"Error during generation {current_gen_index}: {e}", exc_info=True)
                self.log.warning("Simulation stopped due to error.")
                break

        self.log.info(f"Structural simulation loop finished after generation {last_completed_generation_index}.")
        return last_completed_generation_index


# --- Logging Setup ---
def setup_logging(config):
    log_level_str = config['logging']['level'].upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = config['paths']['log_file']

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete.")
    return logger


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # Load Configuration (using structural defaults/overrides)
    config = load_config("config.yaml")

    # Setup Logging
    main_log = setup_logging(config) # Renamed logger instance variable
    main_log.info("--- Starting Structural Influence Simulation ---")
    main_log.info(f"Configuration loaded (using structural defaults where applicable)")

    # Initialize Components
    graph_manager = None
    simulation_runner = None
    start_generation_index = 0

    # Create or Load Initial Graph
    graph_manager = GraphManagerStructural(config)
    graph_base_name = config['paths']['graph_basename']
    graph_load_path = Path(config['paths']['graph_save_dir']) / f"{graph_base_name}.json"

    if graph_load_path.exists():
        main_log.info(f"Attempting to load existing graph from {graph_load_path}")
        try:
            graph_manager.load_graph(graph_base_name)
            start_generation_index = graph_manager.loaded_last_generation + 1
            main_log.info(f"Resuming simulation from generation index {start_generation_index}")
        except Exception as e:
            main_log.error(f"Failed to load graph: {e}. Creating a new graph instead.")
            graph_manager.create_graph()
            start_generation_index = 0
    else:
        main_log.info("No existing graph found. Creating a new graph.")
        graph_manager.create_graph()
        start_generation_index = 0

    # Initialize Simulation Runner
    simulation_runner = SimulationRunner(graph_manager, config)

    # Run Simulation
    last_completed_gen_in_run = start_generation_index - 1
    try:
        main_log.info("Starting simulation loop...")
        num_generations_to_run_config = config['simulation']['generations']
        if num_generations_to_run_config <= 0:
             main_log.warning("Configured number of generations to run is <= 0. No simulation steps will execute.")
             simulation_generator = iter([])
        else:
             main_log.info(f"Running {num_generations_to_run_config} generations...")
             simulation_generator = simulation_runner.run_simulation(start_generation_index=start_generation_index)

        for completed_generation_index in simulation_generator:
             last_completed_gen_in_run = completed_generation_index
             # Optional: Add intermediate saving here if needed
             # if (completed_generation_index + 1) % 50 == 0: # Save every 50 generations
             #     graph_manager.save_graph(f"{graph_base_name}_gen{completed_generation_index+1}", last_completed_gen_in_run)


    except Exception as e:
        main_log.critical(f"Simulation failed during execution: {e}", exc_info=True)
        # Save state even if simulation fails mid-way
        if graph_manager:
            main_log.info("Attempting to save graph state after error...")
            try:
                graph_manager.save_graph(graph_base_name + "_error_state", last_completed_generation=last_completed_gen_in_run)
            except Exception as save_err:
                main_log.error(f"Could not save error state: {save_err}")
        sys.exit(1)

    # --- Final Actions ---
    main_log.info("Performing final actions...")

    # Save Final Graph State
    if graph_manager:
        main_log.info("Saving final graph state...")
        try:
            graph_manager.save_graph(graph_base_name, last_completed_generation=last_completed_gen_in_run)
        except Exception as final_save_err:
            main_log.error(f"Failed to save final graph state: {final_save_err}")

    end_time = time.time()
    total_time = end_time - start_time
    main_log.info(f"--- Structural Simulation Finished ---")
    main_log.info(f"Total execution time: {total_time:.2f} seconds")
    main_log.info(f"Final graph state saved in: {config['paths']['graph_save_dir']}")