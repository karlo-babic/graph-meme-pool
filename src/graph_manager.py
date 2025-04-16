import networkx as nx
import random
import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Any, Optional, Dict
from networkx.readwrite import json_graph

from data_structures import MemeNodeData, PropagationEvent

logger = logging.getLogger(__name__)

class GraphManager:
    """Manages the NetworkX graph, node data, persistence, and propagation history."""

    def __init__(self, config: Dict):
        self.config = config
        self.graph = nx.DiGraph()
        self.propagation_history: List[PropagationEvent] = []
        self.random_seed = config['seed']
        random.seed(self.random_seed)
        self.assignment_strategy = config['graph_generation']['initial_meme_assignment']
        self.loaded_last_generation: int = -1
        logger.info(f"Initial meme assignment strategy: '{self.assignment_strategy}'")

    def _load_initial_memes(self) -> List[str]:
        """Loads initial memes from the configured file."""
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
        except Exception as e:
            logger.error(f"Error reading meme file {meme_file}: {e}")
            raise

    def create_graph(self):
        """Creates a graph based on the configuration."""
        gen_type = self.config['graph_generation']['type']
        params = self.config['graph_generation']['params']
        initial_memes = self._load_initial_memes()

        logger.info(f"Creating graph of type '{gen_type}' with params: {params}")

        if gen_type == 'example': # Simple example for testing
             self._create_example_graph(initial_memes)
        elif gen_type == 'small_world':
             self._create_small_world_graph(initial_memes, **params)
        elif gen_type == 'small_worlds':
             self._create_multiple_small_worlds_graph(initial_memes, **params)
        else:
            raise ValueError(f"Unknown graph generation type: {gen_type}")

        logger.info(f"Graph created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _create_example_graph(self, memes: List[str]):
        G = self.graph # Use the class's graph attribute
        n_nodes = 5
        for i in range(n_nodes):
            meme = memes[i % len(memes)]
            node_id = i + 1
            data = MemeNodeData(node_id=node_id, current_meme=meme)
            G.add_node(node_id, data=data)

        G.add_edge(1, 2, weight=0.8)
        G.add_edge(2, 1, weight=0.2)
        G.add_edge(2, 3, weight=0.5)
        G.add_edge(3, 1, weight=0.4)
        G.add_edge(4, 3, weight=0.8)
        G.add_edge(3, 5, weight=0.6)
        # Add node 4, 5 if not added via edges (needed for isolated nodes too)
        if 4 not in G: G.add_node(4, data=MemeNodeData(node_id=4, current_meme=memes[3 % len(memes)]))
        if 5 not in G: G.add_node(5, data=MemeNodeData(node_id=5, current_meme=memes[4 % len(memes)]))


    def _get_meme_from_index(self, memes: List[str], meme_index: int) -> Optional[str]:
        """Safely retrieves a meme string from the list using an index."""
        if not memes:
            logger.error("Meme list is empty.")
            return None
        if 0 <= meme_index < len(memes):
            return memes[meme_index]
        else:
            logger.warning(f"Meme index {meme_index} out of range ({len(memes)} memes). Using index 0.")
            return memes[0]

    def _calculate_initial_meme_ids(self,
                                    num_nodes: int,
                                    num_initial_memes: int,
                                    strategy: str,
                                    num_groups: Optional[int] = None,
                                    nodes_per_group: Optional[int] = None) -> List[int]:
        """Calculates the list of meme indices for node initialization based on strategy,
           attempting to use all available initial memes."""

        if num_initial_memes <= 0:
            logger.error("Cannot calculate meme IDs: Number of initial memes is zero or less.")
            return [-1] * num_nodes # Return error indicator

        meme_ids = [-1] * num_nodes # Initialize with placeholder

        if strategy == 'random':
            logger.debug(f"Calculating 'random' (balanced/shuffled) assignment for {num_nodes} nodes using {num_initial_memes} memes.")

            # Create a list with approximately equal counts of each meme index
            ids_to_shuffle = []
            base_count = num_nodes // num_initial_memes
            remainder = num_nodes % num_initial_memes

            for i in range(num_initial_memes):
                ids_to_shuffle.extend([i] * base_count) # Add base count for each meme
            # Distribute the remainder among the first 'remainder' memes
            ids_to_shuffle.extend(range(remainder))

            # Shuffle the list using the instance's seed for reproducibility
            local_random = random.Random(self.random_seed)
            local_random.shuffle(ids_to_shuffle)
            meme_ids = ids_to_shuffle
            logger.debug(f"Shuffled meme IDs for random assignment: {meme_ids[:20]}...") # Log first few

        elif strategy == 'structured':
            if num_groups is not None and nodes_per_group is not None and num_groups > 0 and nodes_per_group > 0:
                # Multi-group structured assignment: Cycle through ALL memes based on group index
                logger.debug(f"Calculating 'structured' (by group, cycling all memes) assignment for {num_groups} groups.")
                for i in range(num_nodes):
                    group_idx = i // nodes_per_group
                    meme_ids[i] = group_idx % num_initial_memes # Cycle through all memes
            else:
                # Single-graph structured assignment: Cycle through ALL memes based on node index
                logger.debug(f"Calculating 'structured' (cycling all memes) assignment for {num_nodes} nodes.")
                meme_ids = [i % num_initial_memes for i in range(num_nodes)] # Cycle through all memes
        else:
             logger.warning(f"Unknown assignment strategy '{strategy}'. Defaulting to structured (cycling).")
             # Default logic depends on whether it's single or multi-graph context
             if num_groups is not None and nodes_per_group is not None:
                  # Default to multi-group structured (cycling all)
                   for i in range(num_nodes):
                       group_idx = i // nodes_per_group
                       meme_ids[i] = group_idx % num_initial_memes
             else:
                  # Default to single-graph structured (cycling all)
                  meme_ids = [i % num_initial_memes for i in range(num_nodes)]

        # Final length check
        if len(meme_ids) != num_nodes:
             logger.error(f"Calculated meme ID list length ({len(meme_ids)}) does not match number of nodes ({num_nodes}). Check logic.")
             # Handle error: maybe return error list or raise exception
             return [-1] * num_nodes

        return meme_ids
    
    def _create_small_world_graph(self, memes: List[str], n: int, k: int, p: float, b: float, **kwargs):
        G = self.graph
        num_initial_memes = len(memes)
        if num_initial_memes == 0:
             logger.critical("No initial memes loaded. Cannot create graph nodes properly.")
             return

        # Calculate meme IDs using the helper method
        meme_ids = self._calculate_initial_meme_ids(
            num_nodes=n,
            num_initial_memes=num_initial_memes,
            strategy=self.assignment_strategy
        )

        # Create graph topology
        undirected_G = nx.watts_strogatz_graph(n, k, p, seed=self.random_seed)

        # Add nodes with assigned memes
        for i in range(n):
            meme_index = meme_ids[i]
            if meme_index == -1: # Check for error from helper
                 logger.error(f"Skipping node {i} due to meme ID calculation error.")
                 continue
            meme = self._get_meme_from_index(memes, meme_index)
            if meme is None:
                 logger.error(f"Skipping node {i} due to meme retrieval error for index {meme_index}.")
                 continue
            data = MemeNodeData(node_id=i, current_meme=meme)
            G.add_node(i, data=data)

        # Add edges
        for u, v in undirected_G.edges():
            weight = round(random.uniform(0.1, 1.0), 2)
            G.add_edge(u, v, weight=weight)
            if random.random() < b:
                bidir_weight = round(random.uniform(0.1, 1.0), 2)
                G.add_edge(v, u, weight=bidir_weight)

    def _create_multiple_small_worlds_graph(self, memes: List[str], n: int, k: int, p: float, b: float, g: int, inter_p: float, **kwargs):
        G = self.graph
        num_initial_memes = len(memes)
        if num_initial_memes == 0:
             logger.critical("No initial memes loaded. Cannot create graph nodes properly.")
             return

        total_nodes = n * g

        # Calculate meme IDs using the helper method
        meme_ids = self._calculate_initial_meme_ids(
            num_nodes=total_nodes,
            num_initial_memes=num_initial_memes,
            strategy=self.assignment_strategy,
            num_groups=g,
            nodes_per_group=n
        )

        # Create nodes and intra-group edges
        for group_idx in range(g):
            # Generate topology specific to this group (optional, could be done once globally too)
            undirected_G = nx.watts_strogatz_graph(n, k, p, seed=self.random_seed)
            node_offset = group_idx * n

            for i in range(n):
                global_id = node_offset + i
                meme_index = meme_ids[global_id] # Get pre-assigned meme ID
                if meme_index == -1:
                    logger.error(f"Skipping node {global_id} due to meme ID calculation error.")
                    continue
                meme = self._get_meme_from_index(memes, meme_index)
                if meme is None:
                    logger.error(f"Skipping node {global_id} due to meme retrieval error for index {meme_index}.")
                    continue
                # Assign group info regardless of assignment strategy
                data = MemeNodeData(node_id=global_id, current_meme=meme, group=group_idx)
                G.add_node(global_id, data=data)

            # Add intra-group edges
            for u_local, v_local in undirected_G.edges():
                 u_global = node_offset + u_local
                 v_global = node_offset + v_local
                 # Check if nodes were actually added (in case of errors)
                 if u_global in G and v_global in G:
                      weight = round(random.uniform(0.2, 0.6), 2)
                      G.add_edge(u_global, v_global, weight=weight)
                      if random.random() < b:
                          bidir_weight = round(random.uniform(0.3, 0.95), 2)
                          G.add_edge(v_global, u_global, weight=bidir_weight)

        # Calculate and add inter-group edges based on global_ids...
        for i in range(g):
            for j in range(i + 1, g):
                nodes_i = list(range(i * n, (i + 1) * n))
                nodes_j = list(range(j * n, (j + 1) * n))

                # Filter out nodes that might not exist due to errors
                valid_nodes_i = [node for node in nodes_i if node in G]
                valid_nodes_j = [node for node in nodes_j if node in G]
                if not valid_nodes_i or not valid_nodes_j: continue # Skip if a group is empty

                for u in valid_nodes_i:
                    if random.random() < inter_p:
                        v = random.choice(valid_nodes_j)
                        weight = round(random.uniform(0.1, 0.5), 2)  # 0.05, 0.3
                        G.add_edge(u, v, weight=weight)

                for v in valid_nodes_j:
                    if random.random() < inter_p:
                        u = random.choice(valid_nodes_i)
                        weight = round(random.uniform(0.1, 0.5), 2)
                        G.add_edge(v, u, weight=weight)


    def save_graph(self, filename: Optional[str] = None, last_completed_generation: int = -1):
        """Saves the graph state to a JSON file."""
        if filename is None:
            filename = self.config['paths']['graph_basename']
        save_dir = Path(self.config['paths']['graph_save_dir'])
        filepath = save_dir / f"{filename}.json"

        try:
            # Store the last completed generation index directly in the graph attributes
            self.graph.graph['last_completed_generation'] = last_completed_generation
            logger.info(f"Saving graph state completed up to generation index: {last_completed_generation}")

            # Get node-link data structure
            serializable_data = json_graph.node_link_data(self.graph)

            # --- Conversion Step ---
            # Iterate through nodes and convert data attributes
            for node_dict in serializable_data.get('nodes', []):
                if 'data' in node_dict and isinstance(node_dict['data'], MemeNodeData):
                    # Convert dataclass to dict first
                    data_dict = node_dict['data'].__dict__
                    # Now iterate through the dict and convert numpy types
                    for key, value in data_dict.items():
                        if isinstance(value, list):
                            # Convert elements within lists
                            data_dict[key] = [
                                float(item) if isinstance(item, (np.float_, np.float16, np.float32, np.float64)) else
                                int(item) if isinstance(item, (np.int_, np.int8, np.int16, np.int32, np.int64)) else
                                item # Keep other types as is
                                for item in value
                            ]
                        elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                            data_dict[key] = float(value)
                        elif isinstance(value, (np.int_, np.int8, np.int16, np.int32, np.int64)):
                            data_dict[key] = int(value)
                        # Add other numpy types if needed (e.g., np.bool_)

                    # Replace original data object with the converted dict
                    node_dict['data'] = data_dict
                # Add similar conversion logic if other attributes outside 'data' could be numpy types
                # Example: Convert top-level node attributes if necessary
                # for key, value in node_dict.items():
                #     if isinstance(value, np.float32): node_dict[key] = float(value)
                #     elif isinstance(value, np.int64): node_dict[key] = int(value)
            # --- End Conversion Step ---

            # Save the modified data
            with open(filepath, "w", encoding="utf-8") as f:
                # Use default=str as a fallback, though explicit conversion is better
                json.dump(serializable_data, f, indent=2) # Removed default=str, rely on explicit conversion
            logger.info(f"Graph saved successfully to {filepath}")

        except TypeError as te:
             # Catch potential lingering TypeErrors during dump
             logger.error(f"JSON serialization failed after explicit conversion: {te}. There might be unhandled numpy types.", exc_info=True)
             # Optionally try again with default=str as a last resort.
             try:
                 with open(filepath, "w", encoding="utf-8") as f:
                     json.dump(serializable_data, f, indent=2, default=str)
                 logger.warning(f"Graph saved with default=str fallback after error.")
             except Exception as e:
                 logger.error(f"Failed to save graph even with fallback: {e}")

        except Exception as e:
            logger.error(f"Failed to save graph to {filepath}: {e}", exc_info=True)

    def load_graph(self, filename: Optional[str] = None):
        """Loads the graph state from a JSON file."""
        self.loaded_last_generation = -1 # Reset before loading
        base_filename = filename if filename is not None else self.config['paths']['graph_basename']
        load_dir = Path(self.config['paths']['graph_save_dir'])
        graph_filepath = load_dir / f"{base_filename}.json"

        try:
            with open(graph_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert dictionaries back to MemeNodeData objects
            for node_data_dict in data.get('nodes', []):
                if 'data' in node_data_dict and isinstance(node_data_dict['data'], dict):
                    # Recreate MemeNodeData, handling potential missing fields gracefully
                    node_id = node_data_dict.get('id') # Get id from the main node dict
                    if node_id is None:
                         logger.warning("Node dictionary missing 'id' during load.")
                         continue # Skip if no ID

                    # Use get with defaults for MemeNodeData fields
                    meme_data_args = {
                        'node_id': node_id,
                        'current_meme': node_data_dict['data'].get('current_meme', ''),
                        'history': node_data_dict['data'].get('history', []),
                        'history_scores': node_data_dict['data'].get('history_scores', []),
                        'current_meme_score': node_data_dict['data'].get('current_meme_score', None),
                        'received_memes': node_data_dict['data'].get('received_memes', []),
                        'group': node_data_dict['data'].get('group', None)
                    }
                    node_data_dict['data'] = MemeNodeData(**meme_data_args)

            self.graph = json_graph.node_link_graph(data, directed=True, multigraph=False)
            self.loaded_last_generation = self.graph.graph.get('last_completed_generation', -1)
            graph_loaded_successfully = True # Mark as successful
            logger.info(f"Graph loaded successfully from {graph_filepath}")
            logger.info(f"Loaded graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
            logger.info(f"Loaded graph metadata indicates completion up to generation index: {self.loaded_last_generation}")
        except FileNotFoundError:
            logger.error(f"Graph file not found: {graph_filepath}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from graph file: {graph_filepath}")
            raise
        except Exception as e:
            logger.error(f"Failed to load graph from {graph_filepath}: {e}")
            raise

        # Load propagation history
        if graph_loaded_successfully:
            history_filename = f"{base_filename}_propagation.json"
            history_filepath = load_dir / history_filename
            logger.info(f"Attempting to load propagation history from {history_filepath}")
            self.propagation_history = [] # Reset history before attempting load

            try:
                with open(history_filepath, "r", encoding="utf-8") as f:
                    history_data = json.load(f)

                if isinstance(history_data, list):
                     loaded_count = 0
                     skipped_count = 0
                     for event_dict in history_data:
                          if isinstance(event_dict, dict):
                               try:
                                    # Convert dict back to PropagationEvent dataclass
                                    event = PropagationEvent(**event_dict)
                                    self.propagation_history.append(event)
                                    loaded_count += 1
                               except TypeError as te:
                                    # Handles case where dict keys don't match dataclass fields
                                    logger.warning(f"Skipping history event due to mismatching data: {event_dict}. Error: {te}")
                                    skipped_count += 1
                          else:
                               logger.warning(f"Skipping non-dictionary item found in history file: {event_dict}")
                               skipped_count += 1

                     logger.info(f"Successfully loaded {loaded_count} propagation events. Skipped {skipped_count} invalid entries.")
                else:
                     logger.warning(f"Propagation history file {history_filepath} does not contain a list. Skipping history load.")
                     self.propagation_history = [] # Ensure it's empty

            except FileNotFoundError:
                 logger.warning(f"Propagation history file {history_filepath} not found. Starting with empty history.")
                 self.propagation_history = [] # Ensure it's empty
            except json.JSONDecodeError:
                 logger.error(f"Error decoding JSON from history file {history_filepath}. Starting with empty history.")
                 self.propagation_history = []
            except Exception as e:
                 logger.error(f"Failed to load or process propagation history from {history_filepath}: {e}", exc_info=True)
                 self.propagation_history = [] # Ensure clean state on error


    def save_propagation_history(self, filename: Optional[str] = None):
        """Saves the propagation history to a JSON file."""
        if not self.propagation_history:
            logger.info("No propagation history to save.")
            return

        if filename is None:
            filename = self.config['paths']['graph_basename'] + "_propagation"
        save_dir = Path(self.config['paths']['graph_save_dir'])
        filepath = save_dir / f"{filename}.json"

        try:
            # Convert PropagationEvent objects to dictionaries
            serializable_history = [event.__dict__ for event in self.propagation_history]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_history, f, indent=2)
            logger.info(f"Propagation history saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save propagation history to {filepath}: {e}")

    # --- Graph Data Access Methods ---

    def get_node_data(self, node_id: Any) -> Optional[MemeNodeData]:
        """Retrieves the MemeNodeData for a given node."""
        if node_id in self.graph:
            return self.graph.nodes[node_id].get('data')
        logger.warning(f"Attempted to get data for non-existent node: {node_id}")
        return None

    def get_all_node_ids(self) -> List[Any]:
        """Returns a list of all node IDs."""
        return list(self.graph.nodes)

    def get_all_nodes_data(self) -> Dict[Any, MemeNodeData]:
        """Retrieves MemeNodeData for all nodes."""
        all_data = {}
        for node_id, node_attrs in self.graph.nodes(data=True):
             data = node_attrs.get('data')
             if data:
                 all_data[node_id] = data
             else:
                 logger.warning(f"Node {node_id} is missing 'data' attribute.")
        return all_data


    def get_neighbors(self, node_id: Any) -> List[Any]:
        """Returns the list of successor nodes (neighbors this node influences)."""
        if node_id in self.graph:
            return list(self.graph.successors(node_id))
        return []

    def get_predecessors(self, node_id: Any) -> List[Any]:
         """Returns the list of predecessor nodes (neighbors influencing this node)."""
         if node_id in self.graph:
             return list(self.graph.predecessors(node_id))
         return []

    def get_edge_weight(self, u: Any, v: Any) -> float:
        """Returns the weight of the edge from u to v."""
        if self.graph.has_edge(u, v):
            return self.graph[u][v].get('weight', 0.0)
        return 0.0

    def update_node_meme(self, node_id: Any, new_meme: str, new_score: Optional[float], generation: int):
        """Updates the current meme, score, and history for a node."""
        node_data = self.get_node_data(node_id)
        if node_data:
            node_data.current_meme = new_meme
            node_data.current_meme_score = new_score
            node_data.history.append(new_meme)
            # Ensure history_scores aligns with history
            score_to_add = new_score if new_score is not None else None
            # Pad previous scores if needed (e.g., if initial scoring was skipped)
            if len(node_data.history_scores) < len(node_data.history) - 1:
                padding = [None] * (len(node_data.history) - 1 - len(node_data.history_scores))
                node_data.history_scores.extend(padding)
            node_data.history_scores.append(score_to_add)

            # Optionally log the update
            # logger.debug(f"Gen {generation}: Node {node_id} updated meme to '{new_meme[:30]}...' (Score: {new_score})")
        else:
            logger.warning(f"Attempted to update non-existent node: {node_id}")

    def update_node_score(self, node_id: Any, score: float, history_index: int = -1):
         """Updates the score of a meme (current or historical) for a node."""
         node_data = self.get_node_data(node_id)
         if node_data:
              if history_index == -1 or history_index >= len(node_data.history_scores):
                   node_data.current_meme_score = score
                   # Ensure the latest score in history is also updated
                   if node_data.history_scores:
                        node_data.history_scores[-1] = score
                   # logger.debug(f"Node {node_id}: Updated current score to {score}")
              else:
                   node_data.history_scores[history_index] = score
                   # logger.debug(f"Node {node_id}: Updated historical score at index {history_index} to {score}")
         else:
             logger.warning(f"Attempted to update score for non-existent node: {node_id}")


    def add_received_meme(self, sender_node_id: Any, node_id: Any, meme: str, weight: float):
        """Adds a meme received by a node in the current step."""
        node_data = self.get_node_data(node_id)
        if node_data:
            node_data.received_memes.append((sender_node_id, meme, weight))
        else:
            logger.warning(f"Attempted to add received meme to non-existent node: {node_id}")

    def clear_received_memes(self, node_id: Optional[Any] = None):
        """Clears the list of received memes for one or all nodes."""
        if node_id is not None:
            node_data = self.get_node_data(node_id)
            if node_data:
                node_data.received_memes = []
        else: # Clear for all nodes
            for node_id in self.get_all_node_ids():
                 node_data = self.get_node_data(node_id)
                 if node_data:
                      node_data.received_memes = []

    # --- Propagation Tracking ---

    def record_propagation(self, generation: int, source: Any, target: Any, meme: str):
        """Records a meme propagation event."""
        event = PropagationEvent(
            generation=generation,
            source_node=source,
            target_node=target,
            meme=meme
        )
        self.propagation_history.append(event)

    def get_propagation_history(self) -> List[PropagationEvent]:
        """Returns the recorded propagation history."""
        return self.propagation_history