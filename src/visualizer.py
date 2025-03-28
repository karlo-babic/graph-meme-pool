import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from graph_manager import GraphManager
from data_structures import MemeNodeData
# Import embedding utils if needed, or receive model/functions
import embeddings_utils as emb_utils

logger = logging.getLogger(__name__)

# Set random seed for layout consistency if desired (can be configured)
# np.random.seed(2)

class Visualizer:
    """Handles generation of all graph visualizations."""

    def __init__(self, graph_manager: GraphManager, config: Dict):
        self.graph_manager = graph_manager
        self.vis_config = config['visualization']
        self.path_config = config['paths']
        self.embed_config = config['embeddings']
        self.vis_dir = Path(self.path_config['vis_dir'])
        # Consider loading the embedding model once here or receiving it
        self.sentence_model = None
        if self.vis_config.get('draw_semantic_diff_per_gen', False) or \
           self.vis_config.get('draw_final_embs', False): # Load if any embedding plot is enabled
             try:
                  # Using sentence transformer for both semantic diff and general embs now
                  self.sentence_model = emb_utils.get_sentence_transformer_model(self.embed_config['sentence_model_path'])
             except Exception as e:
                  logger.error(f"Failed to load embedding model for visualization: {e}. Embedding plots will be skipped.")
                  self.sentence_model = None


    def _get_layout(self, G: nx.DiGraph, layout_type='kamada_kawai') -> Dict:
        """Computes graph layout."""
        logger.debug(f"Computing graph layout using {layout_type}...")
        try:
            if layout_type == 'kamada_kawai':
                # Kamada-Kawai can fail on disconnected graphs sometimes
                 if nx.is_connected(G.to_undirected()):
                      pos = nx.kamada_kawai_layout(G)
                 else:
                      logger.warning("Graph is not connected, Kamada-Kawai might be slow or fail. Trying spring layout as fallback.")
                      pos = nx.spring_layout(G, seed=42, k=1.0 / np.sqrt(G.number_of_nodes())) # Adjust k
            elif layout_type == 'spring':
                 pos = nx.spring_layout(G, seed=42, k=1.0 / np.sqrt(G.number_of_nodes())) # Adjust k
            elif layout_type == 'fruchterman_reingold':
                 pos = nx.fruchterman_reingold_layout(G, seed=42)
            else: # Default to spring
                 pos = nx.spring_layout(G, seed=42)
            logger.debug("Layout computation finished.")
            return pos
        except Exception as e:
            logger.warning(f"Layout computation failed ('{layout_type}'): {e}. Falling back to spring layout.")
            return nx.spring_layout(G, seed=42)


    def _calculate_influence(self, G: nx.DiGraph) -> Tuple[Dict[Any, float], Dict[Any, float]]:
        """Calculate and normalize outgoing (influence) and incoming (received) edge weight sums."""
        influences = {}
        received_influences = {}

        for node in G.nodes():
            influences[node] = sum(G[u][v].get('weight', 0.0) for u, v in G.out_edges(node))
            received_influences[node] = sum(G[u][v].get('weight', 0.0) for u, v in G.in_edges(node))

        def normalize_dict(d):
            min_val = min(d.values()) if d else 0
            max_val = max(d.values()) if d else 0
            range_val = max_val - min_val
            if range_val == 0:
                return {k: 0.5 for k in d} # Avoid division by zero, return neutral value
            return {k: (v - min_val) / range_val for k, v in d.items()}

        return normalize_dict(influences), normalize_dict(received_influences)


    def _base_draw(self, G: nx.DiGraph, pos: Dict, node_colors: List, node_sizes: List, filename: str, title: Optional[str] = None):
        """Common drawing logic."""
        plt.figure(figsize=(15, 15))

        edge_weights = [G[u][v].get('weight', 0.0) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1.0
        if max_weight == 0: max_weight = 1.0 # Avoid division by zero

        edge_thickness = [self.vis_config['edge_base_thickness'] * (w / max_weight) for w in edge_weights]
        edge_opacity = [self.vis_config['edge_base_opacity'] * (w / max_weight) for w in edge_weights]

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.coolwarm) # Apply colormap here

        nx.draw_networkx_edges(G, pos,
                               edgelist=list(G.edges()), # Ensure it's a list
                               width=edge_thickness,
                               alpha=edge_opacity,
                               edge_color="black",
                               arrows=True,
                               arrowstyle='-|>', # Standard arrow
                               arrowsize=10) # Adjust size as needed

        # Labels (show current meme, truncated)
        max_len = self.vis_config['label_max_len']
        node_labels = {}
        for node_id, data in G.nodes(data=True):
            node_data: Optional[MemeNodeData] = data.get('data')
            if node_data:
                 text = node_data.current_meme
                 label = text[:max_len] + "..." if len(text) > max_len else text
                 node_labels[node_id] = label
            else:
                 node_labels[node_id] = str(node_id) # Fallback to ID

        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7) # Slightly larger font

        if title:
            plt.title(title, fontsize=16)

        plt.axis('off') # Hide axes
        plt.tight_layout()
        filepath = Path(filename) # Ensure filename is treated as a Path object
        try:
            plt.savefig(filepath, bbox_inches='tight', dpi=self.vis_config['dpi'])
            logger.info(f"Saved visualization to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save visualization {filepath}: {e}")
        plt.close()


    def draw_score(self, generation: Optional[int] = None):
        """Visualizes the graph with nodes colored by current meme score."""
        G = self.graph_manager.graph
        if G.number_of_nodes() == 0: return
        pos = self._get_layout(G)
        norm_influences, _ = self._calculate_influence(G)

        scores = []
        node_ids_ordered = list(G.nodes()) # Ensure consistent order
        for node_id in node_ids_ordered:
            data = self.graph_manager.get_node_data(node_id)
            scores.append(data.current_meme_score if data and data.current_meme_score is not None else 0.5) # Default to neutral 0.5

        # Normalize scores to 0-1 for coloring
        min_s, max_s = min(scores), max(scores)
        color_values = [(s - min_s) / (max_s - min_s) if (max_s - min_s) > 0 else 0.5 for s in scores]

        node_sizes = [self.vis_config['node_min_size'] + (self.vis_config['node_max_size'] - self.vis_config['node_min_size']) * norm_influences.get(nid, 0.5)
                      for nid in node_ids_ordered]

        base_filename = f"graph_score_gen_{generation}.png" if generation is not None else "graph_score_final.png"
        filepath = self.vis_dir / base_filename # Construct full path here
        title = f"Meme Scores (Generation {generation})" if generation is not None else "Final Meme Scores"
        # Pass the full filepath to _base_draw
        self._base_draw(G, pos, color_values, node_sizes, filepath, title)


    def draw_change(self, generation: int, history_lookback: int = 4):
        """Visualizes node color based on recent meme uniqueness/change."""
        G = self.graph_manager.graph
        if G.number_of_nodes() == 0: return
        pos = self._get_layout(G)
        norm_influences, _ = self._calculate_influence(G)

        unique_counts = []
        node_ids_ordered = list(G.nodes())
        max_unique = 1 # Avoid division by zero

        for node_id in node_ids_ordered:
             data = self.graph_manager.get_node_data(node_id)
             if data:
                  # Look at the last 'history_lookback' entries, or fewer if history is short
                  start_index = max(0, len(data.history) - history_lookback)
                  recent_history = data.history[start_index:]
                  count = len(set(recent_history))
                  unique_counts.append(count)
                  if count > max_unique:
                      max_unique = count
             else:
                  unique_counts.append(1) # Default for missing data

        # Normalize counts (0 to 1, where 1 means max changes)
        color_values = [c / max_unique for c in unique_counts]

        node_sizes = [self.vis_config['node_min_size'] + (self.vis_config['node_max_size'] - self.vis_config['node_min_size']) * norm_influences.get(nid, 0.5)
                      for nid in node_ids_ordered]

        base_filename = f"graph_change_gen_{generation}.png"
        filepath = self.vis_dir / base_filename # Construct full path here
        title = f"Recent Meme Change (Gen {generation}, Lookback {history_lookback})"
        # Pass the full filepath to _base_draw
        self._base_draw(G, pos, color_values, node_sizes, filepath, title)


    def draw_embs(self):
        """Visualizes graph with layout based on semantic embeddings of current memes."""
        if not self.sentence_model:
             logger.warning("Skipping draw_embs: Embedding model not available.")
             return

        G = self.graph_manager.graph
        if G.number_of_nodes() == 0: return
        norm_influences, _ = self._calculate_influence(G)

        node_ids_ordered = list(G.nodes())
        texts = []
        groups = [] # For coloring
        node_id_map = {} # Map list index back to node_id

        for i, node_id in enumerate(node_ids_ordered):
             data = self.graph_manager.get_node_data(node_id)
             if data:
                  texts.append(data.current_meme)
                  groups.append(data.group if data.group is not None else 0)
                  node_id_map[i] = node_id
             else:
                  texts.append("") # Empty string for missing data
                  groups.append(0)
                  node_id_map[i] = node_id

        # Calculate embeddings
        embeddings_np = emb_utils.calculate_sentence_embeddings(texts, self.sentence_model)
        if embeddings_np.size == 0:
            logger.error("Failed to calculate embeddings for draw_embs. Skipping plot.")
            return

        # Create dict for reduction
        embeddings_dict = {node_id_map[i]: emb for i, emb in enumerate(embeddings_np)}

        # Reduce dimensions
        embeddings_2d = emb_utils.reduce_dimensions_tsne(embeddings_dict, random_state=42)
        if not embeddings_2d:
            logger.error("Failed to reduce embedding dimensions for draw_embs. Skipping plot.")
            return

        # Use reduced embeddings as layout positions
        pos = embeddings_2d

        # Colors based on group
        unique_groups = sorted(list(set(groups)))
        num_groups = len(unique_groups)
        cmap = plt.cm.get_cmap('viridis', num_groups) # Or 'tab10', 'tab20'
        group_to_color_val = {group: i / (num_groups - 1) if num_groups > 1 else 0.5 for i, group in enumerate(unique_groups)}
        node_color_values = [group_to_color_val[groups[i]] for i in range(len(node_ids_ordered))]


        node_sizes = [self.vis_config['node_min_size'] + (self.vis_config['node_max_size'] - self.vis_config['node_min_size']) * norm_influences.get(nid, 0.5)
                      for nid in node_ids_ordered]

        base_filename = "graph_embeddings_final.png"
        filepath = self.vis_dir / base_filename # Construct full path here
        title = "Graph Layout by Semantic Embeddings (Final State)"
        # Pass the full filepath to _base_draw
        self._base_draw(G, pos, node_color_values, node_sizes, filepath, title)


    def draw_semantic_difference(self, generation: int):
        """Visualizes semantic shift relative to two reference points."""
        if not self.sentence_model:
             logger.warning("Skipping draw_semantic_difference: Embedding model not available.")
             return

        G = self.graph_manager.graph
        if G.number_of_nodes() == 0: return
        pos = self._get_layout(G) # Use standard layout
        norm_influences, _ = self._calculate_influence(G)

        # Define reference points (Consider making these configurable)
        reference_A = "The animal moved because its brain sent a signal to its muscles."
        reference_B = "The animal moved because it was carried by a flood."
        try:
             embedding_A = emb_utils.calculate_sentence_embeddings([reference_A], self.sentence_model)[0]
             embedding_B = emb_utils.calculate_sentence_embeddings([reference_B], self.sentence_model)[0]
        except Exception as e:
             logger.error(f"Failed to embed reference texts for semantic difference: {e}. Skipping plot.")
             return

        semantic_diff_ratios = []
        node_ids_ordered = list(G.nodes())

        current_texts = []
        valid_node_indices = [] # Track indices corresponding to nodes with valid data
        for i, node_id in enumerate(node_ids_ordered):
             data = self.graph_manager.get_node_data(node_id)
             # Check if history is long enough for the requested generation
             if data and len(data.history) > generation:
                  current_texts.append(data.history[generation])
                  valid_node_indices.append(i)
             # else: implicitly skip this node for semantic diff calculation

        if not current_texts:
             logger.warning(f"No valid node data found for generation {generation} in semantic difference plot.")
             # Draw an empty plot or skip? Skipping for now.
             return

        # Batch embed current texts
        current_embeddings = emb_utils.calculate_sentence_embeddings(current_texts, self.sentence_model)
        if current_embeddings.size == 0:
             logger.error(f"Failed to calculate embeddings for generation {generation} in semantic difference plot.")
             return

        # Calculate similarity and ratio for valid nodes
        diff_values_map = {} # node_id -> ratio
        for i, emb_current in enumerate(current_embeddings):
             sim_A = emb_utils.calculate_cosine_similarity(emb_current, embedding_A)
             sim_B = emb_utils.calculate_cosine_similarity(emb_current, embedding_B)
             # Ratio: 0 means close to A, 1 means close to B
             ratio = sim_B / (sim_A + sim_B) if (sim_A + sim_B) > 1e-6 else 0.5 # Avoid division by zero
             original_node_index = valid_node_indices[i]
             node_id = node_ids_ordered[original_node_index]
             diff_values_map[node_id] = ratio

        # Prepare color values for all nodes, using 0.5 for skipped nodes
        color_values = [diff_values_map.get(node_id, 0.5) for node_id in node_ids_ordered]

        node_sizes = [self.vis_config['node_min_size'] + (self.vis_config['node_max_size'] - self.vis_config['node_min_size']) * norm_influences.get(nid, 0.5)
                      for nid in node_ids_ordered]

        # Save to subdirectory
        sem_diff_dir = self.vis_dir / "graph_semantic_difference"
        sem_diff_dir.mkdir(exist_ok=True)
        filename = sem_diff_dir / f"gen_{generation:04d}.png" # Use padding for sorting
        title = f"Semantic Difference (Gen {generation}, Blue=RefA, Red=RefB)"

        # Use _base_draw but provide full path
        self._base_draw(G, pos, color_values, node_sizes, filename, title)


    def plot_score_history_bygroup(self):
        """Plots the average meme score per generation for each group."""
        G = self.graph_manager.graph
        if G.number_of_nodes() == 0: return

        group_histories: Dict[int, List[List[Optional[float]]]] = {}
        max_len = 0

        for node_id, data in G.nodes(data=True):
            node_data: Optional[MemeNodeData] = data.get('data')
            if node_data and node_data.history_scores:
                group = node_data.group if node_data.group is not None else 0
                scores = node_data.history_scores # Already includes Nones potentially

                if group not in group_histories:
                    group_histories[group] = []
                group_histories[group].append(scores)
                if len(scores) > max_len:
                    max_len = len(scores)

        if not group_histories:
            logger.warning("No score history found to plot.")
            return

        # Compute average score per time step for each group
        avg_scores: Dict[int, np.ndarray] = {}
        for group, histories in group_histories.items():
            # Pad histories to max_len using NaN
            histories_padded = []
            for h in histories:
                 padding_needed = max_len - len(h)
                 padded_history = h + ([np.nan] * padding_needed) if padding_needed > 0 else h[:max_len]
                 histories_padded.append(padded_history)

            # Calculate mean ignoring NaNs
            try:
                 with np.errstate(invalid='ignore'): # Suppress mean of empty slice warning if a step has all NaNs
                     avg_scores[group] = np.nanmean(np.array(histories_padded, dtype=float), axis=0)
            except Exception as e:
                 logger.error(f"Error calculating average scores for group {group}: {e}")
                 continue # Skip this group

        # Plotting
        plt.figure(figsize=(12, 7))
        num_groups = len(avg_scores)
        colors = plt.cm.get_cmap("viridis", num_groups)

        for i, (group, avg_history) in enumerate(sorted(avg_scores.items())): # Sort by group ID
             # Find first non-NaN index to start plotting from
             first_valid_index = np.where(~np.isnan(avg_history))[0]
             if len(first_valid_index) > 0:
                  start_index = first_valid_index[0]
                  plt.plot(range(start_index, len(avg_history)), avg_history[start_index:],
                           color=colors(i / (num_groups-1) if num_groups > 1 else 0.5), # Adjust color indexing
                           label=f"Group {group}", linewidth=2)
             else:
                  logger.warning(f"Group {group} has no valid score data to plot.")


        plt.xlabel("Generation")
        plt.ylabel("Average Meme Score (Virality)")
        plt.title("Average Meme Score per Generation by Group")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1.05) # Scores are normalized 0-1

        filename = self.vis_dir / "plot_score_history_bygroup.png"
        try:
            plt.savefig(filename, bbox_inches='tight', dpi=self.vis_config['dpi'])
            logger.info(f"Saved score history plot to {filename}")
        except Exception as e:
            logger.error(f"Failed to save score history plot {filename}: {e}")
        plt.close()