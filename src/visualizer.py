import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List

from PIL import Image

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
        self.embedding_model = None
        if self.vis_config['draw_semantic_diff_per_gen'] or \
           self.vis_config['draw_final_embs']: # Load if any embedding plot is enabled
             try:
                  self.embedding_model = emb_utils.get_sentence_transformer_model(self.embed_config['model_path'])
             except Exception as e:
                  logger.error(f"Failed to load embedding model for visualization: {e}. Embedding plots will be skipped.")
                  self.embedding_model = None


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
        if not self.embedding_model:
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
        embeddings_np = emb_utils.calculate_sentence_embeddings(texts, self.embedding_model)
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
        if not self.embedding_model:
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
             embedding_A = emb_utils.calculate_sentence_embeddings([reference_A], self.embedding_model)[0]
             embedding_B = emb_utils.calculate_sentence_embeddings([reference_B], self.embedding_model)[0]
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
        current_embeddings = emb_utils.calculate_sentence_embeddings(current_texts, self.embedding_model)
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




    def draw_semantic_drift(self, num_generations: int = -1, visible_groups: Optional[List[Any]] = None):
        """
        Visualizes semantic drift of memes per node over time using embedding space trajectories.

        Args:
            num_generations (int): The maximum number of generations (history steps) to include.
                                   Defaults to -1 (include all).
            visible_groups (Optional[List[Any]]): A list of group identifiers for which to display
                                                  trajectories. If None (default), trajectories for
                                                  all groups are displayed. Nodes from *all* groups
                                                  are still used for embedding calculation and
                                                  dimensionality reduction.
        """
        if not self.embedding_model:
            logger.warning("Skipping draw_semantic_drift: Embedding model not available.")
            return

        G = self.graph_manager.graph
        if G.number_of_nodes() == 0:
            logger.warning("Skipping draw_semantic_drift: Graph has no nodes.")
            return

        # Collect all meme histories and associated metadata
        all_texts = []
        text_index_map = []  # List of (node_id, timestep)
        node_groups = {}     # node_id -> group

        for node_id in G.nodes():
            # Use a safer way to access node data if structure varies
            data = self.graph_manager.get_node_data(node_id)
            # Check if data exists and has 'history' attribute
            if hasattr(data, 'history') and data.history:
                group = getattr(data, 'group', 0) # Default group 0 if not present
                node_groups[node_id] = group
                for t, meme in enumerate(data.history):
                    if num_generations >= 0 and t >= num_generations:
                        break
                    all_texts.append(str(meme)) # Ensure text is string
                    text_index_map.append((node_id, t))

        if not all_texts:
             logger.warning("Skipping draw_semantic_drift: No meme history found in any node.")
             return

        # Compute all embeddings
        logger.info(f"Calculating embeddings for {len(all_texts)} memes...")
        all_embeddings = emb_utils.calculate_sentence_embeddings(all_texts, self.embedding_model)
        if all_embeddings is None or all_embeddings.size == 0:
            logger.error("Failed to calculate embeddings for draw_semantic_drift. Skipping plot.")
            return

        # Reduce to 2D using t-SNE
        logger.info(f"Reducing dimensionality for {len(all_texts)} memes...")
        # Ensure embeddings_dict keys match indices expected by text_index_map
        embeddings_dict = {i: emb for i, emb in enumerate(all_embeddings)}
        reduced_2d = emb_utils.reduce_dimensions_tsne(embeddings_dict, random_state=42)
        if not reduced_2d:
            logger.error("Failed to reduce embedding dimensions for draw_semantic_drift. Skipping plot.")
            return

        # Organize by node for plotting
        node_trails = defaultdict(list)      # node_id -> list of (x, y)
        node_timesteps = defaultdict(list)   # node_id -> list of timestep (used for alpha)
        for i, (node_id, timestep) in enumerate(text_index_map):
            if i in reduced_2d:
                node_trails[node_id].append(reduced_2d[i])
                node_timesteps[node_id].append(timestep)

        # Set up colormap for clusters
        unique_groups = sorted(list(set(node_groups.values()))) # Use list() for safety
        if not unique_groups:
             logger.warning("Skipping draw_semantic_drift: No groups found for nodes.")
             return
        cmap = plt.cm.get_cmap('tab10', len(unique_groups))
        group_to_color = {g: cmap(i) for i, g in enumerate(unique_groups)}

        # Convert visible_groups to a set for efficient lookup, if provided
        visible_groups_set = set(visible_groups) if visible_groups is not None else None

        # Plotting
        logger.info(f"Plotting...")
        plt.figure(figsize=(15, 15))
        nodes_plotted = 0
        for node_id, trail in node_trails.items():
            group = node_groups.get(node_id, 0) # Get group, default 0

            # Check if this group should be visible
            if visible_groups_set is None or group in visible_groups_set:
                nodes_plotted += 1
                color = group_to_color.get(group, (0.5, 0.5, 0.5)) # Default color grey

                timesteps = node_timesteps[node_id]
                n = len(trail)
                max_time = max(timesteps) if timesteps else 0

                # Plot lines connecting consecutive points
                for i in range(n - 1):
                    x1, y1 = trail[i]
                    x2, y2 = trail[i + 1]
                    # Alpha based on normalized time step for better temporal flow visualization
                    current_time = timesteps[i]
                    alpha = (current_time + 1) / (max_time + 1) if max_time > 0 else 1.0
                    plt.plot([x1, x2], [y1, y2],
                             color=color,
                             alpha=min(1.0, max(0.1, alpha * 0.8)), # Ensure alpha is within bounds
                             linewidth=0.6)

                # Plot points (markers)
                for i, (x, y) in enumerate(trail):
                    current_time = timesteps[i]
                    alpha = (current_time + 1) / (max_time + 1) if max_time > 0 else 1.0
                     # Use alpha for points as well, maybe slightly stronger
                    plt.scatter(x, y,
                                color=color,
                                alpha=min(1.0, max(0.1, alpha)),
                                s=10,
                                edgecolors='none') # Remove edgecolors for cleaner look

        if nodes_plotted == 0:
            logger.warning("No nodes were plotted. Check if 'visible_groups' parameter is set correctly or if data exists for the specified groups.")

        title = "Semantic Drift of Memes Over Time"
        if visible_groups is not None:
            title += f" (Visible Groups: {', '.join(map(str, visible_groups))})"
        else:
             title += " (All Groups)"
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()

        # Use pathlib for path construction if possible
        # output_file = self.vis_dir / "semantic_drift_trails.png"
        # Assuming self.vis_dir is string for now:
        import os
        output_file = os.path.join(self.vis_dir, "semantic_drift_trails.png")

        try:
            # Use configuration for DPI
            dpi = self.vis_config.get('dpi', 150)
            plt.savefig(output_file, bbox_inches='tight', dpi=dpi)
            logger.info(f"Saved semantic drift visualization to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save semantic drift visualization {output_file}: {e}")
        finally:
             # Always close the plot to free memory
            plt.close()


    def draw_semantic_centroid_drift(self, num_generations: int = -1, visible_groups: Optional[List[Any]] = None):
        """
        Visualizes semantic drift generation by generation. Centroids are calculated
        pre-tSNE. The plot is saved after each generation containing visible data is added.

        Args:
            num_generations (int): Max history steps. Defaults to -1 (all).
            visible_groups (Optional[List[Any]]): Groups to display. Defaults to None (all).
        """
        if not self.embedding_model:
            logger.warning("Skipping draw_semantic_centroid_drift: Embedding model not available.")
            return

        G = self.graph_manager.graph
        if G.number_of_nodes() == 0:
            logger.warning("Skipping draw_semantic_centroid_drift: Graph has no nodes.")
            return

        # 1. Collect Data, Calculate & Store Embeddings ONCE
        node_group_map = {}
        group_timestep_embeddings = defaultdict(lambda: defaultdict(list))
        individual_points_metadata = [] # {'group': g, 'timestep': t, 'id': idx}
        all_individual_embeddings_collected = []
        max_timestep_overall = 0
        embedding_idx_counter = 0

        logger.info("Calculating and collecting embeddings...")
        for node_id in G.nodes():
            data = self.graph_manager.get_node_data(node_id)
            if hasattr(data, 'history') and data.history:
                group = getattr(data, 'group', 0)
                node_group_map[node_id] = group
                current_max_t = -1
                node_texts, node_timesteps = [], []
                for t, meme in enumerate(data.history):
                    if num_generations >= 0 and t >= num_generations: break
                    node_texts.append(str(meme))
                    node_timesteps.append(t)
                    current_max_t = t
                if node_texts:
                    node_embeddings = emb_utils.calculate_sentence_embeddings(node_texts, self.embedding_model)
                    if node_embeddings is not None and node_embeddings.size > 0:
                        for i, embedding in enumerate(node_embeddings):
                            timestep = node_timesteps[i]
                            group_timestep_embeddings[group][timestep].append(embedding)
                            individual_points_metadata.append({'group': group, 'timestep': timestep, 'id': embedding_idx_counter})
                            all_individual_embeddings_collected.append(embedding)
                            embedding_idx_counter += 1
                if current_max_t > max_timestep_overall: max_timestep_overall = current_max_t

        if not all_individual_embeddings_collected:
             logger.warning("Skipping draw_semantic_centroid_drift: No embeddings generated.")
             return

        # 2. Calculate High-Dimensional Centroids
        centroid_embeddings_high_dim = {}
        centroid_metadata = []
        logger.info("Calculating high-dimensional group centroids per timestep...")
        centroid_start_idx = len(all_individual_embeddings_collected)
        current_centroid_idx = centroid_start_idx
        for group, timesteps_data in group_timestep_embeddings.items():
            for timestep, embeddings_list in timesteps_data.items():
                if embeddings_list:
                    centroid = np.mean(np.array(embeddings_list), axis=0)
                    centroid_key = (group, timestep)
                    centroid_embeddings_high_dim[centroid_key] = centroid
                    centroid_metadata.append({'group': group, 'timestep': timestep, 'id': current_centroid_idx})
                    current_centroid_idx += 1

        if not centroid_embeddings_high_dim:
            logger.warning("Skipping draw_semantic_centroid_drift: No centroids calculated.")
            # Proceed to plot individuals if they exist? Or return? Let's return for now.
            return

        # 3. Combine Embeddings for t-SNE
        combined_embeddings_dict = {}
        for i, emb in enumerate(all_individual_embeddings_collected): combined_embeddings_dict[i] = emb
        centroid_id_map = {}
        for meta in centroid_metadata:
             centroid_key = (meta['group'], meta['timestep'])
             centroid_id = meta['id']
             combined_embeddings_dict[centroid_id] = centroid_embeddings_high_dim[centroid_key]
             centroid_id_map[centroid_key] = centroid_id

        total_items_for_tsne = len(combined_embeddings_dict)
        logger.info(f"Reducing dimensionality for {total_items_for_tsne} items...")

        # 4. Apply t-SNE
        reduced_2d = emb_utils.reduce_dimensions_tsne(combined_embeddings_dict, random_state=42)
        if not reduced_2d:
            logger.error("Failed to reduce dimensions. Skipping plot.")
            return

        # 5. Organize 2D Coordinates by Timestep
        points_by_timestep = defaultdict(lambda: {'individuals': [], 'centroids': {}})
        # individuals: list of {'x': x, 'y': y, 'group': g}
        # centroids: dict of {group: (cx, cy)}

        for meta in individual_points_metadata:
            if meta['id'] in reduced_2d:
                x, y = reduced_2d[meta['id']]
                points_by_timestep[meta['timestep']]['individuals'].append({
                    'x': x, 'y': y, 'group': meta['group']
                })
        for meta in centroid_metadata:
            if meta['id'] in reduced_2d:
                points_by_timestep[meta['timestep']]['centroids'][meta['group']] = reduced_2d[meta['id']]

        # 6. Plotting Setup
        all_possible_groups = sorted(list(set(node_group_map.values())))
        if not all_possible_groups:
             logger.warning("Skipping plot: No groups defined.")
             return
        cmap = plt.cm.get_cmap('tab10', max(10, len(all_possible_groups)))
        group_to_color = {g: cmap(i % cmap.N) for i, g in enumerate(all_possible_groups)}
        visible_groups_set = set(visible_groups) if visible_groups is not None else None
        dpi = self.vis_config.get('dpi', 150)
        # Use a temporary filename pattern for generation steps, final save overwrites
        output_file = self.vis_dir / "semantic_drift_centroids.png"
        # Define base filename for multi-file output if desired later
        # output_file_base = self.vis_dir / "semantic_drift_gen"

        # 7. Plotting Execution - Generation by Generation
        logger.info(f"Plotting drift generation by generation up to timestep {max_timestep_overall}...")
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.axis('off') # Turn off axis once

        # Store previous centroid positions to draw connecting lines
        prev_centroids_2d = defaultdict(dict) # group -> timestep -> (cx, cy)

        plotted_anything_ever = False
        for t in range(max_timestep_overall + 1):
            data_for_timestep = points_by_timestep.get(t)
            if not data_for_timestep: continue # Skip if no data for this timestep

            centroids_t = data_for_timestep.get('centroids', {})
            individuals_t = data_for_timestep.get('individuals', [])
            plotted_in_this_gen = False

            # Plot Centroids for timestep 't' and connect to 't-1'
            for group, (cx, cy) in centroids_t.items():
                if visible_groups_set is None or group in visible_groups_set:
                    color = group_to_color.get(group, (0.5, 0.5, 0.5))
                    alpha_val = (t + 1) / (max_timestep_overall + 1) if max_timestep_overall > 0 else 1.0

                    # Plot centroid marker
                    ax.scatter(cx, cy, color=color, alpha=min(1.0, max(0.3, alpha_val)), s=60, marker='o', edgecolors='black', linewidth=0.5, zorder=5)
                    plotted_in_this_gen = True

                    # Plot connecting line from previous timestep's centroid if it exists
                    if t > 0 and group in prev_centroids_2d.get(t-1, {}):
                        prev_cx, prev_cy = prev_centroids_2d[t-1][group]
                        prev_alpha = t / (max_timestep_overall + 1) if max_timestep_overall > 0 else 1.0 # Alpha based on start of line
                        ax.plot([prev_cx, cx], [prev_cy, cy], color=color, alpha=min(1.0, max(0.2, prev_alpha * 0.9)), linewidth=1.5)

            # Plot Individual points for timestep 't'
            for point_data in individuals_t:
                 group = point_data['group']
                 if visible_groups_set is None or group in visible_groups_set:
                     color = group_to_color.get(group, (0.5, 0.5, 0.5))
                     alpha_val = (t + 1) / (max_timestep_overall + 1) if max_timestep_overall > 0 else 1.0
                     ax.scatter(point_data['x'], point_data['y'], color=color, alpha=min(1.0, max(0.05, alpha_val * 0.3)), s=8, edgecolors='none')
                     plotted_in_this_gen = True

            # Update previous centroids *after* potentially drawing lines
            if centroids_t:
                prev_centroids_2d[t] = centroids_t.copy() # Store positions for next iteration

            # Save the plot if anything visible was added in this generation
            if plotted_in_this_gen:
                plotted_anything_ever = True
                # Update title dynamically (optional, could set once at the end)
                visible_group_str = f"(Visible Groups: {', '.join(map(str, sorted(list(visible_groups_set))))})" if visible_groups is not None else "(All Groups)"
                current_title = f"Semantic Drift - Generation {t}/{max_timestep_overall} {visible_group_str}"
                ax.set_title(current_title) # Update title each frame
                fig.tight_layout() # Adjust layout

                # Save - overwrites the single output file each time
                try:
                    # Save each gen to a different file:
                    gen_output_file = self.vis_dir / f"semantic_drift_gen_{t:04d}.png"
                    fig.savefig(gen_output_file, bbox_inches='tight', dpi=dpi)
                    logger.info(f"Saved plot for generation {t} to {gen_output_file}")

                    # Overwrite single file:
                    #fig.savefig(output_file, bbox_inches='tight', dpi=dpi)
                    #logger.info(f"Saved plot up to generation {t} to {output_file}")

                except Exception as e:
                    logger.error(f"Failed to save plot for generation {t} to {output_file}: {e}")
                    plt.close(fig)
                    return # Stop if saving fails

        if not plotted_anything_ever:
            logger.warning("No data was plotted based on visibility settings and available timesteps.")

        # Final title update (optional if updated each frame)
        # visible_group_str = f"(Visible Groups: {', '.join(map(str, sorted(list(visible_groups_set))))})" if visible_groups is not None else "(All Groups)"
        # final_title = f"Semantic Drift - Final (Generation {max_timestep_overall}) {visible_group_str}"
        # ax.set_title(final_title)
        # fig.tight_layout()
        # try: # Optional final save if title needs updating
        #      fig.savefig(output_file, bbox_inches='tight', dpi=dpi)
        #      logger.info(f"Saved final plot state to {output_file}")
        # except Exception as e:
        #      logger.error(f"Failed to save final plot state {output_file}: {e}")

        logger.info(f"Finished plotting generation by generation. Final plot at {output_file}")

        # Final Cleanup
        plt.close(fig)



    def draw_semantic_centroid_std_drift(self, num_generations: int = -1, visible_groups: Optional[List[Any]] = None, min_size=20, max_size=150, min_alpha=0.3, max_alpha=1.0):
        """
        Visualizes semantic drift of group centroids only, generation by generation.
        Centroid size/alpha reflect the standard deviation of constituent node embeddings
        (pre-tSNE). Higher std dev -> larger size, lower alpha.

        Args:
            num_generations (int): Max history steps. Defaults to -1 (all).
            visible_groups (Optional[List[Any]]): Groups to display. Defaults to None (all).
            min_size (int): Minimum marker size for centroids.
            max_size (int): Maximum marker size for centroids.
            min_alpha (float): Minimum marker alpha (opacity) for centroids.
            max_alpha (float): Maximum marker alpha (opacity) for centroids.
        """
        if not self.embedding_model:
            logger.warning("Skipping draw_semantic_centroid_std_drift: Embedding model not available.")
            return

        G = self.graph_manager.graph
        if G.number_of_nodes() == 0:
            logger.warning("Skipping draw_semantic_centroid_std_drift: Graph has no nodes.")
            return

        # 1. Collect Data, Calculate & Store Embeddings ONCE
        node_group_map = {}
        group_timestep_embeddings = defaultdict(lambda: defaultdict(list))
        individual_points_metadata = [] # Still needed for T-SNE context
        all_individual_embeddings_collected = []
        max_timestep_overall = 0
        embedding_idx_counter = 0

        logger.info("Calculating and collecting embeddings...")
        for node_id in G.nodes():
            data = self.graph_manager.get_node_data(node_id)
            if hasattr(data, 'history') and data.history:
                group = getattr(data, 'group', 0)
                node_group_map[node_id] = group
                current_max_t = -1
                node_texts, node_timesteps = [], []
                for t, meme in enumerate(data.history):
                    if num_generations >= 0 and t >= num_generations: break
                    node_texts.append(str(meme))
                    node_timesteps.append(t)
                    current_max_t = t
                if node_texts:
                    node_embeddings = emb_utils.calculate_sentence_embeddings(node_texts, self.embedding_model)
                    if node_embeddings is not None and node_embeddings.size > 0:
                        for i, embedding in enumerate(node_embeddings):
                            timestep = node_timesteps[i]
                            group_timestep_embeddings[group][timestep].append(embedding)
                            # Collect metadata even if not plotted, needed for T-SNE layout
                            individual_points_metadata.append({'group': group, 'timestep': timestep, 'id': embedding_idx_counter})
                            all_individual_embeddings_collected.append(embedding)
                            embedding_idx_counter += 1
                if current_max_t > max_timestep_overall: max_timestep_overall = current_max_t

        if not all_individual_embeddings_collected:
             logger.warning("Skipping draw_semantic_centroid_std_drift: No embeddings generated.")
             return

        # 2. Calculate High-Dimensional Centroids AND Standard Deviations
        centroid_data_high_dim = {} # (group, timestep) -> (centroid_vector, std_dev_scalar)
        centroid_metadata = []
        all_std_devs = []
        logger.info("Calculating high-dimensional centroids and std deviations per timestep...")
        centroid_start_idx = len(all_individual_embeddings_collected)
        current_centroid_idx = centroid_start_idx
        for group, timesteps_data in group_timestep_embeddings.items():
            for timestep, embeddings_list in timesteps_data.items():
                if embeddings_list:
                    embeddings_array = np.array(embeddings_list)
                    centroid = np.mean(embeddings_array, axis=0)
                    if embeddings_array.shape[0] > 1:
                        # Calculate std dev along each dimension, then average
                        std_dev_vector = np.std(embeddings_array, axis=0)
                        std_dev_scalar = np.mean(std_dev_vector)
                    else:
                        std_dev_scalar = 0.0 # No deviation if only one point

                    all_std_devs.append(std_dev_scalar)
                    centroid_key = (group, timestep)
                    centroid_data_high_dim[centroid_key] = (centroid, std_dev_scalar)
                    centroid_metadata.append({'group': group, 'timestep': timestep, 'id': current_centroid_idx})
                    current_centroid_idx += 1

        if not centroid_data_high_dim:
            logger.warning("Skipping draw_semantic_centroid_std_drift: No centroids calculated.")
            return

        # Calculate min/max std dev for scaling, handle edge case of no variance
        min_std = min(all_std_devs) if all_std_devs else 0.0
        max_std = max(all_std_devs) if all_std_devs else 0.0
        std_range = max_std - min_std
        if std_range < 1e-9: # Avoid division by zero if all std devs are the same (or only one)
            std_range = 1.0 # All points will get median size/alpha

        # 3. Combine Embeddings for t-SNE (incl. individuals for layout context)
        combined_embeddings_dict = {}
        for i, emb in enumerate(all_individual_embeddings_collected): combined_embeddings_dict[i] = emb
        centroid_id_map = {}
        for meta in centroid_metadata:
             centroid_key = (meta['group'], meta['timestep'])
             centroid_id = meta['id']
             centroid_vector, _ = centroid_data_high_dim[centroid_key] # Only need vector for T-SNE
             combined_embeddings_dict[centroid_id] = centroid_vector
             centroid_id_map[centroid_key] = centroid_id

        total_items_for_tsne = len(combined_embeddings_dict)
        logger.info(f"Reducing dimensionality for {total_items_for_tsne} items...")

        # 4. Apply t-SNE
        reduced_2d = emb_utils.reduce_dimensions_tsne(combined_embeddings_dict, random_state=42)
        if not reduced_2d:
            logger.error("Failed to reduce dimensions. Skipping plot.")
            return

        # 5. Organize 2D Centroid Coordinates and link Std Dev
        centroid_points_by_timestep = defaultdict(dict) # timestep -> group -> (cx, cy, std_dev)
        for meta in centroid_metadata:
            if meta['id'] in reduced_2d:
                group, timestep = meta['group'], meta['timestep']
                centroid_key = (group, timestep)
                _, std_dev = centroid_data_high_dim[centroid_key]
                centroid_points_by_timestep[timestep][group] = (*reduced_2d[meta['id']], std_dev)

        # 6. Plotting Setup
        all_possible_groups = sorted(list(set(node_group_map.values())))
        if not all_possible_groups:
             logger.warning("Skipping plot: No groups defined.")
             return
        cmap = plt.cm.get_cmap('tab10', max(10, len(all_possible_groups)))
        group_to_color = {g: cmap(i % cmap.N) for i, g in enumerate(all_possible_groups)}
        visible_groups_set = set(visible_groups) if visible_groups is not None else None
        dpi = self.vis_config.get('dpi', 150)
        output_file = self.vis_dir / "semantic_drift_centroids_std.png"

        # 7. Plotting Execution - Generation by Generation (Centroids Only)
        logger.info(f"Plotting centroid drift (std dev mapped to size/alpha) up to timestep {max_timestep_overall}...")
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.axis('off')

        prev_centroids_2d = defaultdict(dict) # Stores only (cx, cy) for line drawing
        plotted_anything_ever = False

        for t in range(max_timestep_overall + 1):
            centroids_t_data = centroid_points_by_timestep.get(t, {})
            if not centroids_t_data: continue

            plotted_in_this_gen = False
            current_gen_centroids = {} # Store (cx,cy) for this gen to update prev_centroids_2d

            for group, (cx, cy, std_dev) in centroids_t_data.items():
                if visible_groups_set is None or group in visible_groups_set:
                    color = group_to_color.get(group, (0.5, 0.5, 0.5))

                    # Calculate size and alpha based on normalized std dev
                    normalized_std = (std_dev - min_std) / std_range if std_range > 1e-9 else 0.5
                    size = min_size + normalized_std * (max_size - min_size)
                    alpha_val = max_alpha - normalized_std * (max_alpha - min_alpha)
                    alpha_val = np.clip(alpha_val, 0.0, 1.0) # Ensure alpha is valid

                    # Plot centroid marker with calculated size/alpha
                    ax.scatter(cx, cy, color=color, alpha=alpha_val, s=size, marker='o', edgecolors='black', linewidth=0.5, zorder=5)
                    plotted_in_this_gen = True
                    current_gen_centroids[group] = (cx, cy) # Store coords for line drawing

                    # Plot connecting line from previous timestep's centroid
                    if t > 0 and group in prev_centroids_2d.get(t-1, {}):
                        prev_cx, prev_cy = prev_centroids_2d[t-1][group]
                        # Line alpha based on time
                        line_alpha = t / (max_timestep_overall + 1) if max_timestep_overall > 0 else 1.0
                        ax.plot([prev_cx, cx], [prev_cy, cy], color=color, alpha=min(1.0, max(0.2, line_alpha * 0.9)), linewidth=1.5)

            # Update previous centroid positions *after* iterating through all groups for the timestep
            if current_gen_centroids:
                prev_centroids_2d[t] = current_gen_centroids

            # Save the plot if anything visible was added in this generation
            if plotted_in_this_gen:
                plotted_anything_ever = True
                visible_group_str = f"(Visible Groups: {', '.join(map(str, sorted(list(visible_groups_set))))})" if visible_groups is not None else "(All Groups)"
                current_title = f"Centroid Drift (Size/Alpha ~ StdDev) - Gen {t}/{max_timestep_overall} {visible_group_str}"
                ax.set_title(current_title)
                fig.tight_layout()

                try:
                    fig.savefig(output_file, bbox_inches='tight', dpi=dpi)
                    logger.info(f"Saved plot up to generation {t} to {output_file}")
                except Exception as e:
                    logger.error(f"Failed to save plot for generation {t} to {output_file}: {e}")
                    plt.close(fig)
                    return

        if not plotted_anything_ever:
            logger.warning("No centroids were plotted based on visibility settings and available timesteps.")

        logger.info(f"Finished plotting std-dev centroid drift. Final plot at {output_file}")

        # Final Cleanup
        plt.close(fig)