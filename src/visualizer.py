import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict
import csv
from typing import Dict, Any, Optional, Tuple, List

from PIL import Image

from graph_manager import GraphManager
from data_structures import MemeNodeData
# Import embedding utils if needed, or receive model/functions
import embeddings_utils as emb_utils

logger = logging.getLogger(__name__)

# Set random seed for graph layout consistency
np.random.seed(2)

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

        #nx.draw_networkx_labels(G, pos, font_size=7) # Slightly larger font

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

        scores = [2**(4*s)-1 for s in scores]
        # Normalize scores to 0-1 for coloring
        min_s, max_s = min(scores), max(scores)
        color_values = [(s - min_s) / (max_s - min_s) if (max_s - min_s) > 0 else 0.5 for s in scores]

        node_sizes = [self.vis_config['node_min_size'] + (self.vis_config['node_max_size'] - self.vis_config['node_min_size']) * norm_influences.get(nid, 0.5)
                      for nid in node_ids_ordered]

        #base_filename = f"graph_score_gen_{generation}.png" if generation is not None else "graph_score_final.png"
        base_filename = "graph_score.png"
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

        #base_filename = f"graph_change_gen_{generation}.png"
        base_filename = "graph_change.png"
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
        embeddings_2d = emb_utils.reduce_dimensions_tsne(embeddings_dict, random_state=41)
        if not embeddings_2d:
            logger.error("Failed to reduce embedding dimensions for draw_embs. Skipping plot.")
            return

        # Use reduced embeddings as layout positions
        pos = embeddings_2d

        # Colors based on group
        unique_groups = sorted(list(set(groups)))
        num_groups = len(unique_groups)
        cmap = plt.cm.get_cmap('tab10', num_groups)
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
        plt.figure(figsize=(12, 7))  # 7, 4
        num_groups = len(avg_scores)
        # Get a colormap; providing num_groups ensures distinct colors are selected if possible
        colors = plt.cm.get_cmap('tab10', num_groups)

        # Define a list of line styles to cycle through
        line_styles = ['-', '--', ':', '-.', (0, (5, 5))] # Solid, Dashed, Dotted, Dash-Dot, Custom Dash-Dot-Dot

        base_linewidth = 2
        background_linewidth_increase = 1 # How much thicker the background line is
        background_alpha = 0.1 # How transparent the background line is

        for i, (group, avg_history) in enumerate(sorted(avg_scores.items())): # Sort by group ID
             # Find first non-NaN index to start plotting from
             first_valid_index = np.where(~np.isnan(avg_history))[0]
             if len(first_valid_index) > 0:
                  start_index = first_valid_index[0]
                  # Select a line style based on the group index
                  current_style = line_styles[i % len(line_styles)]
                  # Get the color for this group
                  current_color = colors(i)

                  # 1. Plot the background line (thicker, lighter, solid)
                  plt.plot(range(start_index, len(avg_history)), avg_history[start_index:],
                           color='black',
                           alpha=background_alpha, # Make it quite transparent
                           linewidth=base_linewidth + background_linewidth_increase, # Make it thicker
                           linestyle='-', # Always solid for the background
                           zorder=1, # Ensure it's plotted behind the main line
                           label=None) # No label for the background line

                  # 2. Plot the main styled line on top
                  plt.plot(range(start_index, len(avg_history)), avg_history[start_index:],
                           color=current_color,
                           linestyle=current_style, # Assign the selected line style
                           linewidth=base_linewidth, # Use the standard width
                           label=f"Group {group}", # Label only the main line for the legend
                           zorder=2) # Ensure it's plotted on top

             else:
                  logger.warning(f"Group {group} has no valid score data to plot.")

        plt.xlabel("Generation")
        plt.ylabel("Average Meme Score")
        # plt.title("Average Meme Score per Generation by Group (with Line Styles)") # Optional: Update title
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1) # Scores are normalized 0-1

        filename = self.vis_dir / "plot_score_history_bygroup.png"
        try:
            plt.savefig(filename, bbox_inches='tight', dpi=self.vis_config['dpi'])
            logger.info(f"Saved score history plot to {filename}")
        except Exception as e:
            logger.error(f"Failed to save score history plot {filename}: {e}")
        plt.close()




    # def draw_semantic_drift(self, num_generations: int = -1, visible_groups: Optional[List[Any]] = None):
    #     """
    #     Visualizes semantic drift of memes per node over time using embedding space trajectories.

    #     Args:
    #         num_generations (int): The maximum number of generations (history steps) to include.
    #                                Defaults to -1 (include all).
    #         visible_groups (Optional[List[Any]]): A list of group identifiers for which to display
    #                                               trajectories. If None (default), trajectories for
    #                                               all groups are displayed. Nodes from *all* groups
    #                                               are still used for embedding calculation and
    #                                               dimensionality reduction.
    #     """
    #     if not self.embedding_model:
    #         logger.warning("Skipping draw_semantic_drift: Embedding model not available.")
    #         return

    #     G = self.graph_manager.graph
    #     if G.number_of_nodes() == 0:
    #         logger.warning("Skipping draw_semantic_drift: Graph has no nodes.")
    #         return

    #     # Collect all meme histories and associated metadata
    #     all_texts = []
    #     text_index_map = []  # List of (node_id, timestep)
    #     node_groups = {}     # node_id -> group

    #     for node_id in G.nodes():
    #         # Use a safer way to access node data if structure varies
    #         data = self.graph_manager.get_node_data(node_id)
    #         # Check if data exists and has 'history' attribute
    #         if hasattr(data, 'history') and data.history:
    #             group = getattr(data, 'group', 0) # Default group 0 if not present
    #             node_groups[node_id] = group
    #             for t, meme in enumerate(data.history):
    #                 if num_generations >= 0 and t >= num_generations:
    #                     break
    #                 all_texts.append(str(meme)) # Ensure text is string
    #                 text_index_map.append((node_id, t))

    #     if not all_texts:
    #          logger.warning("Skipping draw_semantic_drift: No meme history found in any node.")
    #          return

    #     # Compute all embeddings
    #     logger.info(f"Calculating embeddings for {len(all_texts)} memes...")
    #     all_embeddings = emb_utils.calculate_sentence_embeddings(all_texts, self.embedding_model)
    #     if all_embeddings is None or all_embeddings.size == 0:
    #         logger.error("Failed to calculate embeddings for draw_semantic_drift. Skipping plot.")
    #         return

    #     # Reduce to 2D using t-SNE
    #     logger.info(f"Reducing dimensionality for {len(all_texts)} memes...")
    #     # Ensure embeddings_dict keys match indices expected by text_index_map
    #     embeddings_dict = {i: emb for i, emb in enumerate(all_embeddings)}
    #     reduced_2d = emb_utils.reduce_dimensions_tsne(embeddings_dict, random_state=42)
    #     if not reduced_2d:
    #         logger.error("Failed to reduce embedding dimensions for draw_semantic_drift. Skipping plot.")
    #         return

    #     # Organize by node for plotting
    #     node_trails = defaultdict(list)      # node_id -> list of (x, y)
    #     node_timesteps = defaultdict(list)   # node_id -> list of timestep (used for alpha)
    #     for i, (node_id, timestep) in enumerate(text_index_map):
    #         if i in reduced_2d:
    #             node_trails[node_id].append(reduced_2d[i])
    #             node_timesteps[node_id].append(timestep)

    #     # Set up colormap for clusters
    #     unique_groups_overall = sorted(list(set(node_groups.values())))
    #     num_groups = len(unique_groups_overall)
    #     cmap = plt.cm.get_cmap('tab10', len(unique_groups_overall))
    #     group_to_color = {
    #         g: cmap(i / (num_groups - 1) if num_groups > 1 else 0.5)
    #         for i, g in enumerate(unique_groups_overall)
    #     }

    #     # Convert visible_groups to a set for efficient lookup, if provided
    #     visible_groups_set = set(visible_groups) if visible_groups is not None else None

    #     # Plotting
    #     logger.info(f"Plotting...")
    #     plt.figure(figsize=(15, 15))
    #     nodes_plotted = 0
    #     for node_id, trail in node_trails.items():
    #         group = node_groups.get(node_id, 0) # Get group, default 0

    #         # Check if this group should be visible
    #         if visible_groups_set is None or group in visible_groups_set:
    #             nodes_plotted += 1
    #             color = group_to_color.get(group, (0.5, 0.5, 0.5)) # Default color grey

    #             timesteps = node_timesteps[node_id]
    #             n = len(trail)
    #             max_time = max(timesteps) if timesteps else 0

    #             # Plot lines connecting consecutive points
    #             for i in range(n - 1):
    #                 x1, y1 = trail[i]
    #                 x2, y2 = trail[i + 1]
    #                 # Alpha based on normalized time step for better temporal flow visualization
    #                 current_time = timesteps[i]
    #                 alpha = (current_time + 1) / (max_time + 1) if max_time > 0 else 1.0
    #                 plt.plot([x1, x2], [y1, y2],
    #                          color=color,
    #                          alpha=min(1.0, max(0.1, alpha * 0.8)), # Ensure alpha is within bounds
    #                          linewidth=0.6)

    #             # Plot points (markers)
    #             for i, (x, y) in enumerate(trail):
    #                 current_time = timesteps[i]
    #                 alpha = (current_time + 1) / (max_time + 1) if max_time > 0 else 1.0
    #                  # Use alpha for points as well, maybe slightly stronger
    #                 plt.scatter(x, y,
    #                             color=color,
    #                             alpha=min(1.0, max(0.1, alpha)),
    #                             s=10,
    #                             edgecolors='none') # Remove edgecolors for cleaner look

    #     if nodes_plotted == 0:
    #         logger.warning("No nodes were plotted. Check if 'visible_groups' parameter is set correctly or if data exists for the specified groups.")

    #     title = "Semantic Drift of Memes Over Time"
    #     if visible_groups is not None:
    #         title += f" (Visible Groups: {', '.join(map(str, visible_groups))})"
    #     else:
    #          title += ""
    #     plt.title(title)
    #     plt.axis('off')
    #     plt.tight_layout()

    #     # Use pathlib for path construction if possible
    #     # output_file = self.vis_dir / "semantic_drift_trails.png"
    #     # Assuming self.vis_dir is string for now:
    #     import os
    #     output_file = os.path.join(self.vis_dir, "semantic_drift_trails.png")

    #     try:
    #         # Use configuration for DPI
    #         dpi = self.vis_config.get('dpi', 150)
    #         plt.savefig(output_file, bbox_inches='tight', dpi=dpi)
    #         logger.info(f"Saved semantic drift visualization to {output_file}")
    #     except Exception as e:
    #         logger.error(f"Failed to save semantic drift visualization {output_file}: {e}")
    #     finally:
    #          # Always close the plot to free memory
    #         plt.close()


    def draw_semantic_drift(self, num_generations: int = -1, visible_groups: Optional[List[Any]] = None, min_size=5, max_size=2000, min_alpha=0.1, max_alpha=1.0):
        """
        Visualizes semantic drift generation by generation. Centroids are calculated
        pre-tSNE. Centroid size/alpha reflect the standard deviation of constituent node
        embeddings (pre-tSNE). Higher std dev -> larger size, lower alpha.
        Individual points are also plotted with time-based alpha.
        The plot is saved after each generation containing visible data is added.

        Args:
            num_generations (int): Max history steps. Defaults to -1 (all).
            visible_groups (Optional[List[Any]]): Groups to display. Defaults to None (all).
            min_size (int): Minimum marker size for centroids.
            max_size (int): Maximum marker size for centroids.
            min_alpha (float): Minimum marker alpha (opacity) for centroids.
            max_alpha (float): Maximum marker alpha (opacity) for centroids.
        """
        if not self.embedding_model:
            logger.warning("Skipping draw_semantic_drift: Embedding model not available.")
            return

        G = self.graph_manager.graph
        if G.number_of_nodes() == 0:
            logger.warning("Skipping draw_semantic_drift: Graph has no nodes.")
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
             logger.warning("Skipping draw_semantic_drift: No embeddings generated.")
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
            logger.warning("Skipping draw_semantic_drift: No centroids calculated.")
            # Might still plot individuals if they exist, proceed if individuals were collected
            if not all_individual_embeddings_collected:
                return # Return if no individuals either

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

        # 5. Organize 2D Coordinates by Timestep, Link Std Dev
        points_by_timestep = defaultdict(lambda: {'individuals': [], 'centroids': {}})
        # individuals: list of {'x': x, 'y': y, 'group': g}
        # centroids: dict of {group: (cx, cy, std_dev)}

        for meta in individual_points_metadata:
            if meta['id'] in reduced_2d:
                x, y = reduced_2d[meta['id']]
                points_by_timestep[meta['timestep']]['individuals'].append({
                    'x': x, 'y': y, 'group': meta['group']
                })
        for meta in centroid_metadata:
            if meta['id'] in reduced_2d:
                group, timestep = meta['group'], meta['timestep']
                centroid_key = (group, timestep)
                _, std_dev = centroid_data_high_dim.get(centroid_key, (None, 0.0)) # Default std_dev if key somehow missing
                points_by_timestep[timestep]['centroids'][group] = (*reduced_2d[meta['id']], std_dev)

        # 6. Plotting Setup
        unique_groups_overall = sorted(list(set(node_group_map.values())))

        # Determine which groups to actually plot and include in the legend
        if visible_groups is not None:
            visible_groups_set = set(visible_groups)
            # Filter the list of groups we will work with
            unique_groups_to_plot = [g for g in unique_groups_overall if g in visible_groups_set]
        else:
            visible_groups_set = None # Still useful for checks later if needed
            unique_groups_to_plot = unique_groups_overall # Plot all groups

        if not unique_groups_to_plot:
             logger.warning("No groups to plot based on visibility settings.")
             return # Exit if no visible groups

        num_groups = len(unique_groups_to_plot)
        cmap = plt.cm.get_cmap('tab10') # Consider 'viridis', 'plasma', or grayscale-friendly colormaps too
        group_to_color = {
            g: cmap(i / (num_groups - 1) if num_groups > 1 else 0.5)
            for i, g in enumerate(unique_groups_to_plot)
        }

        # Define hatch patterns for textures - add more if you have many groups
        hatch_patterns = ['//', '\\\\', '||', '--', '**', 'xx', '++', 'OO', '..', 'oo']
        group_to_hatch = {
            # Assign a hatch pattern cyclically to each group intended for plotting
            g: hatch_patterns[i % len(hatch_patterns)]
            for i, g in enumerate(unique_groups_to_plot)
        }

        dpi = self.vis_config.get('dpi', 150)
        output_file_base = self.vis_dir / "semantic_drift_gen" # Base name for frame sequence

        # Prepare legend handles using the filtered groups
        legend_handles = []
        for group in unique_groups_to_plot:
            color = group_to_color[group]
            hatch = group_to_hatch[group]
            # Create a representative patch for the legend entry
            # Using Patch allows showing both color and hatch
            handle = patches.Patch(facecolor=color, hatch=hatch, edgecolor='black', label=f'Group {group}')
            legend_handles.append(handle)

              
        # 7. Plotting Execution - Generation by Generation
        logger.info(f"Plotting drift (Centroid Size/Alpha ~ StdDev, Hatch ~ Group) generation by generation up to timestep {max_timestep_overall}...")
        fig, ax = plt.subplots(figsize=(15, 15))      
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off') # Turn off axis once

        # Store previous centroid COORDINATES to draw connecting lines
        prev_centroids_2d_coords = defaultdict(dict) # timestep -> group -> (cx, cy)

        # Define a scaling factor for converting scatter size (area) to circle radius
        # Adjust this value based on visual results
        radius_scale_factor = 0.2

        plotted_anything_ever = False
        for t in range(max_timestep_overall + 1):
            data_for_timestep = points_by_timestep.get(t)
            if not data_for_timestep: continue # Skip if no data for this timestep

            centroids_t_data = data_for_timestep.get('centroids', {})
            individuals_t = data_for_timestep.get('individuals', [])
            plotted_in_this_gen = False
            current_gen_centroid_coords = {} # Store coords for this gen to update prev_centroids_2d_coords

            # Plot Centroids for timestep 't' and connect to 't-1'
            for group, (cx, cy, std_dev) in centroids_t_data.items():
                # Use the pre-filtered visible_groups_set or check if None (all visible)
                # Also check if the group is in our mapping (it should be if filtered correctly)
                if (visible_groups_set is None or group in visible_groups_set) and group in group_to_color:
                    color = group_to_color[group]
                    hatch = group_to_hatch.get(group, None) # Get the hatch pattern

                    # Calculate size and alpha based on normalized std dev (same as before)
                    normalized_std = (std_dev - min_std) / std_range if std_range > 1e-9 else 0.5
                    scaled_norm_std_size = normalized_std ** 5
                    size = min_size + scaled_norm_std_size * (max_size - min_size)
                    scaled_norm_std_alpha = normalized_std ** 5
                    alpha_val = max_alpha - scaled_norm_std_alpha * (max_alpha - min_alpha)
                    alpha_val = np.clip(alpha_val, 0.0, 1.0)

                    # Convert scatter size (area) to radius for Circle patch
                    # Use max(0, size) to avoid errors with negative size if normalization goes wrong
                    radius = np.sqrt(max(0, size) / np.pi) * radius_scale_factor # Adjust scale factor as needed

                    # --- MODIFIED PART: Use patches.Circle ---
                    circle = patches.Circle(
                        (cx, cy),
                        radius=radius,
                        facecolor=color,         # Set face color
                        alpha=alpha_val,         # Set alpha
                        hatch=hatch,             # Set hatch pattern
                        edgecolor='black',       # Keep edge color
                        linewidth=0.5,           # Keep line width
                        zorder=5                 # Keep z-order
                    )
                    ax.add_patch(circle) # Add the circle patch to the axes
                    # --- END MODIFIED PART ---

                    plotted_in_this_gen = True
                    current_gen_centroid_coords[group] = (cx, cy) # Store coords for line drawing

                    # Plot connecting line from previous timestep's centroid if it exists (same as before)
                    if t > 0 and group in prev_centroids_2d_coords.get(t-1, {}):
                        prev_cx, prev_cy = prev_centroids_2d_coords[t-1][group]
                        line_alpha = t / (max_timestep_overall + 1) if max_timestep_overall > 0 else 1.0
                        ax.plot([prev_cx, cx], [prev_cy, cy], color=color, alpha=min(1.0, max(0.3, line_alpha * 0.9)), linewidth=1.5, zorder=4) # Lower zorder for lines

            # Plot Individual points for timestep 't' (keep as scatter, no hatching needed)
            for point_data in individuals_t:
                 group = point_data['group']
                 # Check visibility for individuals too
                 if (visible_groups_set is None or group in visible_groups_set) and group in group_to_color:
                     color = group_to_color[group]
                     indiv_alpha_val = (t + 1) / (max_timestep_overall + 1) if max_timestep_overall > 0 else 1.0
                     ax.scatter(point_data['x'], point_data['y'], color=color, alpha=0.5, s=10, marker='.', edgecolors='none', zorder=3) # Use smaller marker maybe
                     plotted_in_this_gen = True

            # Update previous centroid coordinates *after* iterating through all groups for the timestep
            if current_gen_centroid_coords:
                prev_centroids_2d_coords[t] = current_gen_centroid_coords

            # Save the plot if anything visible was added in this generation
            if plotted_in_this_gen:
                plotted_anything_ever = True
                # Update title dynamically
                visible_group_str = f"(Visible Groups: {', '.join(map(str, sorted(unique_groups_to_plot)))})" if visible_groups is not None else ""
                current_title = f"Gen {t}/{max_timestep_overall}"
                ax.set_title(current_title)

                # --- ADD LEGEND ---
                # Add the legend on each frame if desired
                ax.legend(handles=legend_handles, title="Groups", loc='lower left')
                # --- END LEGEND ---

                fig.tight_layout() # Adjust layout

                gen_output_file = self.vis_dir / f"{output_file_base.stem}_{t:04d}.png"
                #gen_output_file = self.vis_dir / f"{output_file_base.stem}.png"
                try:
                    fig.savefig(gen_output_file, bbox_inches='tight', dpi=dpi)
                    logger.info(f"Saved plot for generation {t} to {gen_output_file}")
                except Exception as e:
                    logger.error(f"Failed to save plot for generation {t} to {gen_output_file}: {e}")
                    plt.close(fig)
                    return

        if not plotted_anything_ever:
            logger.warning("No data was plotted based on visibility settings and available timesteps.")
        # else: # Optional: If you want the legend only on the *last* saved frame uncomment this block
        #     # And comment out the ax.legend() call inside the loop above
        #     try:
        #         visible_group_str = f"(Visible Groups: {', '.join(map(str, sorted(unique_groups_to_plot)))})" if visible_groups is not None else ""
        #         final_title = f"Semantic Drift (Centroid Size/Alpha ~ StdDev, Hatch ~ Group) - Final (Gen {max_timestep_overall}) {visible_group_str}"
        #         ax.set_title(final_title)
        #         ax.legend(handles=legend_handles, title="Groups", loc='best') # Add legend to the final state
        #         fig.tight_layout()
        #         final_output_file = self.vis_dir / f"{output_file_base.stem}_final.png"
        #         fig.savefig(final_output_file, bbox_inches='tight', dpi=dpi)
        #         logger.info(f"Saved final plot state with legend to {final_output_file}")
        #     except Exception as e:
        #          logger.error(f"Failed to save final plot state {final_output_file}: {e}")


        logger.info(f"Finished plotting generation by generation. Frame sequence saved with base name '{output_file_base.stem}'")

        # Final Cleanup
        plt.close(fig)



    def plot_semantic_centroid_distance_drift(self, num_generations: int = -1, visible_groups: Optional[List[Any]] = None, relative_group_id: Optional[Any] = None):
        """
        Plots the Euclidean distance of each group's centroid over generations.
        If relative_group_id is None, plots distance to the nearest other group's centroid.
        If relative_group_id is set, plots distance to that specific group's centroid.
        Line thickness represents the standard deviation of embeddings within the group.

        Args:
            num_generations (int): Max history steps. Defaults to -1 (all).
            visible_groups (Optional[List[Any]]): Groups to display. Defaults to None (all).
            relative_group_id (Optional[Any]): If set, calculate distance relative to this group. Defaults to None.
        """
        if not hasattr(self, 'embedding_model') or not self.embedding_model:
            logger.warning("Skipping plot_semantic_centroid_distance_drift: Embedding model not available.")
            return
        global emb_utils
        if 'emb_utils' not in globals():
             logger.error("emb_utils not found. Cannot calculate embeddings.")
             return

        G = self.graph_manager.graph
        if G.number_of_nodes() == 0:
            logger.warning("Skipping plot_semantic_centroid_distance_drift: Graph has no nodes.")
            return

        node_group_map = {}
        group_timestep_embeddings = defaultdict(lambda: defaultdict(list))
        max_timestep_overall = 0

        logger.info("Calculating and collecting embeddings for distance drift...")
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
                if current_max_t > max_timestep_overall: max_timestep_overall = current_max_t

        if not group_timestep_embeddings:
            logger.warning("Skipping plot_semantic_centroid_distance_drift: No embeddings collected.")
            return

        centroid_data_high_dim = {}
        all_std_devs = []
        logger.info("Calculating high-dimensional centroids and std deviations per timestep...")
        for group, timesteps_data in group_timestep_embeddings.items():
            for timestep, embeddings_list in timesteps_data.items():
                if embeddings_list:
                    embeddings_array = np.array(embeddings_list)
                    centroid = np.mean(embeddings_array, axis=0)
                    if embeddings_array.shape[0] > 1:
                        std_dev_vector = np.std(embeddings_array, axis=0)
                        std_dev_scalar = np.mean(std_dev_vector)
                    else:
                        std_dev_scalar = 0.0

                    all_std_devs.append(std_dev_scalar)
                    centroid_key = (group, timestep)
                    centroid_data_high_dim[centroid_key] = (centroid, std_dev_scalar)

        if not centroid_data_high_dim:
            logger.warning("Skipping plot_semantic_centroid_distance_drift: No centroids calculated.")
            return

        min_std = min(all_std_devs) if all_std_devs else 0.0
        max_std = max(all_std_devs) if all_std_devs else 0.0
        std_range = max_std - min_std
        if std_range < 1e-9:
            std_range = 1.0

        group_centroid_distances = defaultdict(lambda: defaultdict(float))
        group_stds = defaultdict(lambda: defaultdict(float))

        all_groups = sorted(list(set(g for g, t in centroid_data_high_dim.keys())))
        if len(all_groups) < 2 and relative_group_id is None:
             logger.warning("Skipping plot_semantic_centroid_distance_drift: Need at least two groups with centroids to calculate nearest distances.")
             return
        if relative_group_id is not None and relative_group_id not in all_groups:
            logger.error(f"Skipping plot_semantic_centroid_distance_drift: Relative group ID '{relative_group_id}' not found among groups with centroids.")
            return

        min_distance_overall = float('inf')
        max_distance_overall = 0.0

        for t in range(max_timestep_overall + 1):
            centroids_at_timestep = {}
            stds_at_timestep = {}
            groups_present_at_t = []
            for group in all_groups:
                centroid_key = (group, t)
                if centroid_key in centroid_data_high_dim:
                    centroid, std_dev = centroid_data_high_dim[centroid_key]
                    centroids_at_timestep[group] = centroid
                    stds_at_timestep[group] = std_dev
                    groups_present_at_t.append(group)

            if not groups_present_at_t: continue

            relative_centroid_present = False
            reference_centroid = None
            if relative_group_id is not None:
                if relative_group_id in centroids_at_timestep:
                    reference_centroid = centroids_at_timestep[relative_group_id]
                    relative_centroid_present = True
                else:
                    # If relative group doesn't exist at this timestep, skip calculation for this t
                    continue

            if relative_group_id is not None and relative_centroid_present:
                # Calculate distance relative to the specified group
                for group in groups_present_at_t:
                    if group == relative_group_id: continue # Skip distance to self
                    current_centroid = centroids_at_timestep[group]
                    distance = np.linalg.norm(current_centroid - reference_centroid)
                    group_centroid_distances[group][t] = distance
                    group_stds[group][t] = stds_at_timestep[group]
                    min_distance_overall = min(min_distance_overall, distance)
                    max_distance_overall = max(max_distance_overall, distance)
            elif relative_group_id is None:
                # Calculate distance to nearest other group (original behavior)
                 if len(groups_present_at_t) < 2: continue # Need >= 2 for nearest calculation

                 for group in groups_present_at_t:
                    min_distance = float('inf')
                    current_centroid = centroids_at_timestep[group]
                    for other_group in groups_present_at_t:
                        if group != other_group:
                            other_centroid = centroids_at_timestep[other_group]
                            distance = np.linalg.norm(current_centroid - other_centroid)
                            min_distance = min(min_distance, distance)

                    if min_distance != float('inf'):
                        group_centroid_distances[group][t] = min_distance
                        group_stds[group][t] = stds_at_timestep[group]
                        min_distance_overall = min(min_distance_overall, min_distance)
                        max_distance_overall = max(max_distance_overall, min_distance)


        all_possible_groups_with_data = sorted(list(group_centroid_distances.keys()))
        if not all_possible_groups_with_data:
             logger.warning("Skipping plot: No distance data calculated (check relative group or number of groups).")
             return

        unique_groups_overall = sorted(list(set(node_group_map.values())))
        num_groups = len(unique_groups_overall)
        cmap = plt.cm.get_cmap('tab10')
        #cmap = plt.cm.get_cmap('hsv', max(10, len(unique_groups_overall)))
        group_to_color = {
            g: cmap(i / (num_groups - 1) if num_groups > 1 else 0.5)
            for i, g in enumerate(unique_groups_overall)
        }
        # group_to_color = {g: cmap(i % cmap.N) for i, g in enumerate(unique_groups_overall)}

        visible_groups_set = set(visible_groups) if visible_groups is not None else None
        dpi = self.vis_config.get('dpi', 150)

        # Adjust filename and title based on relative_group_id
        if relative_group_id is not None:
            output_file = self.vis_dir / f"plot_semantic_centroid_distance_drift_relative_to_{relative_group_id}.png"
            plot_title = f"Semantic Centroid Distance Drift Relative to Group {relative_group_id}"
            y_label = f"Distance to Centroid of Group {relative_group_id}"
        else:
            output_file = self.vis_dir / "plot_semantic_centroid_distance_drift_nearest.png"
            plot_title = "Semantic Centroid Distance Drift to Nearest Group"
            y_label = "Distance to Nearest Other Centroid"


        logger.info(f"Plotting centroid distance drift: {plot_title}")
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_xlabel("Generation")
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)

        max_gen_overall = 0
        plotted_anything = False
        actual_min_plotted = float('inf')
        actual_max_plotted = 0.0

        for group in all_possible_groups_with_data:
            # Skip the relative group itself if it's specified
            if relative_group_id is not None and group == relative_group_id:
                continue
            if visible_groups_set is None or group in visible_groups_set:
                distances = group_centroid_distances[group]
                stds = group_stds[group]
                if distances:
                    generations = sorted(distances.keys())
                    if len(generations) < 2: continue

                    distance_values = [distances[gen] for gen in generations]
                    std_values = [stds.get(gen, 0.0) for gen in generations]

                    current_min = min(distance_values)
                    current_max = max(distance_values)
                    actual_min_plotted = min(actual_min_plotted, current_min)
                    actual_max_plotted = max(actual_max_plotted, current_max)
                    max_gen_overall = max(max_gen_overall, max(generations) if generations else 0)

                    color = group_to_color.get(group, (0.5, 0.5, 0.5))

                    group_plotted = False
                    for i in range(len(generations) - 1):
                        t0, t1 = generations[i], generations[i+1]
                        # if t1 != t0 + 1: continue # Optional: only plot consecutive steps

                        d0, d1 = distance_values[i], distance_values[i+1]
                        std_dev0 = std_values[i]

                        normalized_std = (std_dev0 - min_std) / std_range if std_range > 1e-9 else 0.5
                        linewidth = (1 + normalized_std) ** 3
                        linewidth = max(0.1, linewidth)

                        segment_label = f"Group {group}" if not group_plotted else None
                        ax.plot([t0, t1], [d0, d1], label=segment_label, color=color, linewidth=linewidth, alpha=0.5, solid_capstyle='round')
                        plotted_anything = True
                        if not group_plotted: group_plotted = True

        if not plotted_anything:
            logger.warning("No data points were plotted based on visibility settings and available distances.")
            plt.close(fig)
            return

        if actual_min_plotted != float('inf') and actual_max_plotted > actual_min_plotted:
            padding = (actual_max_plotted - actual_min_plotted) * 0.05
            if padding < 1e-6: padding = 0.1
            ax.set_ylim(bottom=max(0, actual_min_plotted - padding), top=actual_max_plotted + padding)
        elif actual_max_plotted >= 0:
             ax.set_ylim(bottom=0, top=max(1.0, actual_max_plotted * 1.1))
        else:
             ax.set_ylim(0, 1)

        if max_gen_overall >= 0:
             ax.set_xlim(-0.5, max_gen_overall + 0.5)
        else:
             ax.set_xlim(-0.5, 0.5)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

        ax.grid(True, which='major', linestyle='--', alpha=0.6)
        fig.tight_layout()
        try:
            fig.savefig(output_file, bbox_inches='tight', dpi=dpi)
            logger.info(f"Saved centroid distance drift plot to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save centroid distance drift plot to {output_file}: {e}")
        finally:
            plt.close(fig)

        logger.info(f"Finished plotting centroid distance drift.")




    def generate_centroid_closest_meme_table(self, num_generations: int = -1):
        """
        Generates a CSV table where rows are generations and columns are groups.
        Each cell contains the meme text from that generation and group which is
        semantically closest (Euclidean distance in embedding space) to the
        centroid of all memes within that group at that specific generation.

        Args:
            num_generations (int): Max history steps (generations) to include.
                                   Defaults to -1 (all available generations).
        """
        if not hasattr(self, 'embedding_model') or not self.embedding_model:
            logger.warning("Skipping generate_centroid_closest_meme_table: Embedding model not available.")
            return
        global emb_utils
        if 'emb_utils' not in globals():
             logger.error("emb_utils not found. Cannot calculate embeddings.")
             return

        G = self.graph_manager.graph
        if G.number_of_nodes() == 0:
            logger.warning("Skipping generate_centroid_closest_meme_table: Graph has no nodes.")
            return

        # 1. Collect meme texts grouped by (group, timestep)
        group_timestep_memes = defaultdict(lambda: defaultdict(list))
        max_timestep_overall = 0
        logger.info("Collecting memes per group and timestep...")
        for node_id in G.nodes():
            data = self.graph_manager.get_node_data(node_id)
            if hasattr(data, 'history') and data.history:
                group = getattr(data, 'group', 0) # Default group 0 if not specified
                for t, meme in enumerate(data.history):
                    if num_generations >= 0 and t >= num_generations:
                        break
                    # Store the meme text directly
                    group_timestep_memes[group][t].append(str(meme))
                    max_timestep_overall = max(max_timestep_overall, t)

        if not group_timestep_memes:
            logger.warning("Skipping generate_centroid_closest_meme_table: No historical meme data found.")
            return

        # 2. Calculate centroids and find closest meme for each (group, timestep)
        centroid_closest_memes = defaultdict(dict) # {group: {timestep: closest_meme_text}}
        logger.info("Calculating centroids and finding closest memes...")
        for group, timesteps_data in group_timestep_memes.items():
            for timestep, meme_texts in timesteps_data.items():
                if not meme_texts:
                    continue # Skip if no memes for this group/timestep

                # Calculate embeddings for all memes in this cell
                embeddings = emb_utils.calculate_sentence_embeddings(meme_texts, self.embedding_model)

                if embeddings is None or embeddings.size == 0:
                    logger.warning(f"Could not calculate embeddings for Group {group}, Timestep {timestep}.")
                    continue

                # Calculate the centroid
                centroid = np.mean(embeddings, axis=0)

                # Find the embedding (and corresponding text) closest to the centroid
                if embeddings.shape[0] == 1:
                    # If only one meme, it's trivially the closest
                    closest_idx = 0
                else:
                    distances = np.linalg.norm(embeddings - centroid, axis=1)
                    closest_idx = np.argmin(distances)

                closest_meme_text = meme_texts[closest_idx]
                centroid_closest_memes[group][timestep] = closest_meme_text

        if not centroid_closest_memes:
            logger.warning("Skipping generate_centroid_closest_meme_table: No closest memes could be determined (check embeddings).")
            return

        # 3. Prepare data for CSV output
        all_groups = sorted(list(centroid_closest_memes.keys()))
        if not all_groups:
             logger.warning("Skipping generate_centroid_closest_meme_table: No groups with data found.")
             return

        header = ["Generation"] + [f"Group_{g}" for g in all_groups]
        table_data = [header]

        logger.info("Constructing table data...")
        for t in range(max_timestep_overall + 1):
            row = [t] # First column is the generation number
            for group in all_groups:
                # Get the closest meme text for this group/timestep, or "" if none found
                cell_value = centroid_closest_memes[group].get(t, "")
                row.append(cell_value)
            table_data.append(row)

        # 4. Write to CSV
        output_file = self.vis_dir / "centroid_closest_memes_table.csv"
        try:
            logger.info(f"Writing centroid closest meme table to {output_file}")
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(table_data)
            logger.info(f"Successfully saved centroid closest meme table to {output_file}")
        except Exception as e:
            logger.error(f"Failed to write centroid closest meme table to {output_file}: {e}")

        logger.info("Finished generating centroid closest meme table.")