import networkx as nx
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from networkx.readwrite import json_graph

from data_structures import MemeNodeData, PropagationEvent

logger = logging.getLogger(__name__)

class GraphPersistence:
    """Handles saving and loading the graph state and propagation history."""

    def _serialize_node_data_dict(self, data_dict: Dict) -> Dict:
        """Helper to convert numpy types in a dict to JSON-serializable formats."""
        for key, value in data_dict.items():
            if isinstance(value, list):
                # Process lists, which might contain scores
                data_dict[key] = [
                    float(v) if isinstance(v, np.floating) else
                    int(v) if isinstance(v, np.integer) else v
                    for v in value if v is not None # Keep None as is
                ]
            elif isinstance(value, np.floating):
                data_dict[key] = float(value)
            elif isinstance(value, np.integer):
                data_dict[key] = int(value)
        return data_dict

    def save_graph(self, graph: nx.DiGraph, graveyard: Dict[Any, MemeNodeData], filepath: Path, last_completed_generation: int, dynamics_state: Dict, avg_initial_word_count: Optional[float]):
        """
        Saves the complete simulation state, including the active graph and the
        graveyard of removed nodes, into a single archive file.
        """
        try:
            # 1. Prepare metadata
            metadata = {
                'last_completed_generation': last_completed_generation,
                'dynamics_state': dynamics_state,
                'avg_initial_word_count': avg_initial_word_count
            }

            # 2. Prepare active graph data for serialization
            serializable_graph_data = json_graph.node_link_data(graph)
            for node_dict in serializable_graph_data.get('nodes', []):
                if 'data' in node_dict and isinstance(node_dict['data'], MemeNodeData):
                    # Convert MemeNodeData object to a clean, serializable dict
                    data_as_dict = node_dict['data'].__dict__
                    node_dict['data'] = self._serialize_node_data_dict(data_as_dict)

            # 3. Prepare graveyard data for serialization
            serializable_graveyard = {}
            for node_id, node_data in graveyard.items():
                data_as_dict = node_data.__dict__
                serializable_graveyard[str(node_id)] = self._serialize_node_data_dict(data_as_dict)

            # 4. Combine all parts into a final archive object
            archive_data = {
                'metadata': metadata,
                'active_graph': serializable_graph_data,
                'graveyard': serializable_graveyard
            }

            # 5. Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(archive_data, f, indent=2)
            logger.info(f"Complete simulation archive saved successfully to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save simulation archive to {filepath}: {e}", exc_info=True)

    def load_graph(self, filepath: Path) -> Tuple[Optional[nx.DiGraph], int, Dict, Optional[Dict[Any, MemeNodeData]], Optional[float]]:
        """
        Loads a complete simulation state from an archive file, reconstructing
        the active graph and the graveyard.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                archive_data = json.load(f)

            # 1. Load metadata
            metadata = archive_data.get('metadata', {})
            last_gen = metadata.get('last_completed_generation', -1)
            dynamics_state = metadata.get('dynamics_state', {})
            avg_initial_word_count = metadata.get('avg_initial_word_count', None)

            # 2. Reconstruct the active graph
            graph_data = archive_data.get('active_graph', {})
            for node_dict in graph_data.get('nodes', []):
                if 'data' in node_dict and isinstance(node_dict['data'], dict):
                    # Convert dict back to MemeNodeData object
                    node_dict['data'] = MemeNodeData(**node_dict['data'])
            
            graph = json_graph.node_link_graph(graph_data, directed=True, multigraph=False)

            # 3. Reconstruct the graveyard
            graveyard_data = archive_data.get('graveyard', {})
            graveyard_objects = {}
            for node_id_str, data_dict in graveyard_data.items():
                node_id = int(node_id_str) # JSON keys must be strings
                graveyard_objects[node_id] = MemeNodeData(**data_dict)

            logger.info(f"Archive loaded from {filepath}. Resuming from generation {last_gen + 1}.")
            return graph, last_gen, dynamics_state, graveyard_objects, avg_initial_word_count

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load or parse archive from {filepath}: {e}")
            return None, -1, {}, None, None

    def save_propagation_history(self, history: List[PropagationEvent], filepath: Path):
        if not history: return
        try:
            serializable_history = [event.__dict__ for event in history]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_history, f, indent=2)
            logger.info(f"Propagation history saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save propagation history to {filepath}: {e}")

    def load_propagation_history(self, filepath: Path) -> List[PropagationEvent]:
        history = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            for event_dict in history_data:
                history.append(PropagationEvent(**event_dict))
            logger.info(f"Successfully loaded {len(history)} propagation events.")
        except FileNotFoundError:
            logger.warning(f"Propagation history file {filepath} not found. Starting with empty history.")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to load or process propagation history from {filepath}: {e}")
        return history

    def save_graph_snapshot(self, graph: nx.DiGraph, filepath: Path):
        """
        Saves a snapshot of the current graph state to a JSON file.
        This is a lightweight version for per-generation replay data.
        """
        try:
            # Prepare graph data for serialization, converting MemeNodeData to a dict
            serializable_graph_data = json_graph.node_link_data(graph)
            for node_dict in serializable_graph_data.get('nodes', []):
                if 'data' in node_dict and isinstance(node_dict['data'], MemeNodeData):
                    data_as_dict = node_dict['data'].__dict__
                    node_dict['data'] = self._serialize_node_data_dict(data_as_dict)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_graph_data, f) # No indent for smaller file size
            
        except Exception as e:
            logger.error(f"Failed to save graph snapshot to {filepath}: {e}", exc_info=True)