import networkx as nx
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from networkx.readwrite import json_graph

from data_structures import MemeNodeData, PropagationEvent

logger = logging.getLogger(__name__)

class GraphPersistence:
    """Handles saving and loading the graph state and propagation history."""

    def save_graph(self, graph: nx.DiGraph, filepath: Path, last_completed_generation: int):
        try:
            graph.graph['last_completed_generation'] = last_completed_generation
            serializable_data = json_graph.node_link_data(graph)
            
            for node_dict in serializable_data.get('nodes', []):
                if 'data' in node_dict and isinstance(node_dict['data'], MemeNodeData):
                    data_dict = node_dict['data'].__dict__
                    for key, value in data_dict.items():
                        if isinstance(value, list):
                            data_dict[key] = [float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for v in value]
                        elif isinstance(value, np.floating): data_dict[key] = float(value)
                        elif isinstance(value, np.integer): data_dict[key] = int(value)
                    node_dict['data'] = data_dict
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=2)
            logger.info(f"Graph saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save graph to {filepath}: {e}", exc_info=True)

    def load_graph(self, filepath: Path) -> Tuple[Optional[nx.DiGraph], int]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            for node_data_dict in data.get('nodes', []):
                if 'data' in node_data_dict and isinstance(node_data_dict['data'], dict):
                    node_id = node_data_dict.get('id')
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

            graph = json_graph.node_link_graph(data, directed=True, multigraph=False)
            last_gen = graph.graph.get('last_completed_generation', -1)
            logger.info(f"Graph loaded successfully from {filepath}, completed up to generation {last_gen}")
            return graph, last_gen
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load graph from {filepath}: {e}")
            return None, -1

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