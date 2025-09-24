import json
import argparse
from pathlib import Path
import logging
import sys

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

def get_graph_state(snapshot_data):
    """Extracts node and edge sets from a snapshot for efficient diffing."""
    nodes = {node['id']: node['data'] for node in snapshot_data.get('nodes', [])}
    edges = {
        (edge['source'], edge['target']): edge.get('weight', 1.0)
        for edge in snapshot_data.get('links', [])
    }
    return nodes, edges

def format_node_for_replay(node_id, node_data):
    """Formats node data for the Cytoscape.js replay file."""
    return {
        "data": {
            "id": str(node_id),
            "meme": node_data.get('current_meme', ''),
            "score": node_data.get('current_meme_score'),
            "group": node_data.get('group'),
            "creationGen": node_data.get('creation_generation', 0)
        }
    }

def generate_replay(experiment_dir: Path):
    """
    Processes a series of per-generation graph snapshots to generate a replay file
    for web-based visualization by diffing consecutive states.
    """
    logging.info(f"Starting replay generation for experiment: {experiment_dir}")

    snapshots_dir = experiment_dir / "snapshots"
    prop_history_file = experiment_dir / "checkpoints" / "propagation_history.json"
    output_file = experiment_dir / "simulation_replay.json"

    if not snapshots_dir.is_dir():
        logging.error(f"Snapshots directory not found at: {snapshots_dir}")
        sys.exit(1)
    
    all_files = list(snapshots_dir.glob('gen_*.json'))
    snapshot_files = sorted(all_files, key=lambda f: int(f.stem.split('_')[1]))
    if not snapshot_files:
        logging.error(f"No snapshot files found in {snapshots_dir}")
        sys.exit(1)

    # 1. Load Initial State from the first snapshot
    with open(snapshot_files[0], 'r', encoding='utf-8') as f:
        initial_snapshot = json.load(f)
    
    initial_nodes_data, initial_edges_data = get_graph_state(initial_snapshot)
    
    initial_state = {
        "nodes": [format_node_for_replay(nid, ndata) for nid, ndata in initial_nodes_data.items()],
        "edges": [{"data": {"source": str(s), "target": str(t), "weight": w}} for (s, t), w in initial_edges_data.items()]
    }
    logging.info(f"Loaded initial state from {snapshot_files[0].name}")

    # 2. Generate Events by Diffing Snapshots
    events = []
    prev_nodes, prev_edges = initial_nodes_data, initial_edges_data

    for i in range(1, len(snapshot_files)):
        gen = i
        with open(snapshot_files[i], 'r', encoding='utf-8') as f:
            current_snapshot = json.load(f)
        
        curr_nodes, curr_edges = get_graph_state(current_snapshot)
        
        # Node changes
        prev_node_ids = set(prev_nodes.keys())
        curr_node_ids = set(curr_nodes.keys())
        
        added_nodes = curr_node_ids - prev_node_ids
        removed_nodes = prev_node_ids - curr_node_ids
        
        for node_id in removed_nodes:
            events.append({"gen": gen, "type": "NODE_REMOVE", "nodeId": str(node_id)})
            
        for node_id in added_nodes:
            events.append({"gen": gen, "type": "NODE_ADD", "node": format_node_for_replay(node_id, curr_nodes[node_id])})

        # Edge changes
        added_edges = set(curr_edges.keys()) - set(prev_edges.keys())
        removed_edges = set(prev_edges.keys()) - set(curr_edges.keys())

        for source, target in removed_edges:
            events.append({"gen": gen, "type": "EDGE_REMOVE", "source": str(source), "target": str(target)})

        for source, target in added_edges:
            events.append({"gen": gen, "type": "EDGE_ADD", "edge": {"source": str(source), "target": str(target), "weight": curr_edges[(source, target)]}})
            
        # Meme/Score updates for existing nodes
        for node_id in prev_node_ids.intersection(curr_node_ids):
            if prev_nodes[node_id]['current_meme'] != curr_nodes[node_id]['current_meme']:
                 events.append({
                    "gen": gen,
                    "type": "MEME_UPDATE",
                    "nodeId": str(node_id),
                    "newMeme": curr_nodes[node_id]['current_meme'],
                    "newScore": curr_nodes[node_id]['current_meme_score']
                })
        
        prev_nodes, prev_edges = curr_nodes, curr_edges
    
    # 3. Add Propagation Events
    if prop_history_file.exists():
        with open(prop_history_file, 'r', encoding='utf-8') as f:
            prop_history_data = json.load(f)
        for prop in prop_history_data:
            # Propagation happens *during* a generation, so we add it to the start of the next gen's events
            events.append({
                "gen": prop['generation'] + 1,
                "type": "PROPAGATION",
                "source": str(prop['source_node']),
                "target": str(prop['target_node'])
            })

    # 4. Sort events and save
    events.sort(key=lambda x: x['gen'])
    logging.info(f"Generated a total of {len(events)} simulation events.")

    replay_data = {
        "initialState": initial_state,
        "events": events
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(replay_data, f, indent=2)

    logging.info(f"Successfully created replay file at: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a replay JSON file from GMP simulation snapshots.")
    parser.add_argument(
        'experiment_dir',
        type=str,
        help='Path to the experiment directory (e.g., experiments/20240521-103000_my_run/).'
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_dir():
        logging.error(f"Error: Provided path '{exp_dir}' is not a valid directory.")
        sys.exit(1)
        
    generate_replay(exp_dir)