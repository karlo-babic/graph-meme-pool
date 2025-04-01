import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    # Define sensible defaults here in case keys are missing in YAML
    'paths': {
        'init_memes': "init_memes.txt",
        'graph_save_dir': "saved",
        'graph_basename': "graph_output",
        'vis_dir': "vis",
        'log_file': "simulation.log"
    },
    'graph_generation': {
        'type': 'small_world',
        'initial_meme_assignment': 'random',
        'params': {'n': 20, 'k': 4, 'p': 0.3, 'b': 0.3, 'g': 2, 'inter_p': 0.1}
    },
    'simulation': {
        'generations': 50,
        'threshold': 0.2,
        'initial_score': True,
        'fitness_model': 'llm',
        'fitness_model_huggingpath': ''
    },
    'llm': {
        'huggingpath': "microsoft/Phi-3-mini-4k-instruct",
        'temperature_mutate': 0.5,
        'temperature_merge': 0.5,
        'temperature_score': 0.2,
        'max_new_tokens': 100
    },
    'embeddings': {
        'model_path': "bert-base-uncased"
    },
    'visualization': {
        'draw_change_per_gen': True,
        'draw_score_per_gen': True,
        'draw_semantic_diff_per_gen': True,
        'draw_final_embs': True,
        'plot_final_score_history': True,
        'node_min_size': 20,
        'node_max_size': 400,
        'edge_base_thickness': 1.0,
        'edge_base_opacity': 0.5,
        'label_max_len': 64,
        'dpi': 150
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
            config = _recursive_update(config, yaml_config)
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using default configuration.")
    except Exception as e:
        print(f"Error loading config file '{config_path}': {e}. Using default configuration.")

    # Ensure directories exist
    Path(config['paths']['graph_save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['vis_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['vis_dir'], "graph_semantic_difference").mkdir(parents=True, exist_ok=True)

    return config