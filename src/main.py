import logging
import sys
from pathlib import Path
import time
import random
import numpy as np
import torch

from config_loader import load_config
from llm_service import LLMService
from evolution_engine import EvolutionEngine
from visualizer import Visualizer
from fitness_model import FitnessModel
import embeddings_utils as emb_utils

from graph.graph_manager import GraphManager
from graph.graph_persistence import GraphPersistence
from graph.graph_initializer import SmallWorldsInitializer, ExampleGraphInitializer
from graph.graph_dynamics import (
    GraphDynamicsStrategy, NullDynamicsStrategy, CompositeDynamicsStrategy,
    NodeFusionAction, NodeDivisionAction
)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Global seed set to {seed}")

def setup_logging(config):
    log_level_str = config['logging']['level'].upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = config['paths']['log_file']
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    if log_level > logging.DEBUG: logging.getLogger("transformers").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete.")
    return logger

def build_dynamics_strategy(config: dict, embedding_manager: emb_utils.EmbeddingManager, llm_service: LLMService) -> GraphDynamicsStrategy:
    """Factory function to build the graph dynamics strategy from config."""
    dyn_config = config.get('dynamic_graph', {})
    if not dyn_config.get('enabled', False):
        return NullDynamicsStrategy()

    action_configs = dyn_config.get('actions', [])
    if not action_configs:
        return NullDynamicsStrategy()

    action_objects = []
    for action_conf in action_configs:
        action_type = action_conf.get('type')
        if not action_conf.get('enabled', False):
            continue

        if action_type == 'fusion':
            action_objects.append(NodeFusionAction(action_conf, embedding_manager))
        elif action_type == 'division':
            action_objects.append(NodeDivisionAction(action_conf, llm_service))
        else:
            logger.warning(f"Unknown dynamic graph action type '{action_type}' in config. Skipping.")
    
    return CompositeDynamicsStrategy(action_objects, config)


if __name__ == "__main__":
    start_time = time.time()
    config = load_config("config.yaml")
    set_global_seed(config['seed'])
    logger = setup_logging(config)
    logger.info("--- Starting Graph Meme Pool Simulation ---")

    try:
        logger.info("Initializing services (LLM, Embeddings)...")
        embedding_manager = emb_utils.EmbeddingManager(config['embeddings']['model_path'])
        llm_service = LLMService(config)
        llm_service.load()

        graph_persistence = GraphPersistence()
        graph_base_name = config['paths']['graph_basename']
        save_dir = Path(config['paths']['graph_save_dir'])
        graph_filepath = save_dir / f"{graph_base_name}.json"
        
        graph_obj, start_generation_index = graph_persistence.load_graph(graph_filepath)
        start_generation_index += 1

        if graph_obj is None:
            logger.info("No existing graph found or load failed. Creating a new graph.")
            start_generation_index = 0
            gen_type = config['graph_generation']['type']
            initializer = ExampleGraphInitializer(config) if gen_type == 'example' else SmallWorldsInitializer(config)
            initial_memes = initializer._load_initial_memes()
            graph_obj = initializer.create(initial_memes)

        graph_manager = GraphManager(graph_obj)
        history_filepath = save_dir / f"{graph_base_name}_propagation.json"
        prop_history = graph_persistence.load_propagation_history(history_filepath)
        graph_manager.set_propagation_history(prop_history)
        
        dynamics_strategy = build_dynamics_strategy(config, embedding_manager, llm_service)
        graph_manager.set_dynamics_strategy(dynamics_strategy)

        fitness_model_instance = None
        if config['simulation']['fitness_model_huggingpath']:
            fitness_model_instance = FitnessModel(model_huggingpath=config['simulation']['fitness_model_huggingpath'])
            fitness_model_instance.load()

        evolution_engine = EvolutionEngine(
            graph_manager=graph_manager,
            llm_service=llm_service,
            embedding_manager=embedding_manager,
            config=config,
            fitness_model=fitness_model_instance
        )
        visualizer = Visualizer(graph_manager, config, embedding_manager)

    except Exception as e:
        logger.critical(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)

    last_completed_gen_in_run = -1
    try:
        if start_generation_index == 0:
            logger.info("Performing initial setup for new simulation.")
            evolution_engine.mutate_initial_if_all_same()
            if config['simulation']['initial_score']:
                evolution_engine.initialize_scores()
                if visualizer:
                    if config['visualization']['draw_score_per_gen']: visualizer.draw_score(generation=0)
                    if config['visualization']['draw_change_per_gen']: visualizer.draw_change(generation=0, history_lookback=4)
        
        simulation_generator = evolution_engine.run_simulation(start_generation_index=start_generation_index)
        for completed_generation_index in simulation_generator:
            last_completed_gen_in_run = completed_generation_index
            if visualizer:
                if config['visualization']['draw_score_per_gen']: visualizer.draw_score(generation=completed_generation_index)
                if config['visualization']['draw_change_per_gen']: visualizer.draw_change(generation=completed_generation_index, history_lookback=4)

    except Exception as e:
        logger.critical(f"Simulation failed during execution: {e}", exc_info=True)
        if graph_manager:
            logger.info("Attempting to save graph state after error...")
            error_path = save_dir / f"{graph_base_name}_error_state.json"
            graph_persistence.save_graph(graph_manager.get_graph(), error_path, last_completed_gen_in_run)
        sys.exit(1)

    logger.info("Performing final actions...")
    if graph_manager:
        graph_persistence.save_graph(graph_manager.get_graph(), graph_filepath, last_completed_gen_in_run)
        graph_persistence.save_propagation_history(graph_manager.get_propagation_history(), history_filepath)

    if visualizer:
        if config['visualization']['plot_final_score_history_by_group']: visualizer.plot_score_history_bygroup()
        if config['visualization']['plot_score_history_by_initial_meme']: visualizer.plot_score_history_by_initial_meme()
        if config['visualization']['draw_final_embs']: visualizer.draw_embs()
        if config['visualization']['plot_semantic_drift']: visualizer.draw_semantic_drift()
        if config['visualization']['plot_semantic_centroid_distance_drift']: visualizer.plot_semantic_centroid_distance_drift()

    total_time = time.time() - start_time
    logger.info(f"--- Simulation Finished --- Total execution time: {total_time:.2f} seconds")