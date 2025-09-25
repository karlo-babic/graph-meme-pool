import logging
import sys
from pathlib import Path
import time
import random
import numpy as np
import torch
import argparse
import shutil

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
    NodeFusionAction, NodeDivisionAction, NodeDeathAction, EdgeRewireAction
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

def setup_logging(log_file: Path, config: dict, is_resuming: bool):
    log_level_str = config['logging']['level'].upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_mode = 'a' if is_resuming else 'w'
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode=file_mode)]
    
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
    dyn_config = config.get('dynamic_graph', {})
    if not dyn_config.get('enabled', False) or not dyn_config.get('actions'):
        return NullDynamicsStrategy()

    action_objects = []
    for action_conf in dyn_config['actions']:
        if not action_conf.get('enabled', False): continue
        action_type = action_conf.get('type')
        if action_type == 'fusion':
            action_objects.append(NodeFusionAction(action_conf, embedding_manager))
        elif action_type == 'division':
            action_objects.append(NodeDivisionAction(action_conf, llm_service))
        elif action_type == 'death':
            action_objects.append(NodeDeathAction(action_conf))
        elif action_type == 'rewire': # Use 'rewire' as the type in config
            action_objects.append(EdgeRewireAction(action_conf))
        else:
            logging.getLogger(__name__).warning(f"Unknown dynamic action type '{action_type}'. Skipping.")
    
    return CompositeDynamicsStrategy(action_objects, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Graph Meme Pool simulation.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str, help='Path to a configuration file for a new run.')
    group.add_argument('--resume', type=str, help='Path to an experiment directory to resume.')
    args = parser.parse_args()

    # --- Determine Run Mode and Setup Paths/Config ---
    if args.resume:
        experiment_dir = Path(args.resume)
        if not experiment_dir.is_dir():
            print(f"Error: Resume path '{experiment_dir}' is not a valid directory.")
            sys.exit(1)
        
        config_path = experiment_dir / "config.yaml"
        if not config_path.exists():
            print(f"Error: Cannot find 'config.yaml' in resume directory '{experiment_dir}'.")
            sys.exit(1)
            
        is_resuming = True
        config = load_config(config_path)
        run_name = config.get('run_name', experiment_dir.name)
        
    else: # Start a new run
        config = load_config(args.config)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        run_name = config.get('run_name', 'unnamed_run')
        experiment_dir = Path("experiments") / f"{timestr}_{run_name}"
        is_resuming = False

    checkpoints_dir = experiment_dir / "checkpoints"
    visualizations_dir = experiment_dir / "visualizations"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    snapshots_dir = experiment_dir / "snapshots"
    if not is_resuming:
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup Logging and Save Config for New Runs ---
    log_file_path = experiment_dir / "simulation.log"
    logger = setup_logging(log_file_path, config, is_resuming)
    
    if not is_resuming:
        shutil.copy(args.config, experiment_dir / "config.yaml")

    logger.info(f"--- Graph Meme Pool Simulation: {run_name} ---")
    logger.info(f"Run mode: {'Resuming' if is_resuming else 'New Run'}")
    logger.info(f"Experiment outputs saved to: {experiment_dir.resolve()}")

    start_time = time.time()
    set_global_seed(config['seed'])

    try:
        # --- Initialize Services (same for both modes) ---
        embedding_manager = emb_utils.EmbeddingManager(config['embeddings']['model_path'])
        llm_service = LLMService(config)
        llm_service.load()
        graph_persistence = GraphPersistence()
        graph_basename = "graph_state"
        graph_filepath = checkpoints_dir / f"{graph_basename}_final.json"
        history_filepath = checkpoints_dir / "propagation_history.json"

        # --- Load or Create Graph State ---
        if is_resuming:
            graph_obj, start_gen_idx, dynamics_state, graveyard, avg_initial_wc = graph_persistence.load_graph(graph_filepath)
            if graph_obj is None:
                logger.critical("Failed to load graph state for resume. Aborting.")
                sys.exit(1)
            start_generation_index = start_gen_idx + 1
        else:
            start_generation_index, dynamics_state, graveyard, avg_initial_wc = 0, {}, {}, None
            gen_type = config['graph_generation']['type']
            initializer = ExampleGraphInitializer(config) if gen_type == 'example' else SmallWorldsInitializer(config)
            initial_memes = initializer._load_initial_memes()
            graph_obj = initializer.create(initial_memes)

        graph_manager = GraphManager(graph_obj, graveyard)
        prop_history = graph_persistence.load_propagation_history(history_filepath)
        graph_manager.set_propagation_history(prop_history)
        
        # --- Initialize Remaining Components ---
        dynamics_strategy = build_dynamics_strategy(config, embedding_manager, llm_service)
        dynamics_strategy.set_state(dynamics_state) # Restore state after building
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
            dynamics_strategy=dynamics_strategy, # Pass the strategy object
            fitness_model=fitness_model_instance,
            avg_initial_word_count=avg_initial_wc
        )
        visualizer = Visualizer(graph_manager, config, embedding_manager, visualizations_dir)

    except Exception as e:
        logger.critical(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)

    # --- Run Simulation ---
    last_completed_gen_in_run = -1
    try:
        # Save the initial state (generation -1 or 0) before the loop starts
        if start_generation_index == 0:
            if config['simulation']['initial_score']: evolution_engine.initialize_scores()
            # Save the state before any propagations, as generation 0
            initial_snapshot_path = snapshots_dir / "gen_0000.json"
            graph_persistence.save_graph_snapshot(graph_manager.get_graph(), initial_snapshot_path)

        simulation_generator = evolution_engine.run_simulation(start_generation_index=start_generation_index)
        
        for completed_generation_index in simulation_generator:
            last_completed_gen_in_run = completed_generation_index
            
            # --- Save Snapshot of the graph state after this generation ---
            snapshot_filename = f"gen_{completed_generation_index + 1:04d}.json"
            snapshot_filepath = snapshots_dir / snapshot_filename
            graph_persistence.save_graph_snapshot(graph_manager.get_graph(), snapshot_filepath)
            
            if visualizer:
                if config['visualization']['draw_score_per_gen']: visualizer.draw_score(generation=completed_generation_index)
                if config['visualization']['draw_change_per_gen']: visualizer.draw_change(generation=completed_generation_index)
                if config['visualization']['plot_score_history_individual']: visualizer.plot_score_history_individual_nodes()

    except Exception as e:
        logger.critical(f"Simulation failed during execution: {e}", exc_info=True)
        if graph_manager:
            error_path = checkpoints_dir / f"{graph_basename}_error_state.json"
            dynamics_state = graph_manager.dynamics_strategy.get_state()
            avg_wc = evolution_engine.get_avg_initial_word_count()
            graph_persistence.save_graph(graph_manager.get_graph(), graph_manager.get_graveyard(), error_path, last_completed_gen_in_run, dynamics_state, avg_wc)
        sys.exit(1)

    # --- Final Actions ---
    logger.info("Performing final actions...")
    if graph_manager:
        dynamics_state = graph_manager.dynamics_strategy.get_state()
        avg_wc = evolution_engine.get_avg_initial_word_count()
        graph_persistence.save_graph(graph_manager.get_graph(), graph_manager.get_graveyard(), graph_filepath, last_completed_gen_in_run, dynamics_state, avg_wc)
        graph_persistence.save_propagation_history(graph_manager.get_propagation_history(), history_filepath)
        
    if visualizer:
        if config['visualization']['plot_final_score_history_by_group']: visualizer.plot_score_history_bygroup()
        if config['visualization']['plot_score_history_by_initial_meme']: visualizer.plot_score_history_by_initial_meme()
        if config['visualization']['draw_final_embs']: visualizer.draw_embs()
        if config['visualization']['plot_semantic_drift']: visualizer.draw_semantic_drift()
        if config['visualization']['plot_semantic_centroid_distance_drift']: visualizer.plot_semantic_centroid_distance_drift()

    total_time = time.time() - start_time
    logger.info(f"--- Simulation Finished --- Total execution time: {total_time:.2f} seconds")