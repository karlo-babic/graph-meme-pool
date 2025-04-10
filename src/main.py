import logging
import sys
from pathlib import Path
import time

from config_loader import load_config
from graph_manager import GraphManager
from llm_service import LLMService
from evolution_engine import EvolutionEngine
from visualizer import Visualizer
from fitness_model import FitnessModel

# --- Logging Setup ---
def setup_logging(config):
    log_level_str = config['logging']['level'].upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = config['paths']['log_file']

    handlers = [logging.StreamHandler(sys.stdout)] # Always log to console
    if log_file:
        # Ensure log directory exists if log_file path includes directories
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w')) # Overwrite log file each run

    # Remove existing handlers before adding new ones
    # (Prevents duplicate logs if script is run multiple times in the same session/notebook)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    # Suppress overly verbose logs from libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    # Suppress transformers info unless logging level is DEBUG
    if log_level > logging.DEBUG:
        logging.getLogger("transformers").setLevel(logging.WARNING)


    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete.")
    return logger


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # Load Configuration
    config = load_config("config.yaml") # Use default name

    # Setup Logging
    logger = setup_logging(config)
    logger.info("--- Starting Graph Meme Pool Simulation ---")
    logger.info(f"Configuration loaded from: config.yaml (or defaults)")

    # --- Initialize Components ---
    graph_manager = None
    llm_service = None
    fitness_model_instance = None
    evolution_engine = None
    visualizer = None
    start_generation_index = 0 # Default for new simulation
    
    # Create or Load Initial Graph
    graph_manager = GraphManager(config)
    graph_base_name = config['paths']['graph_basename']
    graph_load_path = Path(config['paths']['graph_save_dir']) / f"{graph_base_name}.json"

    if graph_load_path.exists():
        logger.info(f"Attempting to load existing graph from {graph_load_path}")
        try:
            graph_manager.load_graph(graph_base_name)
            start_generation_index = graph_manager.loaded_last_generation + 1
            logger.info(f"Resuming simulation from generation index {start_generation_index}")
        except Exception as e:
            logger.error(f"Failed to load graph: {e}. Creating a new graph instead.")
            graph_manager.create_graph()
            start_generation_index = 0 # Reset for new graph
    else:
        logger.info("No existing graph found. Creating a new graph.")
        graph_manager.create_graph()
        start_generation_index = 0 # Reset for new graph


    try:
        # Always load LLM Service (needed for mutate/merge)
        logger.info("Initializing LLM Service...")
        llm_service = LLMService(config)
        llm_service.load() # Load LLM model early

        # Conditionally load Fitness Model
        fitness_model_huggingpath = config['simulation']['fitness_model_huggingpath'].lower()
        logger.info(f"Fitness model selected: '{fitness_model_huggingpath if fitness_model_huggingpath else 'LLM'}'")

        if fitness_model_huggingpath:
            logger.info(f"Initializing Fitness Model from path: {fitness_model_huggingpath}")
            try:
                fitness_model_instance = FitnessModel(model_huggingpath=fitness_model_huggingpath)
                fitness_model_instance.load() # Load the fitness model
            except Exception as fm_error:
                logger.error(f"Failed to load Fitness Model: {fm_error}. Simulation may proceed using only LLM for scoring if possible, or fail.", exc_info=True)
                sys.exit(1)


        # --- Pass BOTH services (or None for fitness_model) to EvolutionEngine ---
        logger.info("Initializing Evolution Engine...")
        evolution_engine = EvolutionEngine(
            graph_manager=graph_manager,
            llm_service=llm_service,
            config=config,
            fitness_model=fitness_model_instance # Pass the instance (or None)
        )

        # Visualizer loads embedding models internally if needed
        logger.info("Initializing Visualizer...")
        visualizer = Visualizer(graph_manager, config)

    except Exception as e:
        logger.critical(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)


    # --- Run Simulation ---
    last_completed_gen_in_run = -1
    try:
        if start_generation_index == 0:
            logger.info("Performing initial setup actions for new simulation.")
            # Check and potentially mutate initial uniform state
            evolution_engine.mutate_initial_if_all_same()

            # Initial visualization (optional)
            if config['simulation']['initial_score']:
                logger.info("Performing initial scoring before simulation run...")
                evolution_engine.initialize_scores()
                if visualizer:
                    vis_gen_label = start_generation_index
                    if True: #config['visualization']['draw_score_per_gen']:
                        visualizer.draw_score(generation=vis_gen_label)
                    if config['visualization']['draw_change_per_gen']:
                        visualizer.draw_change(generation=vis_gen_label, history_lookback=4)
                    if config['visualization']['draw_semantic_diff_per_gen']:
                        visualizer.draw_semantic_difference(generation=vis_gen_label)
                else:
                    logger.warning("Visualizer not initialized, skipping initial visualizations.")
        else:
            logger.info(f"Skipping initial setup (mutate_all, initial_score) as simulation resumes from generation {start_generation_index + 1}.")


        logger.info("Starting evolution loop...")
        num_generations_to_run_config = config['simulation']['generations']
        if num_generations_to_run_config <= 0:
            logger.warning("Configured number of generations to run is <= 0. No simulation steps will execute.")
            simulation_generator = iter([]) # Empty iterator
        else:
            logger.info(f"Starting evolution loop to run {num_generations_to_run_config} generations...")
            simulation_generator = evolution_engine.run_simulation(start_generation_index=start_generation_index)

        for completed_generation_index in simulation_generator:
            last_completed_gen_in_run = completed_generation_index
            # Per-generation visualization calls controlled here
            if visualizer:
                generation_num_for_vis = completed_generation_index + 1
                if config['visualization']['draw_score_per_gen']:
                    visualizer.draw_score(generation=generation_num_for_vis)
                if config['visualization']['draw_change_per_gen']:
                    # Draw change might need history, pass lookback parameter if needed
                    visualizer.draw_change(generation=generation_num_for_vis, history_lookback=4) # Example lookback
                if config['visualization']['draw_semantic_diff_per_gen']:
                    visualizer.draw_semantic_difference(generation=generation_num_for_vis)


    except Exception as e:
        logger.critical(f"Simulation failed during execution: {e}", exc_info=True)
        # Save state even if simulation fails mid-way
        if graph_manager:
            logger.info("Attempting to save graph state after error...")
            try:
                graph_manager.save_graph(graph_base_name + "_error_state", last_completed_generation=last_completed_gen_in_run)
                graph_manager.save_propagation_history(graph_base_name + "_error_state_propagation")
            except Exception as save_err:
                logger.error(f"Could not save error state: {save_err}")
        sys.exit(1)


    # --- Final Actions ---
    logger.info("Performing final actions...")

    # Save Final Graph State and History
    if graph_manager: # Check if initialized
        logger.info("Saving final graph state and propagation history...")
        try:
            graph_manager.save_graph(graph_base_name, last_completed_generation=last_completed_gen_in_run)
            graph_manager.save_propagation_history() # Saves with default naming convention
        except Exception as final_save_err:
            logger.error(f"Failed to save final graph state/history: {final_save_err}")

    # Final Visualizations
    if visualizer: # Check if initialized
        if config['visualization']['draw_final_embs']:
            logger.info("Generating final embedding visualization...")
            visualizer.draw_embs()
        if config['visualization']['plot_final_score_history']:
            logger.info("Generating final score history plot...")
            visualizer.plot_score_history_bygroup()
        if config['visualization']['plot_semantic_drift']:
            logger.info("Generating semantic drift plots...")
            visualizer.draw_semantic_drift()
        if config['visualization']['plot_semantic_centroid_distance_drift']:
            logger.info("Generating semantic centroid drift plots...")
            visualizer.plot_semantic_centroid_distance_drift()
        # Add calls to other final visualizations if needed
        # visualizer.draw_score() # Final score plot without generation number
        # visualizer.draw_change(generation=evolution_engine.config['generations'], history_lookback=10) # Final change plot
    else:
        logger.warning("Visualizer not initialized, skipping final visualizations.")

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"--- Simulation Finished ---")
    logger.info(f"Total execution time: {total_time:.2f} seconds")