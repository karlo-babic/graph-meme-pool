import logging
import sys
from pathlib import Path
import time

from config_loader import load_config
from graph_manager import GraphManager
from llm_service import LLMService
from evolution_engine import EvolutionEngine
from visualizer import Visualizer

# --- Logging Setup ---
def setup_logging(config):
    log_level_str = config['logging'].get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = config['paths'].get('log_file')

    handlers = [logging.StreamHandler(sys.stdout)] # Always log to console
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w')) # Overwrite log file each run

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    # Suppress overly verbose logs from libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete.")
    return logger

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # Load Configuration
    config = load_config("config.yaml") # Or pass path via args

    # Setup Logging
    logger = setup_logging(config)
    logger.info("--- Starting Graph Meme Pool Simulation ---")
    #logger.debug(f"Configuration loaded: {config}") # Can be very verbose

    # Initialize Components
    try:
        graph_manager = GraphManager(config)
        llm_service = LLMService(config)
        # Load LLM model early (can take time)
        llm_service.load()

        evolution_engine = EvolutionEngine(graph_manager, llm_service, config)
        # Visualizer loads embedding models internally if needed
        visualizer = Visualizer(graph_manager, config)

    except Exception as e:
        logger.critical(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)

    # Create or Load Initial Graph
    graph_base_name = config['paths']['graph_basename']
    graph_load_path = Path(config['paths']['graph_save_dir']) / f"{graph_base_name}.json"

    if graph_load_path.exists():
         logger.info(f"Attempting to load existing graph from {graph_load_path}")
         try:
             graph_manager.load_graph(graph_base_name)
             # TODO: Decide if I want to load propagation history too if it exists
         except Exception as e:
             logger.error(f"Failed to load graph: {e}. Creating a new graph instead.")
             graph_manager.create_graph()
    else:
         logger.info("No existing graph found. Creating a new graph.")
         graph_manager.create_graph()


    # --- Run Simulation ---
    try:
         # Initial visualization (optional)
         if config['simulation'].get('initial_score', True):
              logger.info("Performing initial scoring before simulation run...")
              evolution_engine.initialize_scores()
              # Visualize state *after* initial scoring
              if config['visualization'].get('draw_score_per_gen', False):
                   visualizer.draw_score(generation=0) # Label as gen 0
              if config['visualization'].get('draw_change_per_gen', False):
                   visualizer.draw_change(generation=0) # Label as gen 0
              if config['visualization'].get('draw_semantic_diff_per_gen', False):
                   visualizer.draw_semantic_difference(generation=0) # Label as gen 0


         logger.info("Starting evolution loop...")
         simulation_generator = evolution_engine.run_simulation()

         for completed_generation_index in simulation_generator:
              logger.info(f"Completed Generation {completed_generation_index + 1}.")
              # Per-generation visualization calls controlled here
              if config['visualization'].get('draw_score_per_gen', False):
                   visualizer.draw_score(generation=completed_generation_index + 1)
              if config['visualization'].get('draw_change_per_gen', False):
                   visualizer.draw_change(generation=completed_generation_index + 1)
              if config['visualization'].get('draw_semantic_diff_per_gen', False):
                   visualizer.draw_semantic_difference(generation=completed_generation_index + 1)

         logger.info("Simulation loop finished.")

    except Exception as e:
         logger.critical(f"Simulation failed during execution: {e}", exc_info=True)
         # Save state even if simulation fails mid-way?
         logger.info("Attempting to save graph state after error...")
         graph_manager.save_graph(graph_base_name + "_error_state")
         graph_manager.save_propagation_history(graph_base_name + "_error_state_propagation")
         sys.exit(1)


    # --- Final Actions ---
    logger.info("Performing final actions...")

    # Final Visualizations
    if config['visualization'].get('draw_final_embs', True):
        logger.info("Generating final embedding visualization...")
        visualizer.draw_embs()
    if config['visualization'].get('plot_final_score_history', True):
        logger.info("Generating final score history plot...")
        visualizer.plot_score_history_bygroup()
    # Add calls to other final visualizations if needed (e.g., final score/change plots)
    # visualizer.draw_score() # Final score plot without generation number
    # visualizer.draw_change(generation=config['simulation']['generations'], history_lookback=10) # Final change plot


    # Save Final Graph State and History
    logger.info("Saving final graph state and propagation history...")
    graph_manager.save_graph(graph_base_name)
    graph_manager.save_propagation_history() # Saves with default naming convention

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"--- Simulation Finished ---")
    logger.info(f"Total execution time: {total_time:.2f} seconds")