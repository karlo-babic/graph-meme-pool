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
import embeddings_utils as emb_utils

def setup_logging(config):
    log_level_str = config['logging']['level'].upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = config['paths']['log_file']

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    if log_level > logging.DEBUG:
        logging.getLogger("transformers").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete.")
    return logger

if __name__ == "__main__":
    start_time = time.time()

    config = load_config("config.yaml")
    logger = setup_logging(config)
    logger.info("--- Starting Post-Simulation Visualization ---")
    logger.info(f"Configuration loaded from: config.yaml (or defaults)")

    graph_manager = None
    visualizer = None

    try:
        logger.info("Initializing components for visualization...")
        embedding_manager = emb_utils.EmbeddingManager(config['embeddings']['model_path'])
        llm_service = LLMService(config)
        graph_manager = GraphManager(config, embedding_manager, llm_service)
    except Exception as e:
        logger.critical(f"Failed to initialize necessary components: {e}", exc_info=True)
        sys.exit(1)

    graph_base_name = config['paths']['graph_basename']
    graph_load_path = Path(config['paths']['graph_save_dir']) / f"{graph_base_name}.json"

    if graph_load_path.exists():
        logger.info(f"Attempting to load existing graph from {graph_load_path}")
        try:
            graph_manager.load_graph(graph_base_name)
        except Exception as e:
            logger.error(f"Failed to load graph: {e}. Cannot proceed with visualization.", exc_info=True)
            sys.exit(1)
    else:
        logger.error(f"No existing graph found at {graph_load_path}. Quitting.")
        sys.exit(1)

    try:
        visualizer = Visualizer(graph_manager, config, embedding_manager)
    except Exception as e:
        logger.critical(f"Failed to initialize Visualizer: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info("Generating selected visualizations...")
    
    visualizer.draw_score()
    #visualizer.plot_score_history_bygroup()
    #visualizer.plot_score_history_by_initial_meme()
    #visualizer.draw_semantic_drift()
    #visualizer.plot_semantic_centroid_distance_drift()
    #visualizer.generate_centroid_closest_meme_table()
    #visualizer.draw_embs()
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"--- Visualization Finished ---")
    logger.info(f"Total execution time: {total_time:.2f} seconds")