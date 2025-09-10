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





start_time = time.time()


# Load Configuration
config = load_config("config.yaml") # Use default name

# Setup Logging
logger = setup_logging(config)
logger.info(f"Configuration loaded from: config.yaml (or defaults)")

# --- Initialize Components ---
graph_manager = None
visualizer = None

# Create or Load Initial Graph
graph_manager = GraphManager(config)
graph_base_name = config['paths']['graph_basename']
graph_load_path = Path(config['paths']['graph_save_dir']) / f"{graph_base_name}.json"

if graph_load_path.exists():
    logger.info(f"Attempting to load existing graph from {graph_load_path}")
    try:
        graph_manager.load_graph(graph_base_name)
        start_generation_index = graph_manager.loaded_last_generation + 1
    except Exception as e:
        logger.error(f"Failed to load graph: {e}. Creating a new graph instead.")
        graph_manager.create_graph()
        start_generation_index = 0 # Reset for new graph
else:
    logger.info("No existing graph found. Quitting.")
    sys.quit()

visualizer = Visualizer(graph_manager, config)



#visualizer.plot_score_history_bygroup()
visualizer.plot_score_history_by_initial_meme()
#visualizer.draw_semantic_drift()
#visualizer.plot_semantic_centroid_distance_drift()
#visualizer.generate_centroid_closest_meme_table()
#visualizer.draw_embs()



end_time = time.time()
total_time = end_time - start_time
logger.info(f"--- Visualization Finished ---")
logger.info(f"Total execution time: {total_time:.2f} seconds")