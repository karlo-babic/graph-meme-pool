import random
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np # For checking NaN

from graph_manager import GraphManager
from llm_service import LLMServiceInterface
from fitness_model import FitnessModel

logger = logging.getLogger(__name__)

class EvolutionEngine:
    """Orchestrates the meme evolution process over generations."""

    def __init__(self,
                 graph_manager: GraphManager,
                 llm_service: LLMServiceInterface,
                 config: Dict,
                 fitness_model: Optional[FitnessModel] = None):
        self.graph_manager = graph_manager
        self.llm_service = llm_service
        self.fitness_model = fitness_model # Store the fitness model instance
        self.config = config['simulation']
        self.llm_config = config['llm'] # For LLM-specific parameters
        self.fitness_model_huggingpath = self.config['fitness_model_huggingpath'].lower() # Store for easy access
        random.seed(config['seed'])

        # Validation
        if self.fitness_model_huggingpath and not self.fitness_model:
             logger.warning("EvolutionEngine initialized for 'custom' fitness type, but no FitnessModel instance provided. Scoring will likely fail.")
        elif not self.fitness_model_huggingpath and not self.llm_service:
             logger.critical("EvolutionEngine initialized for 'llm' fitness type, but no LLMService instance provided. Scoring will fail.")
             raise ValueError("LLMService instance is required for 'llm' fitness model type.")
        
        # Calculate average initial word count for penalty
        self.avg_initial_word_count: Optional[float] = self._calculate_avg_initial_word_count()
        if self.avg_initial_word_count is None:
            logger.warning("Could not determine average initial word count. Word count penalty will be disabled.")

    def _calculate_avg_initial_word_count(self) -> Optional[float]:
        """Calculates the average word count of the initial memes in the graph."""
        initial_memes = []
        try:
            all_nodes_data = self.graph_manager.get_all_nodes_data()
            if not all_nodes_data:
                 logger.warning("Cannot calculate average initial word count: No nodes found.")
                 return None

            for node_id, data in all_nodes_data.items():
                 if data:
                    # Use history[0] if available as the definitive initial state
                    initial_meme = data.history[0] if data.history else data.current_meme
                    if initial_meme and isinstance(initial_meme, str): # Ensure it's a non-empty string
                        initial_memes.append(initial_meme)
                    # else: logger.debug(f"Node {node_id} has no valid initial meme for word count.")

        except Exception as e:
             logger.error(f"Error accessing initial node data for word count: {e}", exc_info=True)
             return None # Cannot calculate safely

        if not initial_memes:
            logger.warning("Cannot calculate average initial word count: No valid initial memes found.")
            return None # No initial memes found

        word_counts = [len(meme.split()) for meme in initial_memes]
        if not word_counts:
             logger.warning("Cannot calculate average initial word count: Could not count words in any initial meme.")
             return None

        average = sum(word_counts) / len(word_counts)
        logger.info(f"Calculated average initial word count: {average:.2f} from {len(initial_memes)} memes.")
        return average

    def _score_memes(self, memes: List[str]) -> List[Optional[float]]:
        """Scores memes using the method specified in the config, handling uniqueness."""
        if not memes:
            return []

        unique_memes = list(dict.fromkeys(memes)) # Maintain order while getting uniques
        unique_scores = {}

        if self.fitness_model_huggingpath:
            if self.fitness_model:
                logger.debug(f"Scoring {len(unique_memes)} unique memes using FitnessModel.")
                scores_list = self.fitness_model.score(unique_memes)
                unique_scores = {meme: score for meme, score in zip(unique_memes, scores_list)}
            else:
                logger.error("FitnessModel ('custom') selected but not available. Returning None scores.")
                unique_scores = {meme: None for meme in unique_memes}
        else:
            if self.llm_service:
                 logger.debug(f"Scoring {len(unique_memes)} unique memes using LLM.")
                 scores_list = self.llm_service.score(unique_memes, temperature=self.llm_config['temperature_score'])
                 unique_scores = {meme: score for meme, score in zip(unique_memes, scores_list)}
            else:
                logger.error("LLMService ('llm') selected but not available. Returning None scores.")
                unique_scores = {meme: None for meme in unique_memes}

        # --- START: Apply word count penalty ---
        if self.avg_initial_word_count is not None:
            logger.debug(f"Applying word count penalty relative to avg: {self.avg_initial_word_count:.2f}")
            penalized_scores = {}
            for meme, raw_score in unique_scores.items():
                # Only apply penalty to valid, non-NaN scores
                if raw_score is not None and not np.isnan(raw_score) and isinstance(meme, str) and meme:
                    try:
                        word_count = len(meme.split())
                        diff = abs(word_count - self.avg_initial_word_count)
                        # Penalty function
                        penalty_factor = np.exp(-0.001 * (diff**2))
                        penalized_score = raw_score * penalty_factor
                        # Optional: Log detailed penalty effect for debugging
                        # logger.debug(f"Meme '{meme[:20]}...' WC:{word_count}, Diff:{diff:.1f}, PenaltyFactor:{penalty_factor:.3f}, RawScore:{raw_score:.3f}, PenalizedScore:{penalized_score:.3f}")
                        penalized_scores[meme] = penalized_score
                    except Exception as e:
                         logger.warning(f"Error applying penalty to meme '{meme[:50]}...': {e}. Using raw score.", exc_info=False)
                         penalized_scores[meme] = raw_score # Fallback to raw score on error
                else:
                    # Keep None, NaN, or invalid meme scores as they are
                    penalized_scores[meme] = raw_score
            unique_scores = penalized_scores # Update the dictionary with penalized scores
        else:
             logger.debug("Skipping word count penalty as average initial word count is not available.")
        # --- END: Apply word count penalty ---

        # Map scores back to the original list
        final_scores = [unique_scores.get(meme) for meme in memes]
        return final_scores
    
    def initialize_scores(self):
        """Scores the initial memes present in the graph if they haven't been scored, using the configured fitness model."""
        all_nodes_data = self.graph_manager.get_all_nodes_data()
        memes_to_score: Set[str] = set()
        nodes_needing_score: Dict[str, List[Any]] = {} # meme -> list of node_ids

        for node_id, data in all_nodes_data.items():
             # Check if score is None or NaN
            if data.current_meme_score is None or np.isnan(data.current_meme_score):
                memes_to_score.add(data.current_meme)
                if data.current_meme not in nodes_needing_score:
                    nodes_needing_score[data.current_meme] = []
                nodes_needing_score[data.current_meme].append(node_id)

        if not memes_to_score:
            logger.info("All initial memes already have valid scores.")
            return

        logger.info(f"Found {len(memes_to_score)} unique initial memes needing scores.")
        meme_list = list(memes_to_score)
        scores = self._score_memes(meme_list)

        score_map = {meme: score for meme, score in zip(meme_list, scores)}

        for meme, score in score_map.items():
            if score is not None:
                if meme in nodes_needing_score:
                    for node_id in nodes_needing_score[meme]:
                        self.graph_manager.update_node_score(node_id, score, history_index=0)
                        node_data = self.graph_manager.get_node_data(node_id)
                        if node_data:
                             node_data.current_meme_score = score
            else:
                logger.warning(f"Failed to get score for initial meme: '{meme[:50]}...'")

        logger.info("Initial scoring complete.")


    def _propagate_memes(self, generation: int):
        """Nodes attempt to spread their memes to neighbors based on edge weights."""
        logger.debug(f"Generation {generation}: Propagating memes...")
        propagated_count = 0
        for node_id in self.graph_manager.get_all_node_ids():
            node_data = self.graph_manager.get_node_data(node_id)
            if not node_data: continue

            current_meme = node_data.current_meme
            neighbors = self.graph_manager.get_neighbors(node_id)

            for neighbor_id in neighbors:
                weight = self.graph_manager.get_edge_weight(node_id, neighbor_id)
                if random.random() < weight:
                    self.graph_manager.add_received_meme(node_id, neighbor_id, current_meme, weight)
                    propagated_count += 1

        logger.debug(f"Generation {generation}: {propagated_count} meme propagations occurred.")

    
    def _process_received_memes(self, generation: int):
        """Nodes process received memes: score, compare, merge/mutate, update."""
        logger.debug(f"Generation {generation}: Processing received memes...")
        nodes_data = self.graph_manager.get_all_nodes_data()
        threshold = self.config['threshold']
        mutation_candidates: List[str] = []
        merge_candidates: List[Tuple[str, str]] = []
        nodes_to_mutate: List[Any] = []
        nodes_to_merge: List[Any] = []

        all_memes_in_play: Set[str] = set()
        meme_score_cache: Dict[str, Optional[float]] = {}

        # Collect memes and populate initial cache from existing node scores
        for node_id, data in nodes_data.items():
            all_memes_in_play.add(data.current_meme)
            if data.current_meme_score is not None and not np.isnan(data.current_meme_score):
                meme_score_cache[data.current_meme] = data.current_meme_score
                # Ensure history score alignment
                if len(data.history_scores) == len(data.history):
                    if data.history_scores[-1] is None or np.isnan(data.history_scores[-1]):
                         self.graph_manager.update_node_score(node_id, data.current_meme_score, history_index=-1)
                elif len(data.history_scores) < len(data.history):
                     padding = [np.nan] * (len(data.history) - 1 - len(data.history_scores)) # Use NaN for padding
                     data.history_scores.extend(padding)
                     data.history_scores.append(data.current_meme_score)

            for _, meme, _ in data.received_memes:
                all_memes_in_play.add(meme)

        # Score any memes not already in the cache
        memes_needing_score = list(all_memes_in_play - set(meme_score_cache.keys()))
        if memes_needing_score:
            logger.info(f"Generation {generation}: Scoring {len(memes_needing_score)} unique memes.")
            new_scores = self._score_memes(memes_needing_score)
            for meme, score in zip(memes_needing_score, new_scores):
                if score is None or np.isnan(score):
                     logger.warning(f"Generation {generation}: Failed to get valid score for meme '{meme[:50]}...'. It cannot be adopted.")
                     meme_score_cache[meme] = None
                else:
                    meme_score_cache[meme] = score
            logger.debug(f"Score cache updated. Size: {len(meme_score_cache)}")
        else:
             logger.debug(f"Generation {generation}: No new memes required scoring.")

        # Decide actions for each node based on cached scores
        updates_pending: Dict[Any, Dict[str, Any]] = {} # node_id -> {action: 'keep'|'mutate'|'merge', meme1: str, meme2: Optional[str]}

        for node_id, data in nodes_data.items():
            if not data.received_memes:
                continue

            current_meme = data.current_meme
            current_meme_score = meme_score_cache.get(current_meme)

            if current_meme_score is None:
                 logger.warning(f"Node {node_id}: Current meme '{current_meme[:50]}...' has no valid score in cache. Keeping current meme.")
                 continue

            best_sender_id: Optional[int] = None
            best_received_meme: Optional[str] = None
            best_received_score: float = -1.0
            for sender_node_id, meme, _ in data.received_memes:
                score = meme_score_cache.get(meme)
                if score is not None and score > best_received_score:
                    best_sender_id = sender_node_id
                    best_received_meme = meme
                    best_received_score = score

            if best_received_meme is None:
                 continue
            
            # Record the successful propagation event
            self.graph_manager.record_propagation(generation, best_sender_id, node_id, best_received_meme)

            # Use a small epsilon for division to prevent errors with zero scores
            score_ratio = best_received_score / (current_meme_score + 1e-9)

            action_type = "keep"
            if 1 - threshold <= score_ratio <= 1 + threshold:
                action_type = "merge"
                merge_candidates.append((current_meme, best_received_meme))
                nodes_to_merge.append(node_id)
            elif score_ratio > 1 + threshold:
                action_type = "mutate"
                mutation_candidates.append(best_received_meme) # Mutate the better meme
                nodes_to_mutate.append(node_id)

            if action_type != "keep":
                updates_pending[node_id] = {
                    "action": action_type,
                    "current_meme": current_meme,
                    "received_meme": best_received_meme
                 }

        # Perform LLM mutations and merges
        mutated_memes_map: Dict[str, str] = {}
        merged_memes_map: Dict[Tuple[str, str], str] = {}

        if mutation_candidates:
            unique_mutation_candidates = list(dict.fromkeys(mutation_candidates))
            logger.info(f"Generation {generation}: Mutating {len(unique_mutation_candidates)} unique memes using LLM.")
            mutated_results = self.llm_service.mutate(unique_mutation_candidates, temperature=self.llm_config['temperature_mutate'])
            mutated_memes_map = {orig: mutated for orig, mutated in zip(unique_mutation_candidates, mutated_results)}

        if merge_candidates:
             unique_merge_candidates = list(dict.fromkeys(merge_candidates))
             logger.info(f"Generation {generation}: Merging {len(unique_merge_candidates)} unique pairs using LLM.")
             merged_results = self.llm_service.merge(
                 [p[0] for p in unique_merge_candidates],
                 [p[1] for p in unique_merge_candidates],
                 temperature=self.llm_config['temperature_merge']
             )
             merged_memes_map = {pair: merged for pair, merged in zip(unique_merge_candidates, merged_results)}

        # Score the actually generated new memes
        all_generated_memes = list(mutated_memes_map.values()) + list(merged_memes_map.values())
        unique_generated_memes = list(dict.fromkeys(filter(None, all_generated_memes)))

        new_meme_score_map: Dict[str, Optional[float]] = {}
        if unique_generated_memes:
            logger.info(f"Generation {generation}: Scoring {len(unique_generated_memes)} newly generated unique memes.")
            new_scores = self._score_memes(unique_generated_memes)
            new_meme_score_map = {meme: score for meme, score in zip(unique_generated_memes, new_scores)}
        else:
             logger.debug(f"Generation {generation}: No valid new memes generated via mutate/merge.")

        # Apply updates to nodes
        for node_id in self.graph_manager.get_all_node_ids():
            original_node_data = nodes_data[node_id]
            current_meme = original_node_data.current_meme
            current_score = meme_score_cache.get(current_meme)

            final_meme = current_meme
            final_score = current_score
            action_taken = "keep"

            if node_id in updates_pending:
                pending_action = updates_pending[node_id]["action"]
                received_meme = updates_pending[node_id]["received_meme"]

                new_meme_text: Optional[str] = None
                if pending_action == "mutate":
                    new_meme_text = mutated_memes_map.get(received_meme)
                elif pending_action == "merge":
                    merge_key = (current_meme, received_meme)
                    new_meme_text = merged_memes_map.get(merge_key)

                if new_meme_text and new_meme_text != current_meme:
                    new_score = new_meme_score_map.get(new_meme_text)

                    if new_score is not None:
                        # Sanity check
                        if current_score is None or new_score >= current_score * 0.80:
                            final_meme = new_meme_text
                            final_score = new_score
                            action_taken = pending_action
                            logger.debug(f"Node {node_id}: Applying {action_taken} result. New score {final_score:.3f}")
                        else:
                             logger.debug(f"Node {node_id}: New meme score ({new_score:.3f}) significantly lower than original ({current_score:.3f}). Reverting to keep.")
                             action_taken = "keep (low score)"
                    else:
                        logger.warning(f"Node {node_id}: Generated meme '{new_meme_text[:50]}...' failed scoring. Reverting to keep.")
                        action_taken = "keep (score failed)"
                else:
                    log_msg = "failed" if not new_meme_text else "resulted in same meme"
                    logger.debug(f"Node {node_id}: Action '{pending_action}' {log_msg}. Reverting to keep.")
                    action_taken = f"keep ({pending_action} {log_msg})"

            # Always update node history
            self.graph_manager.update_node_meme(node_id, final_meme, final_score, generation)
            if action_taken != "keep":
                 logger.debug(f"Node {node_id}: Final action -> {action_taken}. Meme: '{final_meme[:30]}...' Score: {final_score}")


        # Clear received memes for the next generation
        self.graph_manager.clear_received_memes()
        logger.debug(f"Generation {generation}: Finished processing memes.")


    def mutate_initial_if_all_same(self):
        """
        Checks if all memes are identical at the start. If so, mutates all using LLM
        and scores using the configured fitness model.
        Updates the initial state (history[0], history_scores[0], current).
        """
        nodes_data = self.graph_manager.get_all_nodes_data()
        if not nodes_data:
             logger.info("Graph is empty, skipping initial mutation check.")
             return False

        try:
            first_node_data = next(iter(nodes_data.values()))
            first_meme = first_node_data.current_meme
        except StopIteration:
             logger.info("Graph has no nodes with data, skipping initial mutation check.")
             return False

        all_same = all(data.current_meme == first_meme for data in nodes_data.values())

        if all_same:
            logger.info(f"All initial nodes have the same meme ('{first_meme[:50]}...'). Mutating and scoring.")
            all_node_ids = list(nodes_data.keys())
            current_memes = [first_meme] * len(all_node_ids)

            mutated_memes = self.llm_service.mutate(current_memes, temperature=self.llm_config['temperature_mutate'])
            new_scores = self._score_memes(mutated_memes)

            mutation_applied_count = 0
            for i, node_id in enumerate(all_node_ids):
                 if i >= len(mutated_memes) or i >= len(new_scores):
                     logger.error(f"Index out of bounds ({i}) during initial mutation for node {node_id}. Max mutated: {len(mutated_memes)}, Max scored: {len(new_scores)}. Skipping.")
                     continue

                 new_meme = mutated_memes[i]
                 new_score = new_scores[i]

                 node_data = self.graph_manager.get_node_data(node_id)
                 if not node_data:
                      logger.warning(f"Node {node_id}: Could not retrieve data during initial mutation. Skipping update.")
                      continue

                 if new_meme != first_meme and new_meme and new_score is not None and not np.isnan(new_score):
                      node_data.current_meme = new_meme
                      node_data.current_meme_score = new_score
                      if node_data.history: node_data.history[0] = new_meme
                      else: node_data.history = [new_meme]
                      if node_data.history_scores: node_data.history_scores[0] = new_score
                      else: node_data.history_scores = [new_score]
                      mutation_applied_count +=1
                      logger.debug(f"Node {node_id}: Initial meme mutated to '{new_meme[:30]}...' (Score: {new_score:.3f})")
                 else:
                      log_reason = "meme unchanged/empty" if (new_meme == first_meme or not new_meme) else "scoring failed"
                      logger.warning(f"Node {node_id}: Initial mutation failed: {log_reason}. Keeping original.")
                      # Try to ensure the original has a score
                      if node_data.history_scores and node_data.history_scores[0] is not None and not np.isnan(node_data.history_scores[0]):
                           node_data.current_meme_score = node_data.history_scores[0]
                      else:
                           logger.warning(f"Node {node_id}: Original initial meme '{first_meme[:30]}...' lacks valid score. Re-scoring.")
                           original_score_list = self._score_memes([first_meme])
                           original_score = original_score_list[0] if original_score_list else None
                           if original_score is not None and not np.isnan(original_score):
                                node_data.current_meme_score = original_score
                                node_data.history_scores = [original_score] # Reset history score
                           else:
                                logger.error(f"Node {node_id}: Failed to score original initial meme. Score remains None.")
                                node_data.current_meme_score = None
                                node_data.history_scores = [None]

            logger.info(f"Applied initial mutation to {mutation_applied_count}/{len(all_node_ids)} nodes.")
            return True
        else:
            logger.info("Initial memes are diverse. No initial mutation applied.")
            return False
        

    def step(self, generation_index: int) -> None:
        """Performs one full generation step: propagate, process."""
        logger.info(f"--- Starting Generation {generation_index} ---")

        self._propagate_memes(generation_index)
        self._process_received_memes(generation_index)

        logger.info(f"--- Finished Generation {generation_index} ---")


    def run_simulation(self, start_generation_index: int = 0):
        """Runs the simulation loop for a configured number of generations,
        starting from start_generation_index."""
        num_generations_to_run = self.config['generations']
        end_generation_index = start_generation_index + num_generations_to_run
        logger.info(f"Starting simulation run from generation {start_generation_index} up to {end_generation_index-1}.")

        # --- Main Evolution Loop ---
        last_completed_generation_index = start_generation_index - 1
        logger.info("Starting main evolution loop...")
        for current_gen_index in range(start_generation_index, end_generation_index):
            try:
                self.step(current_gen_index)
                last_completed_generation_index = current_gen_index

                # Yield the index of the completed generation for external processing (like visualization)
                yield last_completed_generation_index

            except Exception as e:
                logger.error(f"Error during generation {current_gen_index + 1}: {e}", exc_info=True)
                logger.warning("Simulation stopped due to error.")
                break # Stop simulation on error

        logger.info(f"Simulation loop finished normally after generation {last_completed_generation_index + 1}.")
        return last_completed_generation_index + 1  # Return number of completed generations