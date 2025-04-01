import random
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np # For checking NaN

from graph_manager import GraphManager
from llm_service import LLMServiceInterface

logger = logging.getLogger(__name__)

class EvolutionEngine:
    """Orchestrates the meme evolution process over generations."""

    def __init__(self, graph_manager: GraphManager, llm_service: LLMServiceInterface, config: Dict):
        self.graph_manager = graph_manager
        self.llm_service = llm_service
        self.config = config['simulation']
        self.llm_config = config['llm'] # For temperatures etc.
        random.seed(config['seed'])

    def initialize_scores(self):
        """Scores the initial memes present in the graph if they haven't been scored."""
        all_nodes_data = self.graph_manager.get_all_nodes_data()
        memes_to_score: Set[str] = set()
        nodes_needing_score: Dict[str, List[Any]] = {} # meme -> list of node_ids

        for node_id, data in all_nodes_data.items():
            if data.current_meme_score is None:
                memes_to_score.add(data.current_meme)
                if data.current_meme not in nodes_needing_score:
                    nodes_needing_score[data.current_meme] = []
                nodes_needing_score[data.current_meme].append(node_id)

        if not memes_to_score:
            logger.info("All initial memes already have scores.")
            return

        logger.info(f"Found {len(memes_to_score)} unique initial memes needing scores.")
        meme_list = list(memes_to_score)
        scores = self.llm_service.score(meme_list, temperature=self.llm_config['temperature_score'])

        # Apply scores back to nodes
        for meme, score in zip(meme_list, scores):
            if score is not None:
                for node_id in nodes_needing_score[meme]:
                    # Update current score and the initial entry in history_scores
                    self.graph_manager.update_node_score(node_id, score, history_index=0)
                    # Also set the current_meme_score directly for consistency
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
                    self.graph_manager.add_received_meme(neighbor_id, current_meme, weight)
                    # Record the propagation event
                    self.graph_manager.record_propagation(generation, node_id, neighbor_id, current_meme, weight)
                    propagated_count += 1

        logger.debug(f"Generation {generation}: {propagated_count} meme propagations occurred.")


    def _process_received_memes(self, generation: int):
        """Nodes process received memes: score, compare, merge/mutate, update."""
        logger.debug(f"Generation {generation}: Processing received memes...")
        nodes_data = self.graph_manager.get_all_nodes_data()
        threshold = self.config['threshold']
        mutation_candidates: List[str] = [] # Memes to potentially mutate (best received > current)
        merge_candidates: List[Tuple[str, str]] = [] # Pairs to merge (current, best_received)
        nodes_to_mutate: List[Any] = [] # node_ids corresponding to mutation_candidates
        nodes_to_merge: List[Any] = [] # node_ids corresponding to merge_candidates
        nodes_to_keep: List[Any] = [] # node_ids that decided to keep their current meme
        all_memes_in_play: Set[str] = set() # Track all memes (current, received) for scoring
        meme_score_cache: Dict[str, Optional[float]] = {} # Cache scores for this generation

        # Collect all memes and check existing scores
        for node_id, data in nodes_data.items():
            all_memes_in_play.add(data.current_meme)
            if data.current_meme_score is not None:
                 meme_score_cache[data.current_meme] = data.current_meme_score
                 # Ensure history score is aligned if possible
                 if len(data.history_scores) == len(data.history):
                     if data.history_scores[-1] is None or np.isnan(data.history_scores[-1]):
                          self.graph_manager.update_node_score(node_id, data.current_meme_score, history_index=-1)
                 elif len(data.history_scores) < len(data.history):
                     # Pad and update if history grew but score wasn't recorded
                     padding = [None] * (len(data.history) - 1 - len(data.history_scores))
                     data.history_scores.extend(padding)
                     data.history_scores.append(data.current_meme_score)


            for meme, weight in data.received_memes:
                all_memes_in_play.add(meme)
                # We'll score all unscored memes in batch later

        # Batch score any memes lacking a score in the cache
        memes_needing_score = [meme for meme in all_memes_in_play if meme not in meme_score_cache]
        if memes_needing_score:
            logger.info(f"Generation {generation}: Scoring {len(memes_needing_score)} new or unscored memes.")
            new_scores = self.llm_service.score(memes_needing_score, temperature=self.llm_config['temperature_score'])
            for meme, score in zip(memes_needing_score, new_scores):
                if score is None:
                     logger.warning(f"Generation {generation}: Failed to score meme '{meme[:50]}...'. It cannot be adopted.")
                meme_score_cache[meme] = score # Store None if scoring failed

        # Decide actions for each node
        updates_pending: Dict[Any, Tuple[str, Optional[float], str]] = {} # node_id -> (new_meme, new_score, action_type)

        for node_id, data in nodes_data.items():
            if not data.received_memes:
                # No change, just ensure history/score is consistent for next round
                current_score = meme_score_cache.get(data.current_meme)
                self.graph_manager.update_node_meme(node_id, data.current_meme, current_score, generation) # Effectively appends same meme/score
                continue

            current_meme = data.current_meme
            current_meme_score = meme_score_cache.get(current_meme)

            if current_meme_score is None:
                 logger.warning(f"Node {node_id}: Current meme '{current_meme[:50]}...' has no score. Cannot compare. Keeping current meme.")
                 self.graph_manager.update_node_meme(node_id, current_meme, None, generation)
                 continue # Cannot proceed without current score

            # Find the best received meme (highest score, ignore weighting for now, or use weighted score?)
            # Let's use highest score among received, ignoring None scores
            best_received_meme: Optional[str] = None
            best_received_score: float = -1.0

            for meme, weight in data.received_memes:
                score = meme_score_cache.get(meme)
                if score is not None and score > best_received_score:
                    best_received_meme = meme
                    best_received_score = score

            if best_received_meme is None:
                 # No valid scored memes received
                 self.graph_manager.update_node_meme(node_id, current_meme, current_meme_score, generation)
                 continue

            # Compare scores and decide action
            score_ratio = best_received_score / current_meme_score if current_meme_score > 0 else float('inf')
                
            if 1 - threshold <= score_ratio <= 1 + threshold:
                action_type = "merge"
                merge_candidates.append((current_meme, best_received_meme))
                nodes_to_merge.append(node_id)
            elif score_ratio > 1 + threshold:
                action_type = "mutate"
                mutation_candidates.append(best_received_meme) # Mutate the *better* meme
                nodes_to_mutate.append(node_id)
            elif score_ratio < 1 - threshold:
                action_type = "keep"

            # Store preliminary decision - actual meme/score depends on LLM results
            if action_type != "keep":
                 updates_pending[node_id] = (None, None, action_type) # Placeholder for meme/score
            else:
                 # If keeping, lock it in now
                 self.graph_manager.update_node_meme(node_id, current_meme, current_meme_score, generation)


        # Batch process LLM operations
        mutated_memes: List[str] = []
        merged_memes: List[str] = []

        if mutation_candidates:
            logger.info(f"Generation {generation}: Mutating {len(mutation_candidates)} memes.")
            mutated_memes = self.llm_service.mutate(mutation_candidates, temperature=self.llm_config['temperature_mutate'])

        if merge_candidates:
             logger.info(f"Generation {generation}: Merging {len(merge_candidates)} pairs.")
             merged_memes = self.llm_service.merge(
                 [p[0] for p in merge_candidates],
                 [p[1] for p in merge_candidates],
                 temperature=self.llm_config['temperature_merge']
             )

        # Score the newly generated memes
        all_new_memes = mutated_memes + merged_memes
        new_meme_scores: List[Optional[float]] = []
        if all_new_memes:
            logger.info(f"Generation {generation}: Scoring {len(all_new_memes)} newly generated memes.")
            new_meme_scores = self.llm_service.score(all_new_memes, temperature=self.llm_config['temperature_score'])
        else:
             logger.debug(f"Generation {generation}: No new memes generated via mutate/merge.")


        # Apply updates back to the graph
        mutate_idx, merge_idx, score_idx = 0, 0, 0
        for node_id, (pending_meme, pending_score, action_type) in updates_pending.items():
            final_meme: Optional[str] = None
            final_score: Optional[float] = None
            original_node_data = nodes_data[node_id] # Get original data for comparison

            if action_type == "mutate":
                if mutate_idx < len(mutated_memes) and score_idx < len(new_meme_scores):
                    final_meme = mutated_memes[mutate_idx]
                    final_score = new_meme_scores[score_idx]
                    mutate_idx += 1
                    score_idx += 1
                else:
                     logger.error(f"Index mismatch during mutation update for node {node_id}.")
                     action_type = "keep" # Fallback to keep if something went wrong

            elif action_type == "merge":
                if merge_idx < len(merged_memes) and score_idx < len(new_meme_scores):
                    final_meme = merged_memes[merge_idx]
                    final_score = new_meme_scores[score_idx]
                    merge_idx += 1
                    score_idx += 1
                else:
                     logger.error(f"Index mismatch during merge update for node {node_id}.")
                     action_type = "keep" # Fallback to keep

            # Sanity check: Don't adopt a much worse meme than original
            if action_type != "keep" and final_score is not None and original_node_data.current_meme_score is not None:
                 # Allow some drop, but not drastic (e.g., less than 80% of original score)
                 if final_score < original_node_data.current_meme_score * 0.80:
                      logger.debug(f"Node {node_id}: New meme score ({final_score:.3f}) significantly lower than original ({original_node_data.current_meme_score:.3f}). Reverting to keep.")
                      action_type = "keep"

            # Final update application
            if action_type == "keep" or final_meme is None or final_score is None:
                 # Ensure keep action is recorded if it wasn't already
                 if node_id not in self.graph_manager.get_node_data(node_id).history: # Check if it was already processed as 'keep'
                      self.graph_manager.update_node_meme(node_id, original_node_data.current_meme, original_node_data.current_meme_score, generation)
            else:
                 # Apply the new meme and score
                 self.graph_manager.update_node_meme(node_id, final_meme, final_score, generation)
                 logger.debug(f"Node {node_id}: Updated via {action_type} to meme score {final_score:.3f}")


        # Clear received memes for the next generation
        self.graph_manager.clear_received_memes()
        logger.debug(f"Generation {generation}: Finished processing memes.")


    def _mutate_initial_if_all_same(self):
        """
        Checks if all memes are identical at the start. If so, mutates all.
        This should be called once before the main simulation loop begins.
        Updates the initial state (history[0], history_scores[0], current).
        """
        logger.info("Checking initial meme state for uniformity...")
        nodes_data = self.graph_manager.get_all_nodes_data()
        if not nodes_data:
             logger.info("Graph is empty, skipping initial mutation check.")
             return False # Nothing to do

        # Use an iterator to get the first item without loading all keys/values if not needed
        try:
            first_node_data = next(iter(nodes_data.values()))
            first_meme = first_node_data.current_meme
        except StopIteration:
             logger.info("Graph has no nodes with data, skipping initial mutation check.")
             return False # No data found

        all_same = all(data.current_meme == first_meme for data in nodes_data.values())

        if all_same:
            logger.info(f"All initial nodes have the same meme ('{first_meme[:50]}...'). Applying mutation to all.")
            all_node_ids = list(nodes_data.keys())
            current_memes = [first_meme] * len(all_node_ids) # List of the same initial meme

            # Mutate and score the initial meme variants
            mutated_memes = self.llm_service.mutate(current_memes, temperature=self.llm_config['temperature_mutate'])
            new_scores = self.llm_service.score(mutated_memes, temperature=self.llm_config['temperature_score'])

            mutation_applied_count = 0
            for i, node_id in enumerate(all_node_ids):
                 new_meme = mutated_memes[i]
                 new_score = new_scores[i]

                 # Get the specific node data object to update directly
                 node_data = self.graph_manager.get_node_data(node_id)
                 if not node_data:
                      logger.warning(f"Node {node_id}: Could not retrieve data during initial mutation. Skipping update.")
                      continue

                 # Apply update only if mutation and scoring succeeded
                 if new_meme != first_meme and new_meme != "" and new_score is not None: # Check if mutation is valid and score exists
                      # --- Direct Update of Initial State ---
                      node_data.current_meme = new_meme
                      node_data.current_meme_score = new_score
                      # Replace the first entry in history/scores
                      if node_data.history:
                           node_data.history[0] = new_meme
                      else: # Should not happen if MemeNodeData initializes correctly
                           node_data.history = [new_meme]

                      if node_data.history_scores:
                           # Ensure history_scores is long enough (might happen if init score was off)
                           if len(node_data.history_scores) == 0:
                                node_data.history_scores = [new_score]
                           else:
                                node_data.history_scores[0] = new_score
                      else:
                           node_data.history_scores = [new_score]
                      # --- End Direct Update ---
                      mutation_applied_count +=1
                      logger.debug(f"Node {node_id}: Initial meme mutated to '{new_meme[:30]}...' (Score: {new_score:.3f})")

                 else: # Mutation/scoring failed or returned original/empty
                      logger.warning(f"Node {node_id}: Failed mutation/scoring during initial 'mutate all' or meme unchanged. Keeping original initial meme.")
                      # Ensure current score matches initial score history if it exists
                      if node_data.history_scores and node_data.history_scores[0] is not None:
                           node_data.current_meme_score = node_data.history_scores[0]
                      # No need to call update_node_meme, state is already correct

            logger.info(f"Applied initial mutation to {mutation_applied_count}/{len(all_node_ids)} nodes.")
            return True # Indicates mutation occurred
        else:
            logger.info("Initial memes are diverse. No initial mutation applied.")
            return False # Indicates no mutation occurred


    def step(self, generation: int) -> int:
        """Performs one full generation step: propagate, process."""
        logger.info(f"--- Starting Generation {generation} ---")
        
        self._propagate_memes(generation)
        self._process_received_memes(generation)

        logger.info(f"--- Finished Generation {generation} ---")


    def run_simulation(self) -> int:
        """Runs the full simulation loop."""
        num_generations = self.config['generations']
        logger.info(f"Starting simulation for {num_generations} generations.")

        # Check and potentially mutate initial uniform state
        self._mutate_initial_if_all_same()

        # --- Main Evolution Loop ---
        last_completed_generation = -1
        logger.info("Starting main evolution loop...")
        for gen in range(num_generations):
            try:
                self.step(gen)
                last_completed_generation = gen

                # Yield the index of the completed generation for external processing (like visualization)
                yield last_completed_generation

            except Exception as e:
                logger.error(f"Error during generation {gen + 1}: {e}", exc_info=True)
                logger.warning("Simulation stopped due to error.")
                break # Stop simulation on error

        logger.info(f"Simulation finished after {last_completed_generation + 1} completed generations.")
        return last_completed_generation + 1  # Return number of completed generations