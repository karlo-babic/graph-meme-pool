from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional

@dataclass
class MemeNodeData:
    """Holds the data associated with a node in the Meme Graph."""
    node_id: Any  # Keep track of the node ID it belongs to
    current_meme: str
    history: List[str] = field(default_factory=list)
    history_scores: List[Optional[float]] = field(default_factory=list) # Scores corresponding to history
    current_meme_score: Optional[float] = None
    received_memes: List[Tuple[str, float]] = field(default_factory=list) # List of (meme_text, influence_weight)
    group: Optional[int] = None # Group identifier, if applicable

    def __post_init__(self):
        # Ensure initial meme is in history if history is empty
        if not self.history:
            self.history.append(self.current_meme)
            # If score exists, history_scores should align, otherwise pad
            if len(self.history_scores) < len(self.history):
                 self.history_scores.extend([None] * (len(self.history) - len(self.history_scores)))
            if self.current_meme_score is not None and self.history_scores[-1] is None:
                 self.history_scores[-1] = self.current_meme_score


# Structure for storing propagation events
@dataclass
class PropagationEvent:
    generation: int
    source_node: Any
    target_node: Any
    meme: str
    weight: float # Influence weight used for this propagation