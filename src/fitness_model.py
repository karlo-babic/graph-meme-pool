import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, Pipeline
from typing import List, Optional, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("FitnessModel: Using CUDA device.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("FitnessModel: Using MPS device.")
else:
    device = torch.device("cpu")
    logger.info("FitnessModel: Using CPU device.")


class FitnessModel:
    """
    Calculates fitness scores for texts using a pre-trained sequence classification model.
    This version is specifically adapted for sentiment analysis models like
    'cardiffnlp/twitter-roberta-base-sentiment-latest', returning the positive sentiment score.
    """

    def __init__(self, model_huggingpath: str):
        """
        Initializes the FitnessModel.

        Args:
            model_huggingpath: Path or identifier for the Hugging Face sequence classification model.
                               Expected to be a sentiment model with labels (neg, neut, pos).
        """
        self.model_huggingpath = model_huggingpath
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.label_map: Optional[Dict[int, str]] = None # To store label mapping

    def load(self):
        """Loads the model and tokenizer."""
        if self.model:
            logger.info(f"FitnessModel '{self.model_huggingpath}' already loaded.")
            return
        try:
            logger.info(f"Loading FitnessModel: {self.model_huggingpath}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_huggingpath)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_huggingpath)
            self.model.to(device) # Move model to the appropriate device
            self.model.eval() # Set model to evaluation mode

            # Store label mapping for clarity and potential validation
            self.label_map = self.model.config.id2label
            logger.info(f"FitnessModel label mapping: {self.label_map}")
            # Basic check for expected labels
            if 2 not in self.label_map or not ('pos' in self.label_map[2].lower()):
                 logger.warning(f"Model '{self.model_huggingpath}' might not have 'positive' label at index 2. Label map: {self.label_map}. Scoring assumes index 2 is positive.")


            logger.info(f"FitnessModel '{self.model_huggingpath}' loaded successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to load FitnessModel '{self.model_huggingpath}': {e}", exc_info=True)
            raise RuntimeError(f"FitnessModel loading failed: {e}") from e

    def score(self, texts: List[str]) -> List[Optional[float]]:
        """
        Scores a batch of texts using the loaded sentiment classification model.

        Args:
            texts: A list of strings (memes) to score.

        Returns:
            A list of 'Sentiment Balance Scores' (float between 0 and 1).
            - 1 indicates strongly positive relative to negative.
            - 0 indicates strongly negative relative to positive.
            - 0.5 indicates equal positive/negative strength or purely neutral.
            Returns None if scoring failed for a specific text.
        """
        if not self.model or not self.tokenizer:
            logger.error("FitnessModel not loaded. Cannot score texts.")
            return [None] * len(texts)
        if not texts:
            return []

        logger.debug(f"FitnessModel: Scoring {len(texts)} texts for sentiment balance...")
        results: List[Optional[float]] = [None] * len(texts)

        try:
            # Process in batches if needed, but for moderate numbers, one batch is fine
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            # Move inputs to the same device as the model
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)

                # --- Score Calculation: Sentiment Balance Score ---
                # Extract probabilities for positive (index 2) and negative (index 0)
                if probabilities.shape[1] >= 3: # Ensure there are at least 3 classes
                    positive_scores_tensor = probabilities[:, 2] # Index 2 for positive
                    negative_scores_tensor = probabilities[:, 0] # Index 0 for negative
                else:
                    logger.error(f"FitnessModel '{self.model_huggingpath}' output has fewer than 3 classes ({probabilities.shape[1]}). Cannot reliably calculate sentiment balance.")
                    return [None] * len(texts)

                # Calculate the sum of positive and negative probabilities
                sum_pn = positive_scores_tensor + negative_scores_tensor

                # Define a small threshold for numerical stability (avoid division by zero)
                threshold = 1e-9

                # Calculate score: P_pos / (P_pos + P_neg) where sum is non-zero, else 0.5
                # Uses torch.where for element-wise conditional logic
                sentiment_balance_score_tensor = torch.where(
                    sum_pn > threshold,
                    positive_scores_tensor / sum_pn,
                    # If sum is zero (i.e., both P_pos and P_neg are zero), output 0.5 (balanced)
                    torch.tensor(0.5, device=device, dtype=positive_scores_tensor.dtype) # Ensure 0.5 is on correct device and dtype
                )

                scores_np = sentiment_balance_score_tensor.cpu().numpy()

            # Ensure scores are floats and handle potential NaN/Inf
            for i, score in enumerate(scores_np):
                 if isinstance(score, (float, np.floating)) and np.isfinite(score):
                      # Clamp score to [0, 1] just in case of floating point inaccuracies
                      results[i] = float(np.clip(score, 0.0, 1.0))
                 else:
                      logger.warning(f"FitnessModel produced invalid score ({score}) for text index {i}. Assigning None.")
                      results[i] = None

            logger.debug(f"FitnessModel: Scoring complete.")

        except Exception as e:
            logger.error(f"Error during FitnessModel scoring: {e}", exc_info=True)
            # Return None for all if a batch-level error occurs
            return [None] * len(texts)

        return results