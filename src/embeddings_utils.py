import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer, models
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
# Import BertTokenizer, BertModel only if needed for alternative embedding method
# from transformers import BertTokenizer, BertModel
# import torch

logger = logging.getLogger(__name__)

# --- Sentence Transformer based ---

_sentence_model_cache: Dict[str, SentenceTransformer] = {}

def get_sentence_transformer_model(model_name: str) -> SentenceTransformer:
    """Loads or retrieves a cached SentenceTransformer model."""
    if model_name not in _sentence_model_cache:
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        try:
            # Handling potential need for explicit pooling layer if using base transformers
            if "bert-" in model_name or "roberta-" in model_name: # Add other base models if needed
                 logger.info(f"Using base transformer {model_name}. Adding Pooling layer.")
                 word_embedding_model = models.Transformer(model_name)
                 # CLS pooling is often good for sentence embeddings with BERT
                 pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
                 model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            else:
                 # Assume it's a standard SentenceTransformer model
                 model = SentenceTransformer(model_name)
            _sentence_model_cache[model_name] = model
            logger.info(f"Model {model_name} loaded.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
            raise
    return _sentence_model_cache[model_name]

def calculate_sentence_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Calculates embeddings for a list of texts using a SentenceTransformer model."""
    logger.debug(f"Calculating sentence embeddings for {len(texts)} texts.")
    try:
        embeddings = model.encode(texts, show_progress_bar=False) # Disable progress bar for cleaner logs
        return embeddings
    except Exception as e:
        logger.error(f"Error calculating sentence embeddings: {e}")
        # Return empty array or raise? Returning empty for now.
        return np.array([])

def calculate_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
     """Calculates cosine similarity between two embeddings."""
     # Ensure embeddings are 2D arrays for cosine_similarity
     if emb1.ndim == 1: emb1 = emb1.reshape(1, -1)
     if emb2.ndim == 1: emb2 = emb2.reshape(1, -1)
     return cosine_similarity(emb1, emb2)[0][0]


# --- Dimension Reduction ---

def reduce_dimensions_tsne(embeddings_dict: Dict[Any, np.ndarray], n_components: int = 2, perplexity: int = 30, random_state: Optional[int] = 42) -> Dict[Any, np.ndarray]:
    """Reduces embeddings dimensionality using t-SNE."""
    if not embeddings_dict:
        logger.warning("Cannot reduce dimensions: Embeddings dictionary is empty.")
        return {}

    node_ids = list(embeddings_dict.keys())
    try:
        all_embeddings = np.array(list(embeddings_dict.values()))

        # Check if enough samples for perplexity
        n_samples = all_embeddings.shape[0]
        effective_perplexity = min(perplexity, n_samples - 1)
        if effective_perplexity < 5: # t-SNE might perform poorly with very low perplexity
            logger.warning(f"Low number of samples ({n_samples}) for t-SNE. Perplexity adjusted to {effective_perplexity}. Results might be suboptimal.")
        if n_samples <= n_components:
             logger.warning(f"Number of samples ({n_samples}) is less than or equal to target dimensions ({n_components}). Skipping t-SNE.")
             # Return original embeddings sliced or padded? Or just indicate failure?
             # For now, returning empty dict to signal issue.
             return {}


        tsne = TSNE(n_components=n_components, perplexity=effective_perplexity, random_state=random_state, init='pca', learning_rate='auto') # Added common params
        logger.debug(f"Running t-SNE (perplexity={effective_perplexity})...")
        reduced_embeddings = tsne.fit_transform(all_embeddings)
        logger.debug("t-SNE finished.")

        return {node_id: reduced_embeddings[i] for i, node_id in enumerate(node_ids)}

    except ValueError as ve:
         # Catch specific t-SNE errors like perplexity > n_samples - 1
         logger.error(f"t-SNE failed: {ve}. Input shape: {all_embeddings.shape}")
         return {} # Return empty on failure
    except Exception as e:
        logger.error(f"Error during t-SNE dimension reduction: {e}")
        return {} # Return empty on failure


# --- (Optional) Alternative BERT embedding using raw transformers ---
# Keep this commented out unless specifically needed, SentenceTransformer is generally easier

# _bert_model_cache: Dict[str, Any] = {}
# _bert_tokenizer_cache: Dict[str, Any] = {}

# def get_bert_model_and_tokenizer(model_name: str):
#     """Loads or retrieves cached BERT model and tokenizer."""
#     if model_name not in _bert_model_cache:
#         logger.info(f"Loading BERT model/tokenizer: {model_name}")
#         try:
#             tokenizer = BertTokenizer.from_pretrained(model_name)
#             model = BertModel.from_pretrained(model_name)
#             model.eval() # Set to evaluation mode
#             _bert_tokenizer_cache[model_name] = tokenizer
#             _bert_model_cache[model_name] = model
#             logger.info(f"BERT model/tokenizer {model_name} loaded.")
#         except Exception as e:
#             logger.error(f"Failed to load BERT model/tokenizer {model_name}: {e}")
#             raise
#     return _bert_model_cache[model_name], _bert_tokenizer_cache[model_name]

# def calculate_bert_embeddings(texts: List[str], model, tokenizer) -> Dict[int, np.ndarray]:
#     """Calculates embeddings using raw BERT model (mean pooling)."""
#     embeddings = {}
#     logger.debug(f"Calculating BERT embeddings for {len(texts)} texts.")
#     try:
#         with torch.no_grad():
#             for i, text in enumerate(texts):
#                 inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
#                 outputs = model(**inputs)
#                 # Mean pooling of the last hidden state
#                 embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                 embeddings[i] = embedding
#         return embeddings
#     except Exception as e:
#         logger.error(f"Error calculating BERT embeddings: {e}")
#         return {}