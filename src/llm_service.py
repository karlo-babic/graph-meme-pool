import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
import re

logger = logging.getLogger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Helper function (consider moving to a utils module if it grows)
def remove_unfinished_sentence(text: str) -> str:
    """Removes potentially incomplete sentence fragments at the end."""
    text = text.strip()
    if not text:
        return ""
    # Basic check: if it doesn't end with standard punctuation, try to truncate
    if not re.search(r'[.!?]$', text):
        # Find the last standard sentence-ending punctuation
        last_punctuation_pos = -1
        for char in '.!?':
            pos = text.rfind(char)
            if pos > last_punctuation_pos:
                last_punctuation_pos = pos

        if last_punctuation_pos != -1:
            # Return text up to and including the last punctuation
            return text[:last_punctuation_pos + 1].strip()
        else:
            # No standard punctuation found, maybe return original or handle differently
            # For now, return original as it might be a title or single phrase
            return text
    return text


class LLMServiceInterface(ABC):
    """Interface for LLM operations required by the simulation."""

    @abstractmethod
    def load(self):
        """Load the LLM model and tokenizer."""
        pass

    @abstractmethod
    def mutate(self, texts: List[str], temperature: float) -> List[str]:
        """Mutates a batch of texts."""
        pass

    @abstractmethod
    def merge(self, texts1: List[str], texts2: List[str], temperature: float) -> List[str]:
        """Merges pairs of texts (adjusting text1 towards text2)."""
        pass

    @abstractmethod
    def score(self, texts: List[str], temperature: float) -> List[Optional[float]]:
        """Scores a batch of texts for virality (normalized 0-1)."""
        pass


class LLMService(LLMServiceInterface):
    """Concrete implementation using Hugging Face Transformers pipeline."""

    def __init__(self, config: Dict):
        self.config = config['llm']
        self.huggingpath = self.config['huggingpath']
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.text_generator: Optional[Pipeline] = None
        self.device = 0 if torch.cuda.is_available() else -1 # Use GPU if available

    def load(self):
        """Loads the model and tokenizer."""
        if self.model is not None:
            logger.info("Model already loaded.")
            return

        logger.info(f"Loading LLM model: {self.huggingpath}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.huggingpath, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.huggingpath, trust_remote_code=True)
            # Ensure pad_token_id is set if tokenizer doesn't have one (common with some models like Phi-3)
            if self.tokenizer.pad_token_id is None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token
                 self.model.config.pad_token_id = self.model.config.eos_token_id

            self.text_generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            logger.info(f"Model {self.huggingpath} loaded successfully on device {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load model {self.huggingpath}: {e}")
            raise RuntimeError(f"LLM loading failed: {e}") from e
            sys.exit(1)

    def _generate_with_retry(self,
                                prompts: List[str],
                                max_new_tokens: int,
                                temperature: float,
                                stop_sequence: str,
                                validation_fn,
                                original_texts: List[str]) -> List[str]: # Removed Optional and default=None
            """Handles text generation with retries for invalid outputs."""
            if not self.text_generator:
                logger.error("LLM model not loaded. Call load() first.")
                raise RuntimeError("LLM model not loaded.")
            if not prompts:
                return []

            results = [None] * len(prompts)
            pending_indices = list(range(len(prompts)))

            # Initial length check (optional but good sanity check)
            if len(original_texts) != len(prompts):
                logger.error(f"CRITICAL: Mismatch between prompts ({len(prompts)}) and original_texts ({len(original_texts)}) length in _generate_with_retry. Fallback WILL be incorrect.")
                # In this case, using "" might be safer than potentially wrong original text
                # Or raise an error earlier? Let's use "" for now and log the error.
                use_original_fallback = False
            else:
                use_original_fallback = True


            max_retries = 3
            retry_count = 0

            while pending_indices and retry_count < max_retries:
                if retry_count > 0:
                    logger.warning(f"Retrying generation for {len(pending_indices)} prompts (Attempt {retry_count + 1})")

                batch_prompts = [prompts[i] for i in pending_indices]
                try:
                    outputs = self.text_generator(
                        batch_prompts,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id, # Explicitly set for batching
                        return_full_text=False
                    )
                except Exception as e:
                    logger.error(f"Error during LLM generation: {e}")
                    retry_count += 1
                    continue

                new_pending_indices = []
                output_map = {original_index: output for original_index, output in zip(pending_indices, outputs)}

                for i in pending_indices:
                    output = output_map[i]
                    if not output or not output[0]['generated_text']:
                        logger.warning(f"Empty response received for prompt index {i}. Retrying.")
                        new_pending_indices.append(i)
                        continue

                    response_text = output[0]['generated_text']
                    response_text = response_text.split(stop_sequence, 1)[0].strip().split('\n', 1)[0].strip()

                    is_valid, processed_response = validation_fn(response_text)

                    if is_valid:
                        results[i] = processed_response
                    else:
                        # Log invalid response BEFORE adding to retry list
                        logger.warning(f"Invalid response for prompt index {i}: '{response_text[:100]}...'. Retrying.")
                        new_pending_indices.append(i)


                pending_indices = new_pending_indices
                if pending_indices:
                    retry_count += 1


            # Handle prompts that failed after all retries
            for i in pending_indices:
                logger.error(f"Failed to get valid response for prompt index {i} after {max_retries} retries. Using fallback.")
                # Add logging here
                logger.debug(f"Fallback check for index {i}: use_original_fallback={use_original_fallback}, i={i}, len(original_texts)={len(original_texts) if original_texts else 'N/A'}")

                # Use original text if lengths matched initially and index is valid
                if use_original_fallback and i < len(original_texts):
                    results[i] = original_texts[i]
                    logger.warning(f"Fallback for index {i}: Returning original text.")
                else:
                    results[i] = "" # Fallback to empty string if original text isn't usable (e.g., length mismatch)
                    logger.warning(f"Fallback for index {i}: Original text unavailable or index/length mismatch. Returning empty string.")


            return results


    def mutate(self, texts: List[str], temperature: Optional[float] = None) -> List[str]:
        if temperature is None:
            temperature = self.config['temperature_mutate']

        preprompt = '<|system|>' + self.config['prompt_mutate'] + '<|end|><|user|>'
        prompts = [preprompt + f"{text}<|assistant|>" for text in texts]
        stop_sequence = "<|end|>"

        def validation_fn(response):
            response = response.strip()
            # More robust check for tags or unwanted artifacts
            if not response or "<text>" in response or "<modified text>" in response or "##" in response or "**" in response or response[0] == '"':
                return False, response
            cleaned_response = remove_unfinished_sentence(response)
            return bool(cleaned_response), cleaned_response # Ensure not empty after cleaning

        logger.info(f"Mutating {len(texts)} texts...")
        results = self._generate_with_retry(
            prompts,
            max_new_tokens=self.config['max_new_tokens'],
            temperature=temperature,
            stop_sequence=stop_sequence,
            validation_fn=validation_fn,
            original_texts=texts
        )
        logger.info("Mutation finished.")
        return results


    def merge(self, texts1: List[str], texts2: List[str], temperature: Optional[float] = None) -> List[str]:
        if len(texts1) != len(texts2):
            raise ValueError("merge requires equal length lists for texts1 and texts2")
        if temperature is None:
            temperature = self.config['temperature_merge']

        preprompt = '<|system|>' + self.config['prompt_merge'] + '<|end|><|user|>'
        prompts = [preprompt + f"{t1}\n{t2}<|assistant|>" for t1, t2 in zip(texts1, texts2)]
        stop_sequence = "<|end|>"

        def validation_fn(response):
            response = response.strip()
            if not response or "<text" in response or "<new text>" in response or "##" in response or "**" in response or response[0] == '"':
                return False, response
            cleaned_response = remove_unfinished_sentence(response)
            return bool(cleaned_response), cleaned_response

        logger.info(f"Merging {len(texts1)} pairs of texts...")
        results = self._generate_with_retry(
            prompts,
            max_new_tokens=self.config['max_new_tokens'],
            temperature=temperature,
            stop_sequence=stop_sequence,
            validation_fn=validation_fn,
            original_texts=texts1
        )
        logger.info("Merging finished.")
        return results
    

    def score(self, texts: List[str], temperature: Optional[float] = None) -> List[Optional[float]]:
        if temperature is None:
            temperature = self.config['temperature_score']

        max_score = 9 # The scale used in the prompt
        preprompt = '<|system|>' + self.config['prompt_score'] + '<|end|><|user|>'
        prompts = [preprompt + f"{text}<|assistant|>Rating: " for text in texts]
        # No explicit stop sequence needed, relying on parsing the number
        stop_sequence = "<|end|>" # Or just rely on max_new_tokens

        raw_scores = [None] * len(texts) # Store raw text results
        pending_indices = list(range(len(texts)))

        max_retries = 3
        retry_count = 0

        logger.info(f"Scoring {len(texts)} texts for virality...")

        while pending_indices and retry_count < max_retries:
             if retry_count > 0:
                 logger.warning(f"Retrying scoring for {len(pending_indices)} prompts (Attempt {retry_count + 1})")

             batch_prompts = [prompts[i] for i in pending_indices]
             try:
                 outputs = self.text_generator(
                     batch_prompts,
                     max_new_tokens=1, # Score should be very short
                     do_sample=True,
                     temperature=temperature,
                     pad_token_id=self.tokenizer.eos_token_id,
                     return_full_text=False
                 )
             except Exception as e:
                 logger.error(f"Error during LLM scoring generation: {e}")
                 retry_count += 1
                 continue

             new_pending_indices = []
             output_map = {original_index: output for original_index, output in zip(pending_indices, outputs)}

             for i in pending_indices:
                 output = output_map[i]
                 if not output or not output[0]['generated_text']:
                     logger.warning(f"Empty score response received for prompt index {i}. Retrying.")
                     new_pending_indices.append(i)
                     continue

                 response = output[0]['generated_text'].strip().lower()
                 # Try to extract the score number robustly
                 match = re.search(r'^(\d)', response) # Look for a digit at the beginning
                 if match:
                     score_val = int(match.group(1))
                     if 0 <= score_val <= max_score:
                         raw_scores[i] = score_val # Store valid score
                     else:
                         logger.warning(f"Score out of range ({score_val}) for prompt index {i}: '{response[:50]}...'. Retrying.")
                         new_pending_indices.append(i)
                 else:
                     logger.warning(f"Could not parse score digit for prompt index {i}: '{response[:50]}...'. Retrying.")
                     new_pending_indices.append(i)

             pending_indices = new_pending_indices
             if pending_indices:
                 retry_count += 1


        # Process final scores, normalize, handle failures
        final_scores = []
        for i, score_val in enumerate(raw_scores):
            if score_val is not None:
                final_scores.append(float(score_val) / max_score) # Normalize
            else:
                logger.error(f"Failed to get valid score for prompt index {i} after {max_retries} retries. Using None.")
                final_scores.append(None)

        logger.info("Scoring finished.")
        return final_scores