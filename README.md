# Graph Meme Pool (GMP)

This repository contains the source code for the research paper:

**"Selection in the Meme Pool: Graph-Based Evolution of Textual Content"**

*Karlo Babić*

(Link to paper: TODO)

## Abstract

This paper presents a computational framework for simulating the evolution of text by modeling textual units as memes - replicable cultural information analogous to genes. Memes occupy nodes in a graph, where edges represent pathways of interaction. Memes propagate probabilistically along these edges based on weights. Following propagation, a fitness function evaluates each meme, guiding a selection process based on configurable criteria. This selection determines whether Large Language Models (LLMs) then introduce variation by mutating or merging memes, adapting content during transmission. Through iterative cycles of propagation, selection, and variation, the system models how memes evolve, spread, and adapt within networked populations.

## Framework Overview

The Graph Meme Pool (GMP) framework simulates the lifecycle of textual memes within a networked population. Main components include:

1.  **Graph Representation:** Memes inhabit nodes in a directed, weighted graph. The graph structure (e.g., small-world, multiple small-worlds) is configurable.
2.  **Propagation:** Memes spread stochastically between nodes based on edge weights.
3.  **Selection:** Nodes evaluate received memes against their current meme using a fitness function (either a dedicated model like RoBERTa for sentiment, or LLM-based scoring). A word count penalty can also be applied. Based on fitness comparison against a threshold, a decision is made to:
    *   Keep the current meme.
    *   Trigger Mutation of a superior incoming meme.
    *   Trigger Merging of the current meme with a comparably-fit incoming meme.
4.  **Variation:** Large Language Models (LLMs) generate new meme candidates:
    *   **Mutation:** Slightly alters a meme's meaning, style, or tone.
    *   **Merging:** Modifies one meme to be semantically closer to another.
5.  **Iterative Evolution:** The cycle of propagation, selection, and variation repeats over multiple generations, driving meme evolution.

## Running Simulations

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure your simulation:**
    *   Edit `config.yaml` to set parameters for graph generation, simulation (generations, threshold), LLM models and prompts, fitness model, and visualization options.
    *   The `config.yaml` file specifies Hugging Face model paths for the LLM (e.g., `microsoft/Phi-3-mini-4k-instruct`) and embedding models. These will be downloaded automatically on first use if not cached.
    *   If you are using a gated model or one that requires an API key through a specific service (not directly via Hugging Face `transformers`), you might need to set environment variables or update the `llm_service.py` accordingly. The provided `config.yaml` has an `api_key` field commented out.
    *   Place your initial meme(s) in `init_memes.txt`.

3.  **Run the main script:**
    ```bash
    python main.py
    ```

4.  **Output:**
    *   Simulation logs will be printed to the console and saved to `simulation.log` (or as configured).
    *   Graphs states (`.json`) and propagation history are saved in the `saved/` directory.
    *   Visualizations (plots, graph drawings) are saved in the `vis/` directory.

5.  **Resuming Simulations:**
    If a `graph_basename.json` file exists in the `graph_save_dir` (as specified in `config.yaml`), the simulation will attempt to load this graph and resume from the last completed generation. Otherwise, it will create a new graph.

## Configuration (`config.yaml`)

The `config.yaml` file is central to controlling the simulation. Main sections include:

*   `paths`: File paths for initial memes, save directories, log file.
*   `graph_generation`: Type of graph (`small_worlds`, `example`), parameters (nodes, groups, probabilities), and initial meme assignment strategy.
*   `simulation`: Number of generations, selection threshold, fitness model choice (Hugging Face path or LLM-based).
*   `llm`: Hugging Face path for the LLM, temperature settings for different operations, max new tokens, and prompts for mutation, merging, and LLM-based scoring.
*   `embeddings`: Model paths for text embeddings.
*   `visualization`: Toggles for various per-generation and final plots, and plot settings.

## Citation

If you use this code or framework in your research, please cite the original paper:

```bibtex
TODO
K. Babić. "Selection in the Meme Pool: Graph-Based Evolution of Textual Content." ECAI, 2025, Bologna, Italy. Accepted for publication.
```

## Semantic Space Evolution Animation

The following animation visualizes meme evolution in a 2D semantic space (BERT embeddings projected with t-SNE) over 300 generations, as discussed in the paper (Section 4.1, Figure 3). Larger circles are group centroids connected chronologically, tracking each group's semantic trajectory.

![Animated visualization of meme evolution in semantic space](https://github.com/karlo-babic/graph-meme-pool/blob/main/semantic_space_evolution.gif?raw=true)
