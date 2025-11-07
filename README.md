[Webpage](https://karlo-babic.github.io/graph-meme-pool)

# Graph Meme Pool (GMP) v2-dev

Graph Meme Pool (GMP) is a computational framework for simulating the evolution of text by modeling textual units as memes - replicable cultural information analogous to genes. This branch contains the **v2-dev source code**, which extends the concepts from the original research paper with new features like dynamic graph topologies, advanced selection mechanisms, and a robust experiment management system.

Memes occupy nodes in a graph where edges represent pathways of interaction. They propagate probabilistically along these edges, after which a fitness function evaluates them to guide a selection process. This selection determines whether Large Language Models (LLMs) introduce variation by merging memes, adapting content during transmission. Through iterative cycles of propagation, selection, and variation, the system models how memes evolve, spread, and adapt within networked populations that can themselves change over time.

The source code for the research paper **"Selection in the Meme Pool: Graph-Based Evolution of Textual Content"** can be found in the [v1 branch](https://github.com/karlo-babic/graph-meme-pool/tree/v1).

## Framework Overview

The GMP v2 framework simulates the lifecycle of textual memes with several enhancements:

1.  **Graph Representation:** Memes inhabit nodes in a directed, weighted graph. The initial graph structure (e.g., `small_worlds`) is configurable.
2.  **Propagation:** Memes spread stochastically between nodes based on edge weights, modeling information flow.
3.  **Selection:** Nodes evaluate received memes against their current meme using a fitness function (e.g., a sentiment model or LLM-based scoring). The best incoming candidate is selected based on a strategy like `fitness_similarity_product`, which balances raw fitness with semantic similarity. A decision is then made to:
    *   **Keep** the current meme if the incoming candidate is not a significant improvement.
    *   Trigger a **Converge Merge** if the candidate is comparably fit and semantically very similar.
    *   Trigger an **Influence Merge** if the candidate is comparably fit but only moderately similar.
4.  **Variation (LLM-driven):** LLMs generate new meme candidates by blending existing ones:
    *   **Converge Merge:** Modifies one meme to be semantically closer to another, targeting refinement.
    *   **Influence Merge:** Modifies one meme using another as inspiration, promoting innovative recombination.
5.  **Dynamic Topology:** The graph structure itself can evolve. Based on configurable rules, nodes can be removed or created during the simulation:
    *   **Death:** Persistently low-fitness nodes are removed.
    *   **Fusion:** Highly similar, connected nodes are merged into a single new node.
    *   **Division:** High-fitness nodes split into two distinct new nodes, introducing novelty.
6.  **Iterative Evolution:** The cycle repeats over multiple generations, driving the co-evolution of memes and the network structure.

## Main Features of v2

Compared to the v1 system described in the paper, v2 introduces major improvements:

*   **Dynamic Graph Evolution:** The network is no longer static. Nodes can be created (`Division`) or removed (`Death`, `Fusion`) during a simulation, allowing the population structure to adapt.
*   **Advanced Selection & Merging:** The selection process now considers both fitness and semantic similarity. It employs two distinct merging strategies (`converge` and `influence`) for more nuanced textual evolution.
*   **Experiment Management:** Each simulation run is saved in a dedicated, timestamped directory, containing its configuration, logs, checkpoints, and visualizations for better organization and reproducibility.
*   **Resumable Simulations:** Long simulations can be stopped and resumed from the last completed generation, saving significant time and computational resources.

## Running Simulations

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/karlo-babic/graph-meme-pool.git
cd graph-meme-pool
pip install -r requirements.txt
```

### 2. Configuration

Simulations are controlled by YAML files in the `configs/` directory.

1.  **Create a Configuration:** Copy an existing config (e.g., `configs/potato_friends_sentiment_dynamic_graph.yaml`) or create a new one.
2.  **Set Parameters:** Edit your config file to define the graph structure, simulation length, fitness models, LLM prompts, and dynamic graph rules.
3.  **Initial Memes:** Place your starting meme(s) in `data/init_memes.txt`, with each meme on a new line.

The configuration specifies Hugging Face model paths for the LLM (e.g., `microsoft/Phi-3-mini-4k-instruct`) and embedding models. These will be downloaded automatically on first use. If using a model that requires an API key, you may need to set environment variables or adjust `src/llm_service.py`.

### 3. Running a New Simulation

Use the `src/main.py` script and point to your configuration file with the `--config` flag.

```bash
python src/main.py --config configs/your_config_file.yaml
```

### 4. Resuming a Simulation

To resume a previous run, use the `--resume` flag and provide the path to the experiment directory.

```bash
python src/main.py --resume experiments/20240521-103000_my_simulation_run/
```

### 5. Output Structure

All outputs for a given run are saved in a dedicated directory inside `experiments/`, for example: `experiments/20240521-103000_my_simulation_run/`. This directory contains:
*   `config.yaml`: A copy of the configuration used for the run.
*   `simulation.log`: Detailed logs of the simulation process.
*   `checkpoints/`:
    *   `graph_state_final.json`: The complete state of the graph at the end of the simulation, used for resuming.
    *   `propagation_history.json`: A record of all successful propagation events.
*   `visualizations/`: All generated plots, graphs, and animations.

## Configuration (`config.yaml`)

The YAML configuration file is central to controlling the simulation. Main sections include:

*   `run_name`: A descriptive name for your experiment directory.
*   `graph_generation`: Defines the initial network structure (`small_worlds` or `example`) and its parameters (nodes, groups, connectivity).
*   `dynamic_graph`: Controls the evolution of the network topology.
    *   `enabled`: Set to `true` to activate dynamic graph features.
    *   `actions`: A list of rules to apply, including `death`, `fusion`, and `division`, each with its own parameters (e.g., fitness thresholds, similarity scores).
*   `simulation`: Core evolutionary parameters.
    *   `generations`: The total number of cycles to run.
    *   `selection_strategy`: Method for choosing the best candidate meme (`fitness` or `fitness_similarity_product`).
    *   `merge_converge_similarity_threshold`: The semantic similarity score above which a `converge` merge is triggered.
    *   `merge_influence_similarity_threshold`: The similarity score for an `influence` merge.
*   `llm`: Configures the Large Language Model.
    *   `huggingpath`: The Hugging Face path to the model.
    *   `prompt_mutate`, `prompt_merge_converge`, `prompt_merge_influence`: The specific instructions given to the LLM for each variation type.
*   `embeddings`: Specifies the model for generating text embeddings, used for similarity calculations and visualization.
*   `visualization`: Toggles and settings for all plots and output images.

## Citation

If you use this code or framework in your research, please cite the original paper which introduced the foundational concepts:

```bibtex
@inproceedings{babic2025selection,
  title={Selection in the Meme Pool: Graph-Based Evolution of Textual Content},
  author={Babi{\'c}, Karlo},
  booktitle={European Conference on Artificial Intelligence (ECAI 2025)},
  volume={413},
  pages={1825--1832},
  year={2025}
}
```