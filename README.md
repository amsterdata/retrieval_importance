# retrieval_importance

Implementation and experimentation code for the submission _Improving Retrieval-Augmented Large Language Models with Data-Centric Refinement_ to NeurIPS'23.

## Algorithm Implementation

We provide a Rust-based [implementation of the weight learning algorithm](https://github.com/anon756/retrieval_importance/blob/main/src/lib.rs) (callable from Python via pyo3) and several [helper functions](https://github.com/anon756/retrieval_importance/blob/main/python/retrieval_importance/utils.py). 

## Source Code for the Experiments

 * The experiments for **question answering** are implemented in [question_answering_url.py](wikifact.py). The commandline argument ``-m`` specifies the metric to use (LOO, reweight, prune) and the argument  ``-s`` specifies the scenario ('raw' for no change in the retrieval corpus, 'noise' for adding noise to the retrieval corpus, and 'fake' for adding new wiki sources to the retrieval corpus).
 * The experiments for **data imputation** are implemented in  [imputation_experiment.py](imputation.py).
 * The experiment for the **computational performance** is implemented in [src/bin/synth_runtime.rs](synth_runtime.rs). This experiment can be executed via ``RUSTFLAGS="-C target-cpu=native" cargo run --release --bin synth_runtime``.

## Installation for development

 * Requires Python 3.9 and [Rust](https://www.rust-lang.org/tools/install) to be available
 
 1. Clone this repository
 1. Change to the project directory: `cd retrieval_importance`
 1. Create a virtualenv: `python3.9 -m venv venv`
 1. Activate the virtualenv `source venv/bin/activate`
 1. Install the dev dependencies with `pip install -r requirements-dev.txt`
 1. Build the project `maturin develop --release`
