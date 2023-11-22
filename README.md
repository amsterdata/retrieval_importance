# retrieval_importance

Implementation and experimentation code for the paper on _Improving Retrieval-Augmented Large Language Models via Data Importance Learning_.

## Algorithm Implementation

We provide a [Rust-based implementation of the weight learning algorithm](https://github.com/amsterdata/retrieval_importance/blob/main/src/mle/mod.rs) and corresponding [Python bindings](https://github.com/amsterdata/retrieval_importance/blob/main/src/lib.rs) via Pyo3.

## Source Code for the Experiments

 * The experiments for **question answering** are implemented in [question_answering_url.py](wikifact.py). The commandline argument ``-m`` specifies the metric to use (LOO, reweight, prune) and the argument  ``-s`` specifies the scenario ('raw' for no change in the retrieval corpus, 'noise' for adding noise to the retrieval corpus, and 'fake' for adding new wiki sources to the retrieval corpus).
 * The experiments for **data imputation** are implemented in  [imputation_experiment.py](imputation.py).
 * The experiment for the **computational performance** is implemented in [src/bin/synth_runtime.rs](synth_runtime.rs). This experiment can be executed via ``RUSTFLAGS="-C target-cpu=native" cargo run --release --bin synth_runtime``.

## Local Installation

 * Requires Python 3.9 and [Rust](https://www.rust-lang.org/tools/install) to be available
 
 1. Clone this repository
 1. Change to the project directory: `cd retrieval_importance`
 1. Create a virtualenv: `python3.9 -m venv venv`
 1. Activate the virtualenv `source venv/bin/activate`
 1. Install the dev dependencies with `pip install -r requirements-dev.txt`
 1. Build the project `maturin develop --release`
