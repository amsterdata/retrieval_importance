# retrieval_importance

Implementation and experimentation code for the paper on _Improving Retrieval-Augmented Large Language Models via Data Importance Learning_.

## Algorithm Implementation

We provide a [Rust-based implementation of the weight learning algorithm](https://github.com/amsterdata/retrieval_importance/blob/main/src/mle/mod.rs) and corresponding [Python bindings](https://github.com/amsterdata/retrieval_importance/blob/main/src/lib.rs) via Pyo3.

## Source Code for the Experiments

### Improving Prediction Quality with Learned Data Importance

#### Question Answering & Data Imputation
 * The experiments for **question answering on the WikiFact relations** are implemented in [wikifact.py](wikifact.py) and TODO. 
 * The experiments for **question answering on the WebQA dataset** are implemented in TODO and [webquestions_gpt3.py](webquestions_gpt3.py).
 * The experiments for **data imputation** are implemented in [imputation.py](imputation.py) and [imputation_gpt3.py](imputation_gpt3.py).

#### Mitigating the Impact of Noise in the Retrieval Corpus

 * The experiments for **question answering on the noisy WikiFact relations** are implemented in [wikifact.py](wikifact.py), where one has to supply the argument  ``-s noise`` to specify the scenario where we add noise to the retrieval corpus.

#### Application to GPT-3.5

TODO

#### Data Importance Beyond Large Language Models

 * The experiment for **improving the precision of a session-based recommender** on ecommerce clicks is implemented in Rust in [reco.rs](src/bin/reco.rs). Note that we cannot share the click data for legal reasons.

### Efficiency & Scalability 

#### Microbenchmark for Optimisations in Computing Subset and Boundary Value Probabilities

 * The corresponding microbenchmarks are implemented in Rust in [generate_iprp.rs](benches/generate_iprp.rs) and [generate_b.rs](benches/generate_b.rs).

#### Microbenchmark for End-to-End Benefits
 * The corresponding microbenchmark is implemented in Rust in [end_to_end_runtime.rs](src/bin/end_to_end_runtime.rs) 

#### Scalability

 * The scalability experiment with a synthetic corpus is implemented in Rust in [synth_runtime.rs](src/bin/synth_runtime.rs).
 * The end-to-end runtimes for the click datasets are computed in Rust in [reco.rs](src/bin/reco.rs).

## Local Installation

 * Requires Python 3.9 and [Rust](https://www.rust-lang.org/tools/install) to be available
 
 1. Clone this repository
 1. Change to the project directory: `cd retrieval_importance`
 1. Create a virtualenv: `python3.9 -m venv venv`
 1. Activate the virtualenv `source venv/bin/activate`
 1. Install the dev dependencies with `pip install -r requirements-dev.txt`
 1. Build the project `maturin develop --release`
