# retrieval_importance

Implementation and experimentation code for the paper on _Improving Retrieval-Augmented Large Language Models via Data Importance Learning_.

## Algorithm Implementation

We provide a [Rust-based implementation of the weight learning algorithm](https://github.com/amsterdata/retrieval_importance/blob/main/src/mle/mod.rs) and corresponding [Python bindings](https://github.com/amsterdata/retrieval_importance/blob/main/src/lib.rs) via Pyo3.

Below is example code showing how to learn data importance weights for a retrieval corpus collected from the web. In addition, we provide an [executable notebook with an end-to-end toy example](https://github.com/amsterdata/retrieval_importance/blob/main/example-question-answering.ipynb) that demonstrates how to improve prediction quality via data importance learning.

```python
from retrieval_importance import learn_importance, encode_retrievals, encode_groups, grouped_weights

# Retrieval corpus for a question answering task collected from the web
retrieval_corpus = [
  { "question": "The author of Charon's Landing is",
    "correct_answers": ["Jack Du Brul"],
    "source_websites": ["en.wikipedia.org", "www.goodreads.com", "books.google.com", ...],
    "generated_answers": ["Jack Du Brul", "Barbara Hambly", "Barbara Hambly", ...] },
  { "question": "The author of Cathedral of the Sea is",
    "correct_answers": ["Ildefonso Falcones", "Ildefonso Falcones de Sierra"],
    "source_websites": ["en.wikipedia.org", "actualidadliteratura.com", "www.goodreads.com", ...],
    "generated_answers": ["Ildefonso Falcones", "Ildefonso Falcones", "J. K. Rowling", ...]
  },
  ...
]

# Accuracy as utility function
def utility(retrieval, prediction):
    if prediction in retrieval["correct_answers"]:
        return 1.0
    else:
        return 0.0

# Grouping function to define data sources (web domains in this case)
def group_by_domain(source_website):    
    url_parts = tldextract.extract(source_website)
    return f'{url_parts.domain}.{url_parts.suffix}'

# Encode and group retrieval corpus
encoded_corpus, mapping = encode_retrievals(retrieval_corpus, "source_websites", "generated_answers", utility)
grouping, group_mapping = encode_groups(mapping, group_by_domain)

# Importance weight learning
importance_weights = learn_importance(encoded_corpus,
                                      k=10,
                                      learning_rate=40.0,
                                      num_steps=50,
                                      n_jobs=-1,
                                      grouping=group_by_domain)

# Importances per data source (web domains in this case)
importance_weights_by_domain = grouped_weights(importance_weights, group_by_domain, group_mapping)

# The weights can subsequently be inspected and use to prune low-quality data sources from the retrieval corpus
```


## Source Code for the Experiments

### Improving Prediction Quality with Learned Data Importance

#### Question Answering & Data Imputation
 * The experiments for **question answering on the selection of 70 WikiFact relations** are implemented in [wikifact.py](wikifact.py) and taken from the [HELM API](https://crfm-helm.readthedocs.io/en/latest/scenarios/#helm.benchmark.scenarios.wikifact_scenario) for GPT-3.5. 
 * The experiments for **question answering on the WebQA dataset** are implemented in [webquestions.py](webquestions.py). and [webquestions_gpt3.py](webquestions_gpt3.py).
 * The experiments for **data imputation** are implemented in [imputation.py](imputation.py) and [imputation_gpt3.py](imputation_gpt3.py).

#### Mitigating the Impact of Noise in the Retrieval Corpus

 * The experiments for **question answering on the noisy WikiFact relations** are implemented in [wikifact.py](wikifact.py), where one has to supply the argument  ``-s noise`` to specify the scenario where we add noise to the retrieval corpus.

#### Application to GPT-3.5

 * The experiments for improving the retrieval-augmented GPT-3.5 model are implemented in [importance_gpt3.py](wimportance_gpt3.py).

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
