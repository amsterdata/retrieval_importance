extern crate rand;

use retrieval_importance::mle as mle;
use mle::types::Retrieval;
use std::time::Instant;
use rand::thread_rng;
use rand::distributions::{Distribution, Bernoulli};

const LEARNING_RATE: f64 = 0.1;
const NUM_STEPS: usize = 10;

//RUSTFLAGS="-C target-cpu=native" cargo run --release --bin end_to_end_runtime |tee /Users/ssc/nextcloud/papers/mle-neurips/notebooks/end_to_end_runtime.csv
fn main() {

    let mut rng = thread_rng();
    let prob_of_right_answer = 0.25;
    let bernoulli = Bernoulli::new(prob_of_right_answer).unwrap();
    let corpus_size = 100_000;
    let num_repetitions = 7;

    println!("variant,N,corpus_size,d,k,duration");


    #[allow(non_snake_case)]
    //for N in [1_000, 10_000, 100_000, 1_000_000] {
    for N in [2_000] {
        for (k, d) in [(10, 500), (20, 500), (50, 500), (100, 500)] {

            let all_retrieved: Vec<Retrieval> = (0..N)
                .map(|_| {
                    let retrieved: Vec<usize> =
                        rand::seq::index::sample(&mut rng, corpus_size, d).into_vec();

                    let utilities: Vec<f64> = (0..d)
                        .map(|_| {
                            if bernoulli.sample(&mut rng) {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();

                    Retrieval::new(retrieved, utilities)
                })
                .collect();

            for _ in 0..num_repetitions {
                let all_retrieved_copy = all_retrieved.clone();

                let start_time = Instant::now();
                let _v = mle::mle_importance(
                    all_retrieved_copy,
                    corpus_size,
                    None,
                    k,
                    LEARNING_RATE,
                    NUM_STEPS,
                    4
                );

                let duration = (Instant::now() - start_time).as_millis();
                println!("ALL_OPTIMISATIONS_PARALLEL,{},{},{},{},{}",
                         N, corpus_size, d, k, duration);
            }

                for _ in 0..num_repetitions {
                    let all_retrieved_copy = all_retrieved.clone();

                    let start_time = Instant::now();
                    let _v = mle::mle_importance(
                        all_retrieved_copy,
                        corpus_size,
                        None,
                        k,
                        LEARNING_RATE,
                        NUM_STEPS,
                        1
                    );

                    let duration = (Instant::now() - start_time).as_millis();
                    println!("ALL_OPTIMISATIONS,{},{},{},{},{}", N, corpus_size, d, k, duration);
            }

            for _ in 0..num_repetitions {
                let all_retrieved_copy = all_retrieved.clone();

                let start_time = Instant::now();
                let _v = mle::end_to_end_reuse::mle_importance(
                    all_retrieved_copy,
                    corpus_size,
                    k,
                    LEARNING_RATE,
                    NUM_STEPS,
                );

                let duration = (Instant::now() - start_time).as_millis();
                println!("TENSORS_REUSES,{},{},{},{},{}", N, corpus_size, d, k, duration);
            }

            for _ in 0..num_repetitions {
                let all_retrieved_copy = all_retrieved.clone();

                let start_time = Instant::now();
                let _v = mle::end_to_end_tensors::mle_importance(
                    all_retrieved_copy,
                    corpus_size,
                    k,
                    LEARNING_RATE,
                    NUM_STEPS,
                );

                let duration = (Instant::now() - start_time).as_millis();
                println!("TENSORS,{},{},{},{},{}", N, corpus_size, d, k, duration);
            }

            for _ in 0..num_repetitions {
                let all_retrieved_copy = all_retrieved.clone();

                let start_time = Instant::now();
                let _v = mle::end_to_end_notensors::mle_importance(
                    all_retrieved_copy,
                    corpus_size,
                    k,
                    LEARNING_RATE,
                    NUM_STEPS,
                );

                let duration = (Instant::now() - start_time).as_millis();
                println!("BASE,{},{},{},{},{}", N, corpus_size, d, k, duration);
            }

        }
    }
}
