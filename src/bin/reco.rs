extern crate retrieval_importance;
extern crate num_cpus;
extern crate average;

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use retrieval_importance::mle::types::Retrieval;
use retrieval_importance::mle::mle_importance;

use retrieval_importance::app_reco::vmis_index::VMISIndex;
use retrieval_importance::app_reco::io::{TrainingSessionId, ItemId, read_test_data_evolving};
use retrieval_importance::app_reco::similarity_indexed::SimilarityComputationNew;
use retrieval_importance::app_reco::linear_score;
use std::cmp::Ordering;
use itertools::Itertools;
use retrieval_importance::app_reco::SessionScore;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use average::MeanWithError;

fn main() {

    let m = 5000;
    let k_plus_b = 500;
    let validation_size = 5000;

    let learning_rate = 40.0;
    let num_epochs = 100;
    let num_predictions = 1;

    let seeds = [42, 1311, 1810, 67, 12345, 1701, 1801, 805, 2056, 912];

    let index = VMISIndex::new_from_csv("TRAIN_CLICKS_CANNOT_BE_SHARED_FOR_LEGAL_REASONS.txt", m, 1.0);
    let num_sessions_in_corpus = index.session_to_items_sorted.len();

    let evolving = read_test_data_evolving("TEST_CLICKS_CANNOT_BE_SHARED_FOR_LEGAL_REASONS.txt");


    println!("# Retrieval corpus with {} sessions, validation & test set with {} sessions",
             num_sessions_in_corpus, evolving.len());

    let mut evolving_session_keys: Vec<_> = evolving.keys().map(|k| k).collect();

    for k in [10, 50, 100] {

        let mut sum_imp = 0.0;
        let mut sum_vanilla_acc = 0.0;
        let mut sum_fractions = 0.0;

        let mut corpus_inference_durations = Vec::new();
        let mut weight_learning_durations = Vec::new();
        let mut tuning_durations = Vec::new();

        for seed in seeds {

            evolving_session_keys.sort();
            let mut rng = StdRng::seed_from_u64(seed);
            evolving_session_keys.shuffle(&mut rng);

            let validation_keys: Vec<_> = evolving_session_keys[0..validation_size].to_vec();

            let validation_sessions: HashMap<TrainingSessionId, Vec<ItemId>> = evolving.clone()
                .into_iter()
                .filter(|(k, _v)| validation_keys.contains(&k))
                .collect();

            let test_sessions: HashMap<TrainingSessionId, Vec<ItemId>> = evolving.clone()
                .into_iter()
                .filter(|(k, _v)| !validation_keys.contains(&k))
                .collect();


            let mut retrievals = Vec::with_capacity(validation_sessions.len());
            let mut observed_session_ids = HashSet::new();

            let corpus_inference_start_time = Instant::now();

            for (_id, evolving_session) in validation_sessions.iter() {
                let last_item = evolving_session.last().unwrap();
                let without_last_item = &evolving_session[0..(evolving_session.len() - 1)];

                let neighbors = index.find_neighbors(without_last_item, k_plus_b, m).into_sorted_vec();

                let mut retrieved: Vec<usize> = Vec::with_capacity(neighbors.len());
                let mut utilities = Vec::with_capacity(neighbors.len());

                for neighbor in neighbors {
                    observed_session_ids.insert(neighbor.id);
                    let neighbor_items = index.items_for_session(&neighbor.id);

                    let utility = if neighbor_items.contains(&last_item) {
                        1.0
                    } else {
                        0.0
                    };
                    retrieved.push(neighbor.id as usize);
                    utilities.push(utility);
                }

                retrievals.push(Retrieval::new(retrieved, utilities));
            }
            let corpus_inference_duration =
                (Instant::now() - corpus_inference_start_time).as_millis();

            //println!("# Learning importance weights...");
            let weight_learning_start_time = Instant::now();
            let v = mle_importance(retrievals, num_sessions_in_corpus, None, k, learning_rate,
                                   num_epochs, num_cpus::get_physical());
            let weight_learning_duration =
                (Instant::now() - weight_learning_start_time).as_millis();

            let thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99];

            let tuning_start_time = Instant::now();

            let mut best_threshold = 0.5;
            let mut best_num_correct = 0;
            for threshold in thresholds {
                let num_correct = predict_from_neighbors_pruned(&validation_sessions, &index, k_plus_b,
                                                                k, m, num_predictions, threshold, &v, &observed_session_ids);

                if num_correct > best_num_correct {
                    best_num_correct = num_correct;
                    best_threshold = threshold;
                }
            }
            let tuning_duration = (Instant::now() - tuning_start_time).as_millis();

            let num_correct_pruned = predict_from_neighbors_pruned(&test_sessions, &index, k_plus_b,
                                                                   k, m, num_predictions, best_threshold, &v, &observed_session_ids);

            let acc_pruned = num_correct_pruned as f64 / test_sessions.len() as f64;

            let num_correct_vanilla = predict_from_neighbors(&test_sessions, &index, k, m,
                                                             num_predictions);
            let acc_vanilla = num_correct_vanilla as f64 / test_sessions.len() as f64;

            let num_retained = v.iter().filter(|x| **x >= best_threshold).count();
            let fraction = (num_retained as f64 / num_sessions_in_corpus as f64) * 100.0;
            println!("{:.4} ({}/{}) -> {:.4} ({}/{}) with {:.4}% of sessions",
                     acc_vanilla, num_correct_vanilla, test_sessions.len(), acc_pruned,
                     num_correct_pruned, test_sessions.len(), fraction);
            sum_imp += acc_pruned - acc_vanilla;
            sum_vanilla_acc += acc_vanilla;
            sum_fractions += fraction;

            println!(
                "# Runtime - corpus: {}, weights: {}, tuning: {}",
                corpus_inference_duration,
                weight_learning_duration,
                tuning_duration
            );

            corpus_inference_durations.push(corpus_inference_duration as f64);
            weight_learning_durations.push(weight_learning_duration as f64);
            tuning_durations.push(tuning_duration as f64);

        }
        println!("---------------------------------------");
        println!("{}: {} - {} with {}", k, sum_vanilla_acc / seeds.len() as f64, sum_imp / seeds.len() as f64, sum_fractions / seeds.len() as f64);

        let corpus_inference_stats: MeanWithError =
            corpus_inference_durations.into_iter().collect();
        let weight_learning_duration_stats: MeanWithError =
            weight_learning_durations.into_iter().collect();
        let tuning_duration_stats: MeanWithError =
            tuning_durations.into_iter().collect();

        println!(
            "Runtimes - corpus: {:.1} (+- {:.1}), weights:  {:.1} (+- {:.1}), tuning:  {:.1} (+- {:.1})",
            corpus_inference_stats.mean(),
            corpus_inference_stats.error().sqrt(),
            weight_learning_duration_stats.mean(),
            weight_learning_duration_stats.error().sqrt(),
            tuning_duration_stats.mean(),
            tuning_duration_stats.error().sqrt(),
        );
        println!("---------------------------------------");
    }
}

fn predict_from_neighbors(
    sessions: &HashMap<TrainingSessionId, Vec<ItemId>>,
    index: &VMISIndex,
    k: usize,
    m: usize,
    num_predictions: usize,
) -> usize {
    let mut num_correct = 0;

    for (_id, evolving_session) in sessions.iter() {
        let last_item = evolving_session.last().unwrap();
        let without_last_item = &evolving_session[0..(evolving_session.len() - 1)];

        let neighbors = index.find_neighbors(without_last_item, k, m).into_sorted_vec();

        let mut item_counts = HashMap::new();

        let nc = neighbors.clone();

        for neighbor in neighbors {
            let mw = match_weight(index, neighbor.id as usize, &without_last_item);
            for item in index.items_for_session(&neighbor.id) {
                let count = item_counts.entry(item).or_insert_with(|| 0.0);
                *count += mw;
            }
        }

        let predicted = prediction_from_scores(index, &nc, without_last_item, num_predictions);
        if predicted.contains(&(*last_item as usize)) {
            num_correct += 1;
        }
    }
    num_correct
}

fn match_weight(index: &VMISIndex, scored_session_id: usize, evolving_session: &[u64]) -> f64 {
    let training_item_ids: &[u64] = index.items_for_session(&(scored_session_id as u32));

    let (first_match_index, _) = evolving_session
        .iter()
        .rev()
        .enumerate()
        .find(|(_, item_id)| training_item_ids.contains(*item_id))
        .unwrap();

    let first_match_pos = first_match_index + 1;

    linear_score(first_match_pos)
}

fn predict_from_neighbors_pruned(
    sessions: &HashMap<TrainingSessionId, Vec<ItemId>>,
    index: &VMISIndex,
    kplusb: usize,
    k: usize,
    m: usize,
    num_predictions: usize,
    threshold: f64,
    v: &[f64],
    observed_session_ids: &HashSet<u32>,
) -> usize {
    let mut num_correct = 0;
    for (_id, evolving_session) in sessions.iter() {
        let last_item = evolving_session.last().unwrap();
        let without_last_item = &evolving_session[0..(evolving_session.len() - 1)];

        let neighbors = index.find_neighbors(without_last_item, kplusb, m).into_sorted_vec();

        let pruned_neighbors: Vec<SessionScore> = neighbors.into_iter()
            .filter(|neighbor| {
                let nid = neighbor.id as usize;
                if observed_session_ids.contains(&(nid as u32)) {
                    v[nid] >= threshold
                } else {
                    true
                }
            })
            .take(k)
            .collect();

        let predicted = prediction_from_scores(index, &pruned_neighbors, without_last_item,
                                               num_predictions);
        if predicted.contains(&(*last_item as usize)) {
            num_correct += 1;
        }
    }

    num_correct
}

fn prediction_from_scores(
    index: &VMISIndex,
    neighbors: &Vec<SessionScore>,
    without_last_item: &[ItemId],
    how_many: usize,
) -> Vec<usize> {

    let mut item_scores = HashMap::new();
    for neighbor in neighbors {
        let weight = match_weight(index, neighbor.id as usize, &without_last_item);
        for item in index.items_for_session(&neighbor.id) {
            let scores = item_scores.entry(item).or_insert_with(|| 0.0);
            *scores += weight * neighbor.score;
        }
    }

    item_scores
        .iter_mut()
        .sorted_by(|a, b| {
            match a.1.partial_cmp(&b.1) {
                Some(Ordering::Less) => Ordering::Greater,
                Some(Ordering::Greater) => Ordering::Less,
                _ => a.0.cmp(b.0)
            }
        })
        .map(|(item, _weight)| **item as usize)
        .take(how_many)
        .collect()
}