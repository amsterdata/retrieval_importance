use crate::mle::types::Retrieval;
use crate::mle::tensors::{DenseMatrix, DenseTensor};
use crate::mle::{max_distinct_retrieved, max_distinct_utility_contributions};

pub fn mle_importance(
    mut retrievals: Vec<Retrieval>,
    corpus_size: usize,
    k: usize,
    learning_rate: f64,
    num_epochs: usize,
) -> Vec<f64> {

    let mut v = vec![0.5_f64; corpus_size];

    retrievals
        .iter_mut()
        .for_each(|retrieval| {
            retrieval.utility_contributions.iter_mut()
                // TODO Make discretization configurable here
                .for_each(|u| *u = (*u * 100.0).round() / 100.0)
        });

    #[allow(non_snake_case)]
        let N = retrievals.len();

    for _ in 0..num_epochs {
        let g = mle_importance_gradient(
            &retrievals,
            &v,
            k,
            N,
        );

        for i in 0..v.len() {
            v[i] += learning_rate * g[i];

            // Clipping
            if v[i] > 1.0 {
                v[i] = 1.0;
            } else if v[i] < 0.0 {
                v[i] = 0.0;
            }
        }
    }

    v
}

#[allow(non_snake_case)]
fn mle_importance_gradient(
    D_val: &Vec<Retrieval>, // validation set with ranked retrieved samples
    v: &Vec<f64>, // existence variables
    K: usize, // k of knn-classifier,
    N: usize,
) -> Vec<f64> {

    let mut g = vec![0.0_f64; v.len()];

    let M_max = max_distinct_retrieved(D_val);
    let E_max = max_distinct_utility_contributions(D_val);

    let mut IP = DenseMatrix::new(K + 1, M_max + 2);
    let mut RP = DenseMatrix::new(K + 1, M_max + 2);
    let mut B = DenseTensor::new(K + 1, M_max + 2, E_max);

    for retrieval in D_val {
        // TODO maybe reuse a buffer here
        let p = retrieval.existence_probabilities(v);
        let s = additive_any_loss_mle_gradient(
            &retrieval.utility_contributions,
            &p,
            K,
            N,
            &mut IP,
            &mut RP,
            &mut B
        );

        for (retrieved, contribution) in retrieval.retrieved.iter().zip(s.iter()) {
            g[*retrieved] += contribution;
        }
    }

    g
}

#[allow(non_snake_case)]
fn additive_any_loss_mle_gradient(
    utility_contributions: &[f64],
    p: &[f64],
    K: usize,
    N: usize,
    IP: &mut DenseMatrix,
    RP: &mut DenseMatrix,
    B: &mut DenseTensor,
) -> Vec<f64> {

    let num_retrieved = p.len();
    assert_eq!(num_retrieved, utility_contributions.len());

    let mut s = vec![0_f64; num_retrieved];

    compute_prob_from_tensors(p, K, num_retrieved, IP, RP);

    let mut distinct_utility_contributions: Vec<f64> = Vec::new();

    for utility_contribution in utility_contributions {
        if !distinct_utility_contributions.contains(utility_contribution) {
            distinct_utility_contributions.push(*utility_contribution);
        }
    }

    compute_boundary_set_prob_any_loss_from_tensor(
        utility_contributions,
        &distinct_utility_contributions,
        p,
        K,
        num_retrieved,
        B
    );

    for i in 1..num_retrieved + 1 {
        let c = utility_contributions[i-1];

        // G_1
        let mu_1 = (c as f64 / K as f64) / N as f64;
        for k in 0..K {
            for j in 0..k + 1 {
                s[i - 1] += mu_1 * IP[[j,i - 1]] * RP[[k - j,i + 1]];
            }
        }


        // G_2
        for e in 0..distinct_utility_contributions.len() {
            let difference = c - distinct_utility_contributions[e];

            let mu_2 = (difference as f64 / K as f64)  / N as f64;
            for j in 0..K {
                s[i - 1] += mu_2 * IP[[j,i - 1]] * B[[K - j,i + 1, e]];
            }

        }
    }

    s
}

#[allow(non_snake_case,unused)]
pub fn compute_prob_from_tensors(
    p: &[f64],
    K: usize,
    M: usize,
    IP: &mut DenseMatrix,
    RP: &mut DenseMatrix,
) {
    IP[[0,0]] = 1.0;
    RP[[0,M+1]] = 1.0;

    // Required because we reuse un-zeroed memory
    for k in 1..K+1 {
        IP[[k,0]] = 0.0;
        RP[[k,M+1]] = 0.0;
    }
    // Required because we reuse un-zeroed memory
    for j in 1..M {
        IP[[0,j]] = 0.0;
    }

    for j in 1..M+1 {
        IP[[0,j]] = IP[[0,j-1]] * (1.0 - p[j-1]);
        for k in 1..K+1 {
            IP[[k,j]] = IP[[k,j-1]] * (1.0 - p[j-1]) + IP[[k-1, j-1]] * p[j-1];
        }
    }

    for j in (1..M+1).rev() {
        RP[[0,j]] = RP[[0,j+1]] * (1.0 - p[j-1]);
        for k in 1..K+1 {
            RP[[k,j]] = RP[[k,j+1]] * (1.0 - p[j-1]) + RP[[k-1,j+1]] * p[j-1];
        }
    }
}

#[allow(non_snake_case)]
fn compute_boundary_set_prob_any_loss_from_tensor(
    retrieved_costs: &[f64],
    distinct_costs: &Vec<f64>,
    p: &[f64],
    K: usize,
    M: usize,
    B: &mut DenseTensor,
) {
    let size_of_e = distinct_costs.len();

    // Required because we reuse un-zeroed memory
    for i in 1..M+2 {
        for e in 0..size_of_e {
            B[[0, i, e]] = 0.0;
        }
    }
    // Required because we reuse un-zeroed memory
    for k in 1..K+1 {
        for e in 0..size_of_e {
            B[[k, M + 1, e]] = 0.0;
        }
    }

    for i in (1..M+1).rev() {
        for k in 1..K+1 {
            for e in 0..size_of_e {
                // This expression should be vectorizable (compute for all e at once)
                B[[k,i,e]] = B[[k,i+1,e]] * (1.0 - p[i-1]) + B[[k-1,i+1,e]] * p[i-1];
                if k==1 && distinct_costs[e] == retrieved_costs[i-1] {
                    B[[k,i,e]] += p[i-1];
                }
            }
        }
    }
}