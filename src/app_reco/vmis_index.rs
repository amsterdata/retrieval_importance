use crate::app_reco::similarity_indexed::SimilarityComputationNew;
use crate::app_reco::SessionScore;
use crate::app_reco::SessionTime;
use dary_heap::OctonaryHeap;
use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::error::Error;

use itertools::Itertools;

pub struct VMISIndex {
    pub(crate) item_to_top_sessions_ordered: HashMap<u64, Vec<u32>>,
    pub session_to_max_time_stamp: Vec<u32>,
    pub(crate) item_to_idf_score: HashMap<u64, f64>,
    pub session_to_items_sorted: Vec<Vec<u64>>,
    pub session_id_sorted: Vec<usize>,
}

impl VMISIndex {
    pub fn new_from_csv(path_to_training: &str, m_most_recent_sessions: usize, idf_weighting: f64) -> Self {
        let data_train = read_from_file(path_to_training);
        let (
            historical_sessions_train,
            _historical_sessions_id_train,
            historical_sessions_max_time_stamp,
            mut session_id_sorted,
        ) = data_train.unwrap();

        session_id_sorted.dedup();

        let (
            item_to_top_sessions_ordered,
            item_to_idf_score,
            _session_to_items_sorted,
        ) = prepare_hashmap(
            &historical_sessions_train,
            &historical_sessions_max_time_stamp,
            m_most_recent_sessions,
            idf_weighting,
        );

        VMISIndex {
            item_to_top_sessions_ordered,
            session_to_max_time_stamp: historical_sessions_max_time_stamp,
            item_to_idf_score,
            session_to_items_sorted: historical_sessions_train,
            session_id_sorted,
        }
    }

}

impl SimilarityComputationNew for VMISIndex {
    fn items_for_session(&self, session: &u32) -> &[u64] {
        &self.session_to_items_sorted[*session as usize]
    }

    fn idf(&self, item: &u64) -> f64 {
        self.item_to_idf_score[item]
    }

    fn find_neighbors(
        &self,
        evolving_session: &[u64],
        k: usize,
        m: usize,
    ) -> BinaryHeap<SessionScore> {
        // We use a d-ary heap for the (timestamp, session_id) tuple, a hashmap for the (session_id, score) tuples, and a hashmap for the unique items in the evolving session
        let mut heap_timestamps = OctonaryHeap::<SessionTime>::with_capacity(m);
        let mut session_similarities = HashMap::with_capacity(m);
        let len_evolving_session = evolving_session.len();
        let mut unique = evolving_session.iter().clone().collect_vec();
        unique.sort_unstable();
        unique.dedup();

        let qty_unique_session_items = unique.len() as f64;

        let mut hash_items = HashMap::with_capacity(len_evolving_session);

        //  Loop over items in evolving session in reverse order
        for (pos, item_id) in evolving_session.iter().rev().enumerate() {
            // Duplicate items: only calculate similarity score for the item in the farthest position in the evolving session
            match hash_items.insert(*item_id, pos) {
                Some(_) => {}
                None => {
                    // Find similar sessions in training data
                    if let Some(similar_sessions) = self.item_to_top_sessions_ordered.get(item_id) {
                        let decay_factor =
                            (len_evolving_session - pos) as f64 / qty_unique_session_items;
                        // Loop over all similar sessions.
                        'session_loop: for session_id in similar_sessions {
                            match session_similarities.get_mut(session_id) {
                                Some(similarity) => *similarity += decay_factor,
                                None => {
                                    let session_time_stamp =
                                        self.session_to_max_time_stamp[*session_id as usize];
                                    if session_similarities.len() < m {
                                        session_similarities.insert(*session_id, decay_factor);
                                        heap_timestamps.push(SessionTime::new(
                                            *session_id,
                                            session_time_stamp,

                                        ));
                                    } else {
                                        let mut bottom = heap_timestamps.peek_mut().unwrap();
                                        if session_time_stamp > bottom.time {
                                            // Remove the the existing minimum time stamp.
                                            session_similarities
                                                .remove_entry(&bottom.session_id);
                                            // Set new minimum timestamp
                                            session_similarities
                                                .insert(*session_id, decay_factor);
                                            *bottom = SessionTime::new(
                                                *session_id,
                                                session_time_stamp,
                                            );
                                        } else {
                                            break 'session_loop;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Return top-k
        let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
        for (session_id, score) in session_similarities.iter() {
            if closest_neighbors.len() < k {
                let scored_session = SessionScore::new(*session_id, *score);
                closest_neighbors.push(scored_session);
            } else {
                let mut bottom = closest_neighbors.peek_mut().unwrap();
                if score > &bottom.score {
                    let scored_session = SessionScore::new(*session_id, *score);
                    *bottom = scored_session;
                } else if (score - bottom.score).abs() < f64::EPSILON
                    && (self.session_to_max_time_stamp[*session_id as usize]
                    > self.session_to_max_time_stamp[bottom.id as usize])
                {
                    let scored_session = SessionScore::new(*session_id, *score);
                    *bottom = scored_session;
                }
            }
        }
        // Closest neigbours contain unique session_ids and corresponding top-k similarity scores
        closest_neighbors
    }
}

pub(crate) fn prepare_hashmap(
    historical_sessions: &[Vec<u64>],
    timestamps: &[u32],
    m_most_recent_sessions: usize,
    idf_weighting: f64,
) -> (
    HashMap<u64, Vec<u32>>,
    HashMap<u64, f64>,
    HashMap<u32, Vec<u64>>,
) {
    /***
    Returns
    item_to_top_sessions_ordered: HashMap<u64, Vec<u32>>
    item_to_idf_score: HashMap<u64, f64>
    session_to_items_sorted: HashMap<u32, Vec<u64>>,
    */

    // Initialize arrays
    let max_capacity: usize = historical_sessions.iter().map(|x| x.len()).sum();
    let mut historical_sessions_values = Vec::with_capacity(max_capacity);
    let mut historical_sessions_session_indices = Vec::with_capacity(max_capacity);
    let mut historical_sessions_indices = Vec::with_capacity(max_capacity);
    let mut historical_sessions_timestamps = Vec::with_capacity(max_capacity);
    let mut iterable = 0_usize;
    let mut session_to_items_sorted = HashMap::with_capacity(historical_sessions.len());

    // Create (i) vector of historical sessions, (ii) vector of historical session indices, (iii) vector of session indices
    for (session_id, session) in historical_sessions.iter().enumerate() {
          for (item_id, _) in session.iter().enumerate() {
              historical_sessions_values.push(historical_sessions[session_id][item_id]);
              historical_sessions_indices.push(iterable);
              historical_sessions_session_indices.push(session_id);
              historical_sessions_timestamps.push(timestamps[session_id]);
              iterable += 1;
          }
          let session_items = historical_sessions[session_id].clone();
          session_to_items_sorted.insert(session_id as u32, session_items);
    }

    // Sort historical session values and session indices array
    historical_sessions_indices.sort_by_key(|&i| historical_sessions_values[i]);
    let historical_sessions_values_sorted: Vec<u64> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_values[i] as u64)
        .collect();
    let historical_sessions_session_indices_sorted: Vec<u32> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_session_indices[i] as u32)
        .collect();
    let historical_sessions_timestamps_sorted: Vec<u64> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_timestamps[i] as u64)
        .collect();

    // Get unique item_ids and create hashmap
    let mut unique_items = historical_sessions_values_sorted.clone();
    unique_items.dedup();
    let mut item_to_top_sessions_ordered = HashMap::with_capacity(unique_items.len());
    let mut item_to_idf_score = HashMap::with_capacity(unique_items.len());
    // Loop over unique items to remove all sessions per item older than n_most_recent_sessions and fill hashmap with n_most_recent_similar_sessions per item.
    for current_item in unique_items.iter() {
        let left_index =
            binary_search_left(&historical_sessions_values_sorted, *current_item).unwrap();
        let right_index =
            binary_search_right(&historical_sessions_values_sorted, *current_item).unwrap();
        let current_item_timestamps: Vec<u64> =
            historical_sessions_timestamps_sorted[left_index..right_index + 1].to_vec();
        let current_item_similar_sessions_ids: Vec<u32> =
            historical_sessions_session_indices_sorted[left_index..right_index + 1].to_vec();
        // Sort session ids by reverse timestamp and truncate to n_most_recent_sessions
        let mut timestamp_indices: Vec<usize> = (0..current_item_timestamps.len()).collect();
        timestamp_indices.sort_by_key(|&i| current_item_timestamps[i]);
        let mut current_item_similar_sessions_id_sorted: Vec<u32> = timestamp_indices
            .iter()
            .map(|&i| current_item_similar_sessions_ids[i] as u32)
            .collect();
        current_item_similar_sessions_id_sorted.reverse();
        current_item_similar_sessions_id_sorted.truncate(m_most_recent_sessions);
        // Store (item, similar_sessions) in hashmap
        item_to_top_sessions_ordered.insert(*current_item, current_item_similar_sessions_id_sorted);
        // Store (item, idf score) in second hashmap
        let idf_score =
            (historical_sessions_values_sorted.len() as f64
                / current_item_timestamps.len() as f64)
                .ln() * idf_weighting;
        item_to_idf_score.insert(*current_item, idf_score);
    }

    // Return hashmap(keys, values): (item_id, Vec[session_ids])
    (
        item_to_top_sessions_ordered,
        item_to_idf_score,
        session_to_items_sorted,
    )
}


// Custom binary search because this is stable unlike the rust default (i.e. this always returns right-most index in case of duplicate entries instead of a random match)
fn binary_search_right(array: &[u64], key: u64) -> Result<usize, usize> {
    let mut top: usize = array.len();
    let mut mid: usize;
    let mut bottom: usize = 0;

    if top == 0 {
        return Err(0);
    }

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if unsafe { array.get_unchecked(mid) } > &key {
            top = mid;
        } else {
            bottom = mid + 1;
        }
    }

    if top > 0 {
        if array[top - 1] == key {
            Ok(top - 1)
        } else {
            Err(top - 1)
        }
    } else {
        Err(top)
    }
}

fn binary_search_left(array: &[u64], key: u64) -> Result<usize, usize> {
    let mut top: usize = array.len();
    let mut mid: usize;
    let mut bottom: usize = 0;

    if top == 0 {
        return Err(0);
    }

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if unsafe { array.get_unchecked(mid) } < &key {
            bottom = mid + 1;
        } else {
            top = mid;
        }
    }

    if top < array.len() {
        if array[top] == key {
            Ok(top)
        } else {
            Err(top)
        }
    } else {
        Err(top)
    }
}


pub fn read_from_file(
    path: &str,
) -> Result<(Vec<Vec<u64>>, Vec<Vec<usize>>, Vec<u32>, Vec<usize>), Box<dyn Error>> {
    // Creates a new csv `Reader` from a file
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(path)?;

    // Vector initialization
    let mut session_id: Vec<usize> = Vec::with_capacity(100_000_000);
    let mut item_id: Vec<usize> = Vec::with_capacity(100_000_000);
    let mut time: Vec<usize> = Vec::with_capacity(100_000_000);

    reader.deserialize().for_each(|result| {
        if result.is_ok() {
            let raw: (usize, usize, f64) = result.unwrap();
            let (a_session_id, a_item_id, a_time): (usize, usize, usize) =
                (raw.0, raw.1, raw.2.round() as usize);

            session_id.push(a_session_id);
            item_id.push(a_item_id);
            time.push(a_time);
        } else {
            eprintln!("Unable to parse input!");
        }
    });

    // Sort by session id - the data is unsorted
    let mut session_id_indices: Vec<usize> = (0..session_id.len()).into_iter().collect();
    session_id_indices.sort_by_key(|&i| &session_id[i]);
    let session_id_sorted: Vec<usize> = session_id_indices
        .iter()
        .map(|&i| session_id[i])
        .collect();
    let item_id_sorted: Vec<usize> = session_id_indices
        .iter()
        .map(|&i| item_id[i])
        .collect();
    let time_sorted: Vec<usize> = session_id_indices
        .iter()
        .map(|&i| time[i])
        .collect();

    // Get unique session ids
    session_id.sort_unstable();
    session_id.dedup();

    // Get unique item ids
    // let mut unique_item_ids = item_id.clone();
    item_id.sort_unstable();
    item_id.dedup();

    // Create historical sessions array (deduplicated), historical sessions id array and array with max timestamps.
    //let mut i: usize = 0;
    let mut historical_sessions: Vec<Vec<u64>> = Vec::with_capacity(session_id.len());
    let mut historical_sessions_id: Vec<Vec<usize>> = Vec::with_capacity(session_id.len());
    let mut historical_sessions_max_time_stamp: Vec<u32> =
        Vec::with_capacity(session_id.len());
    let mut history_session: Vec<u64> = Vec::with_capacity(1000);
    let mut history_session_id: Vec<usize> = Vec::with_capacity(1000);
    let mut max_time_stamp: usize = time_sorted[0];
    // Push initial session and item id
    history_session.push(item_id_sorted[0] as u64);
    //BUG history_session_id.push(item_id_sorted[0]);
    history_session_id.push(session_id_sorted[0]);
    // Loop over length of data
    for i in 1..session_id_sorted.len() {
        if session_id_sorted[i] == session_id_sorted[i - 1] {
            if !history_session.contains(&(item_id_sorted[i] as u64)) {
                history_session.push(item_id_sorted[i] as u64);
                history_session_id.push(session_id_sorted[i]);
                if time_sorted[i] > max_time_stamp {
                    max_time_stamp = time_sorted[i];
                }
            }
        } else {
            let mut history_session_sorted = history_session.clone();
            history_session_sorted.sort_unstable();
            historical_sessions.push(history_session_sorted);
            historical_sessions_id.push(history_session_id.clone());
            historical_sessions_max_time_stamp.push(max_time_stamp as u32);
            history_session.clear();
            history_session_id.clear();
            history_session.push(item_id_sorted[i] as u64);
            history_session_id.push(session_id_sorted[i]);
            max_time_stamp = time_sorted[i];
        }
    }

    Ok((
        historical_sessions,
        historical_sessions_id,
        historical_sessions_max_time_stamp,
        session_id_sorted
    ))
}
