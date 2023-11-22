pub mod io;


use std::cmp::Ordering;


pub mod similarity_indexed;
pub mod vmis_index;

#[derive(PartialEq, Debug, Clone)]
pub struct SessionScore {
    pub id: u32,
    pub score: f64,
}

impl SessionScore {
    fn new(id: u32, score: f64) -> Self {
        SessionScore { id, score }
    }
}

impl Eq for SessionScore {}

impl Ord for SessionScore {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.score.partial_cmp(&other.score) {
            Some(Ordering::Less) => Ordering::Greater,
            Some(Ordering::Greater) => Ordering::Less,
            _ => self.id.cmp(&other.id)
        }
    }
}

impl PartialOrd for SessionScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


#[derive(Eq, Debug)]
pub struct SessionTime {
    pub session_id: u32,
    pub time: u32,
}

impl SessionTime {
    pub fn new(session_id: u32, time: u32) -> Self {
        SessionTime { session_id, time }
    }
}

impl Ord for SessionTime {
    fn cmp(&self, other: &Self) -> Ordering {
        // reverse order by time
        other.time.cmp(&self.time)
    }
}

impl PartialOrd for SessionTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SessionTime {
    fn eq(&self, other: &Self) -> bool {
        self.session_id == other.session_id
    }
}


pub fn linear_score(pos: usize) -> f64 {
    if pos < 10 {
        1.0 - (0.1 * pos as f64)
    } else {
        0.0
    }
}

