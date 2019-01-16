use crate::NumT;

pub enum Learner {
    Baseline,
    BitLearner,
}

pub struct Config {

    // Dataset
    pub target_feature_id: isize,

    pub csv_has_header: bool,
    pub csv_separator: char,

    pub categorical_columns: Vec<usize>,

    // Tree
    pub learner: Learner,
    pub max_tree_depth: usize,
    pub reg_lambda: NumT,
    pub min_sum_hessian: NumT,
    pub bagging_fraction: NumT,
    pub min_gain: NumT,

    pub discr_nbits: usize,
    pub discr_bounds: (NumT, NumT),

    /// A threshold for the ratio #zero-block / #blocks; if ratio > threshold, then apply
    /// compression. Disable by setting to 1.0 or higher.
    pub compression_threshold: NumT,
}

impl Config {
    pub fn new() -> Config {
        Config {
            target_feature_id: -1,

            csv_has_header: true,
            csv_separator: ',',

            categorical_columns: Vec::new(),

            learner: Learner::BitLearner,
            max_tree_depth: 4,
            reg_lambda: 0.0,
            min_sum_hessian: 100.0,
            bagging_fraction: 1.0,
            min_gain: 1e-3,

            discr_nbits: 4,
            discr_bounds: (-1.0, 1.0),

            compression_threshold: 0.75,
        }
    }
}
