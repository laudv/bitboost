use crate::NumT;

pub struct Config {

    // Dataset
    pub target_feature_id: isize,

    pub csv_has_header: bool,
    pub csv_separator: char,

    pub categorical_columns: Vec<usize>,

    pub objective: String,

    // Tree
    pub max_tree_depth: usize,
    pub reg_lambda: NumT,
    pub min_examples_leaf: u32,
    pub min_gain: NumT,
    pub nsplits_cands: usize,
    pub huber_alpha: NumT,
    
    pub discr_nbits: usize,

    /// A threshold for the ratio #zero-block / #blocks; if ratio > threshold, then apply
    /// compression. Disable by setting to 1.0 or higher.
    pub compression_threshold: NumT,


    // Boosting
    pub learning_rate: NumT,
    pub niterations: usize,
}

impl Config {
    pub fn new() -> Config {
        Config {
            target_feature_id: -1,

            csv_has_header: true,
            csv_separator: ',',

            categorical_columns: Vec::new(),

            objective: String::from("L2"),

            max_tree_depth: 4,
            reg_lambda: 0.0,
            min_examples_leaf: 1,
            min_gain: 0.0,
            nsplits_cands: 16,
            huber_alpha: 0.9,

            discr_nbits: 4,

            compression_threshold: 0.75,

            learning_rate: 0.1,
            niterations: 100,
        }
    }
}
