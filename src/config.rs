use crate::NumT;

pub struct Config {

    // Dataset
    pub target_feature_id: isize,

    pub csv_has_header: bool,
    pub csv_separator: char,

    pub categorical_columns: Vec<usize>,

    // Tree
    pub max_tree_depth: usize,
    pub reg_lambda: NumT,
    pub min_examples_leaf: u32,
    pub min_gain: NumT,
    pub nbuckets: usize,

    pub discr_nbits: usize,

    /// A threshold for the ratio #zero-block / #blocks; if ratio > threshold, then apply
    /// compression. Disable by setting to 1.0 or higher.
    pub compression_threshold: NumT,


    // Boosting
    pub learning_rate: NumT,
    pub niterations: usize,
    pub optimize_leaf_values: bool,
}

impl Config {
    pub fn new() -> Config {
        Config {
            target_feature_id: -1,

            csv_has_header: true,
            csv_separator: ',',

            categorical_columns: Vec::new(),

            max_tree_depth: 4,
            reg_lambda: 0.0,
            min_examples_leaf: 1,
            min_gain: 0.0,
            nbuckets: 128,

            discr_nbits: 4,

            compression_threshold: 0.75,

            learning_rate: 0.1,
            niterations: 100,
            optimize_leaf_values: true,
        }
    }
}
