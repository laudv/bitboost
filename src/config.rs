use NumT;

pub struct Config {

    // Dataset
    pub target_feature_id: isize,

    pub csv_has_header: bool,
    pub csv_separator: char,

    pub categorical_columns: Vec<usize>,

    // Tree
    pub max_tree_depth: usize,
    pub reg_lambda: NumT,
    pub min_sum_hessian: NumT,
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
            min_sum_hessian: 100.0,
        }
    }
}
