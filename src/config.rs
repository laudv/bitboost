
pub struct Config {
    pub target_feature_id: isize,

    pub csv_has_header: bool,
    pub csv_separator: char,

    pub categorical_columns: Vec<usize>,
}

impl Config {
    pub fn new() -> Config {
        Config {
            target_feature_id: -1,

            csv_has_header: true,
            csv_separator: ',',

            categorical_columns: Vec::new(),
        }
    }
}
