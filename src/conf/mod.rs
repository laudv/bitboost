use std::default::Default;

pub struct Config {
    pub max_lowcard_nominal_cardinality: usize
}

impl Default for Config {
    fn default() -> Config {
        Config {
            max_lowcard_nominal_cardinality: 16
        }
    }
}
