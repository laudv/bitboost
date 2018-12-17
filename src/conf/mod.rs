use std::default::Default;

use dataset::NumericalType;

#[derive(PartialEq, Eq)]
pub enum Objective {
    Regression,
    Classification,
}

//#[derive(Deserialize)]
pub struct Config {
    pub objective: Objective,
    pub max_lowcard_nominal_cardinality: usize,
    pub max_tree_depth: usize,
    pub lowcard_nominal_features: Vec<usize>,
    pub ignored_features: Vec<usize>,
    pub target_feature: usize,

    pub target_values_nbits: u8,
    pub target_values_limits: (NumericalType, NumericalType),

    pub reg_lambda: NumericalType,

    //#[serde(flatten)]
    //extra: HashMap<String, Value>,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            objective: Objective::Regression,
            max_lowcard_nominal_cardinality: 64,
            max_tree_depth: 5,
            lowcard_nominal_features: Vec::new(),
            ignored_features: Vec::new(),
            target_feature: 0,

            target_values_nbits: 4,
            target_values_limits: (-1.0, 1.0),

            reg_lambda: 1.0,
        }
    }
}

// TODO impove error messages
//macro_rules! try_or_str {
//    ($res:expr, $msg:expr) => {{
//        match $res {
//            Ok(a) => a,
//            Err(_) => return Err(String::from($msg)),
//        }
//    }}
//}

impl Config {

    //pub fn from_file(filename: &str) -> Result<Config, String> {
    //    let s = try_or_str!(fs::read_to_string(filename), "cannot read config");
    //    Self::from_str(&s)
    //}

    //pub fn from_str(string: &str) -> Result<Config, String> {
    //    let t = try_or_str!(toml::from_str(string), "cannot parse config");
    //    Ok(t)
    //}

    //pub fn replace_values(&mut self, replace: Config) {
    // TODO generate automatically using macro
    //}
}


#[cfg(test)]
mod test {
    use conf::Config;

    #[test]
    pub fn test_config() {
    }

}
