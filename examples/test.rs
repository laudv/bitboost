extern crate pretty_env_logger;
extern crate spdyboost;

use std::env;

use spdyboost::config::Config;
use spdyboost::dataset::Dataset;

pub fn main() -> Result<(), String> {
    pretty_env_logger::init();

    let mut config = Config::new();
    config.target_feature_id = -1;

    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).ok_or("no data file given")?;

    let dataset = Dataset::from_csv_file(&config, filename)?;

    Ok(())
}
