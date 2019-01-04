extern crate pretty_env_logger;
extern crate spdyboost;

use std::env;
use std::time::Instant;

use spdyboost::config::Config;
use spdyboost::dataset::Dataset;
use spdyboost::tree::baseline::TreeLearner;
use spdyboost::tree::loss::{L2Loss, LossFunGrad};

pub fn main() -> Result<(), String> {
    pretty_env_logger::init();

    let mut config = Config::new();
    config.target_feature_id = -1;
    config.categorical_columns = (0..256).collect();
    config.max_tree_depth = 5;

    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).ok_or("no data file given")?;

    let dataset = Dataset::from_csv_file(&config, filename)?;
    let target = dataset.target().get_raw_data();
    let loss = L2Loss::new();
    let gradients = target.iter().map(|&v| loss.eval_grad(v, 0.0)).collect();

    let mut learner = TreeLearner::new(&config, &dataset, gradients);

    let r = 1;
    let now = Instant::now();
    for _ in 0..r { learner.reset(); learner.train(); }
    let elapsed = now.elapsed();
    println!("TRAINED IN {} ms", (elapsed.as_secs() as f32 * 1e3 +
             elapsed.subsec_micros() as f32 * 1e-3) / r as f32);

    Ok(())
}
