extern crate spdyboost;
extern crate pretty_env_logger;
extern crate log;

use std::time::Instant;

use spdyboost::dataset::{DataSetBuilder, FeatureData};
use spdyboost::conf::Config;
use spdyboost::tree::{TreeLearner};
use spdyboost::tree::loss;
use spdyboost::tree::eval::FirstOrderSplitEvaluator;

pub fn main() -> Result<(), String> {
    pretty_env_logger::init();

    let mut conf = Config::default();
    conf.lowcard_nominal_features = (0..4).collect();
    conf.target_feature = 4;
    conf.max_tree_depth = 5;

    let dataset = DataSetBuilder::from_csv_file(&conf, "/tmp/data1000000.csv")?;
    //let dataset = DataSetBuilder::from_gzip_csv_file(&conf, "/tmp/data1000.csv.gz")?;
    let nexamples = dataset.nexamples();

    let prev_predictions = (0..nexamples).map(|_| 0.0);
    let target_values = if let FeatureData::Numerical(vec) = dataset.get_target_feature().get_data() {
        vec.iter().cloned()
    } else { panic!() };

    let leaf_eval = FirstOrderSplitEvaluator::new(&conf, loss::L2Loss::new());
    let mut learner = TreeLearner::new(&conf, &dataset, &leaf_eval, target_values, prev_predictions);

    let start = Instant::now();
    learner.train();
    println!("trained in {}sec {}ms", start.elapsed().as_secs(), start.elapsed().subsec_micros() as f32/1000.0);

    Ok(())
}
