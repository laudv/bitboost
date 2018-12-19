extern crate spdyboost;
extern crate pretty_env_logger;
extern crate log;

use std::time::Instant;
use std::io;

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
    conf.max_tree_depth = 4;
    conf.target_values_nbits = 1;

    let dataset = DataSetBuilder::from_csv_file(&conf, "/tmp/data1000000.csv")?;
    //let dataset = DataSetBuilder::from_gzip_csv_file(&conf, "/tmp/data1000.csv.gz")?;
    let nexamples = dataset.nexamples();

    let prev_predictions = (0..nexamples).map(|_| 0.0);
    let target_values = if let FeatureData::Numerical(vec) = dataset.get_target_feature().get_data() {
        vec.iter().cloned()
    } else { panic!() };

    loop {
        println!("max_tree_depth=");
        let mut input_text = String::new();
        if io::stdin().read_line(&mut input_text).is_err() { break; }

        let max_tree_depth = if let Ok(max_tree_depth) = input_text.trim().parse::<usize>() {
            max_tree_depth
        } else { println!("invalid input"); break };

        conf.max_tree_depth = max_tree_depth;

        let r = 10;
        let nleaves = 1 << conf.max_tree_depth;

        let eval = FirstOrderSplitEvaluator::new(&conf, loss::L2Loss::new());
        let mut learner = TreeLearner::new(&conf, &dataset, &eval,
                                           target_values.clone(), prev_predictions.clone());

        let start = Instant::now();

        for _ in 0..r {
            learner.train();
        }

        let (s, ms) = (start.elapsed().as_secs(), start.elapsed().subsec_micros() as f32 * 1e-3);
        let time = (s as f32 * 1e3 + ms) / r as f32;

        println!("trained in {}ms ({} leaves)", time, nleaves);
    }

    Ok(())
}
