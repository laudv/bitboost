use std::env;

use spdyboost::NumT;
use spdyboost::config::Config;
use spdyboost::dataset::Dataset;
use spdyboost::boost::Booster;
use spdyboost::tree::loss;
use spdyboost::tree::eval::Evaluator;

pub fn main() {
    pretty_env_logger::init();

    let mut config = Config::new();
    config.target_feature_id = -1;
    config.categorical_columns = (0..256).collect();
    config.reg_lambda = 0.05;
    config.max_tree_depth = 6;
    config.discr_nbits = 1;
    config.compression_threshold = 0.50;

    config.learning_rate = 0.05;
    config.niterations = 200;
    config.optimize_leaf_values = false;

    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).expect("no data file given");
    let dataset = Dataset::from_csv_file(&config, filename).expect("data error");

    let mut booster = Booster::new(&config, &dataset);
    booster.train();

    let model = booster.into_model();

    if let Some(testfile) = args.get(2) {
        println!("Loading test set");
        let testset = Dataset::from_csv_file(&config, testfile).expect("test data error");
        let targets = testset.target().get_raw_data();
        let pred = model.predict(&testset);

        evaluate(&loss::L2Loss::new(), targets, &pred);
        //evaluate(&loss::LogLoss::new(), targets, &pred);

        write_results(&pred).unwrap();
    }
}

use std::fs::File;
use std::io::{Write, Error};

fn write_results(res: &[NumT]) -> Result<(), Error> {
    let mut file = File::create("/tmp/spdyboost_predictions.txt")?;
    for &r in res {
        writeln!(file, "{}", r)?;
    }
    Ok(())
}

fn evaluate<E: Evaluator>(e: &E, targets: &[NumT], predictions: &[NumT]) {
    let eval = e.evaluate(targets.iter().cloned(), predictions.iter().cloned());
    println!("performance: {} {} (sqrt {})", e.evaluator_name(), eval, eval.sqrt());
}
