use std::env;
use std::time::Instant;

use spdyboost::config::Config;
use spdyboost::NumT;
use spdyboost::dataset::Dataset;
use spdyboost::tree_learner::{TreeLearner, TreeLearnerContext};
use spdyboost::objective::Objective;
use spdyboost::objective;
use spdyboost::metric::Metric;
use spdyboost::metric;

pub fn main() {
    pretty_env_logger::init();

    let mut config = Config::new();
    config.target_feature_id = -1;
    config.categorical_columns = (0..16).collect();
    config.optimize_leaf_values = true;
    config.max_tree_depth = 6;
    config.discr_nbits = 1;
    config.compression_threshold = 0.50;
    //config.compression_threshold = 1.0;
    config.min_gain = 1e-6;

    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).expect("no data file given");

    let dataset = Dataset::from_csv_file(&config, filename).expect("data error");
    let targets = dataset.target().get_raw_data();
    let predictions = vec![0.0; targets.len()];

    // Timings
    let r = 10;
    let mut context = TreeLearnerContext::new(&config, &dataset);
    let mut obj = objective::L1::new(); obj.initialize(&targets, &predictions);
    let mut objective: Box<dyn Objective> = Box::new(obj);
    let now = Instant::now();
    for _ in 0..r {
        let learner = TreeLearner::new(&mut context, objective.as_mut());
        let _tree = learner.train();
    }
    let elapsed = now.elapsed();
    println!("TRAINED IN {} ms", (elapsed.as_secs() as f32 * 1e3 +
             elapsed.subsec_micros() as f32 * 1e-3) / r as f32);

    // Results
    let learner = TreeLearner::new(&mut context, objective.as_mut());
    let tree = learner.train();

    let pred = tree.predict(&dataset);
    let ms: [Box<dyn Metric>; 4] = [Box::new(metric::L2::new()),
                                    Box::new(metric::Rmse::new()),
                                    Box::new(metric::BinaryLoss::new()),
                                    Box::new(metric::BinaryError::new())];
    println!("objective: {}, discr_nbits: {}", objective.name(), config.discr_nbits);
    for m in &ms {
        let eval = m.eval(&targets, &pred);
        println!("eval: {:e} ({})", eval, m.name());
    }

    //println!();
    //for (i, (x, y)) in dataset.target().get_raw_data().iter().zip(pred).enumerate() {
    //    println!("{:4}: {:15} {:15} {:15e}", i, x, y, (x-y).abs());
    //}

    println!("{:?}", tree);

    //println!("writing results...");
    //write_results(&pred).expect("writing failed");
    //println!("done");
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
