use std::env;
use std::time::Instant;

use spdyboost::config::{Config, Learner};
use spdyboost::NumT;
use spdyboost::dataset::Dataset;
use spdyboost::tree::baseline_tree_learner::TreeLearner as BaselineLearner;
use spdyboost::tree::bit_tree_learner::TreeLearner as BitTreeLearner;
use spdyboost::tree::loss::{L2Loss, LossFunGrad};
use spdyboost::tree::eval::Evaluator;

pub fn main() {
    pretty_env_logger::init();

    let mut config = Config::new();
    config.target_feature_id = -1;
    config.categorical_columns = (0..256).collect();
    config.max_tree_depth = 6;
    config.discr_nbits = 1;
    config.compression_threshold = 0.75;
    //config.compression_threshold = 1.0;
    //config.learner = Learner::Baseline;
    config.learner = Learner::BitLearner;

    let grad_bounds = (-1.0, 1.0);

    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).expect("no data file given");

    let dataset = Dataset::from_csv_file(&config, filename).expect("data error");
    let target = dataset.target().get_raw_data();
    let loss = L2Loss::new();
    let gradients: Vec<NumT> = target.iter().map(|&v| loss.eval_grad(v, 0.0)).collect();

    // Timings
    let r = 20;
    let now = Instant::now();
    for _ in 0..r {
        let mut learner = BitTreeLearner::new(&config, &dataset, &gradients, grad_bounds);
        learner.train();
    }
    let elapsed = now.elapsed();
    println!("TRAINED IN {} ms", (elapsed.as_secs() as f32 * 1e3 +
             elapsed.subsec_micros() as f32 * 1e-3) / r as f32);

    // Results
    let mut learner = BitTreeLearner::new(&config, &dataset, &gradients, grad_bounds);
    learner.train();

    let tree = learner.into_tree();
    let pred = tree.predict(&dataset);
    let eval = L2Loss::new().eval(dataset.target().get_raw_data().iter().cloned(),
                                  pred.iter().cloned());

    println!("eval: {:e}", eval);
    println!("{:?}", tree);

    println!("writing results...");
    write_results(&pred).expect("writing failed");
    println!("done");
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
