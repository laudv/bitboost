use std::env;
use std::time::Instant;

use spdyboost::config::Config;
use spdyboost::NumT;
use spdyboost::dataset::Dataset;
use spdyboost::tree_learner::{TreeLearner, TreeLearnerContext};
use spdyboost::objective::{L2, Objective};

pub fn main() {
    pretty_env_logger::init();

    let mut config = Config::new();
    config.target_feature_id = -1;
    config.categorical_columns = (0..256).collect();
    config.max_tree_depth = 6;
    config.discr_nbits = 1;
    config.compression_threshold = 0.75;
    //config.compression_threshold = 1.0;

    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).expect("no data file given");

    let dataset = Dataset::from_csv_file(&config, filename).expect("data error");
    let targets = dataset.target().get_raw_data();
    let predictions = vec![0.0; targets.len()];

    // Timings
    let r = 20;
    let mut context = TreeLearnerContext::new(&config, &dataset);
    let objective: Box<dyn Objective> = Box::new(L2::new(&targets, &predictions));
    let now = Instant::now();
    for _ in 0..r {
        let learner = TreeLearner::new(&mut context, objective.as_ref());
        let _tree = learner.train();
    }
    let elapsed = now.elapsed();
    println!("TRAINED IN {} ms", (elapsed.as_secs() as f32 * 1e3 +
             elapsed.subsec_micros() as f32 * 1e-3) / r as f32);

    // Results
    //let mut learner = TreeLearner::new(&config, &dataset, &gradients, grad_bounds,
    //                                   &mut context);
    //learner.train();

    //let tree = learner.into_tree();
    //let pred = tree.predict(&dataset);
    //let eval = L2Loss::new().evaluate(dataset.target().get_raw_data().iter().cloned(),
    //                                  pred.iter().cloned());

    //println!("eval: {:e}", eval);
    //println!("{:?}", tree);

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
