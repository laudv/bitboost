use std::env;
use std::time::Instant;

use spdyboost::config::Config;
use spdyboost::NumT;
use spdyboost::data::{Data, Dataset};
use spdyboost::tree_learner::{TreeLearner, TreeLearnerContext};
use spdyboost::objective::Objective;
use spdyboost::objective;
use spdyboost::metric::Metric;
use spdyboost::metric;

pub fn main() {
    pretty_env_logger::init();

    let mut config = Config::new();
    config.target_feature = -1;
    config.learning_rate = 1.0;
    config.categorical_features = (0..4).collect();
    config.max_tree_depth = 6;
    config.discr_nbits = 4;
    config.compression_threshold = 0.50;
    config.min_gain = 1e-6;
    config.max_nbins = 4;

    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).expect("no data file given");

    //let data = Dataset::from_csv_file(&config, filename).expect("data error");
    let data = Data::from_csv_path(&config, filename).expect("data error");
    let target = data.get_target();
    let dataset = Dataset::construct_from_data(&config, &data, &target);

    // Timings
    let r = 0;
    let mut context = TreeLearnerContext::new(&config, &dataset);

    let mut obj = objective::L2::new();
    obj.initialize(&config, &target);
    obj.update(&target);

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
    objective.initialize(&config, target);
    objective.update(target);
    let learner = TreeLearner::new(&mut context, objective.as_mut());
    let mut tree = learner.train();
    tree.set_bias(objective.bias());

    let pred = tree.predict(&dataset);
    let ms: [Box<dyn Metric>; 4] = [Box::new(metric::L2::new()),
                                    Box::new(metric::Rmse::new()),
                                    Box::new(metric::BinaryLoss::new()),
                                    Box::new(metric::BinaryError::new())];
    println!("objective: {}, discr_nbits: {}", objective.name(), config.discr_nbits);
    for m in &ms {
        let eval = m.eval(&target, &pred);
        println!("eval: {:e} ({})", eval, m.name());
    }

    println!();
    println!("{:4}  {:>15} {:>15} {:>15}", "", "target", "pred", "diff");
    for (i, (x, y)) in target.iter().zip(&pred).enumerate() {
        println!("{:4}: {:15} {:15} {:15e}", i, x, y, (x-y).abs());
    }

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
