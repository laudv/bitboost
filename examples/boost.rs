use std::env;

use spdyboost::NumT;
use spdyboost::config::Config;
use spdyboost::dataset::Dataset;
use spdyboost::boost::Booster;
use spdyboost::metric::Metric;
use spdyboost::metric;

pub fn main() {
    pretty_env_logger::init();

    let mut config = Config::new();
    config.target_feature_id = -1;
    config.categorical_columns = (0..784).collect();
    config.min_examples_leaf = 1;
    config.reg_lambda = 0.0;
    config.max_tree_depth = 6;
    config.discr_nbits = 1;
    config.compression_threshold = 0.50;
    config.min_gain = 1e-4;

    config.learning_rate = 0.1;
    config.niterations = 200;
    config.objective = String::from("binary");

    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).expect("no data file given");
    let dataset = Dataset::from_csv_file(&config, filename).expect("data error");
    let ms: [Box<dyn Metric>; 4] = [Box::new(metric::L2::new()),
                                    Box::new(metric::Rmse::new()),
                                    Box::new(metric::BinaryLoss::new()),
                                    Box::new(metric::BinaryError::new())];

    //let booster = Booster::new(&config, &dataset, &ms);
    let booster = Booster::new(&config, &dataset, &[]);
    let model = booster.train();

    let targets = dataset.target().get_raw_data();
    let pred = model.predict(&dataset);

    for m in &ms {
        let eval = m.eval(targets, &pred);
        println!("train eval: {:e} ({})", eval, m.name());
    }

    //if dataset.nexamples() <= 200 {
        println!();
        println!("{:4}  {:>15} {:>15} {:>15}", "", "target", "pred", "diff");
        for (i, (x, y)) in targets.iter().zip(pred).enumerate().take(100) {
            println!("{:4}: {:15} {:15} {:15e}", i, x, y, (x-y).abs());
        }
    //}

    if let Some(testfile) = args.get(2) {
        println!("Loading test set");
        let testset = Dataset::from_csv_file(&config, testfile).expect("test data error");
        let targets = testset.target().get_raw_data();
        let pred = model.predict(&testset);

        for m in &ms {
            let eval = m.eval(targets, &pred);
            println!(" test eval: {:e} ({})", eval, m.name());
        }

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
