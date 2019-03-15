use std::env;

use spdyboost::NumT;
//use spdyboost::config::Config;
//use spdyboost::dataset::Dataset;
//use spdyboost::boost::Booster;
//use spdyboost::metric::Metric;
//use spdyboost::metric;

pub fn main() {
    pretty_env_logger::init();

//    let mut config = Config::new();
//    config.target_feature = -1;
//    config.categorical_columns = (0..800).collect();
//    //config.categorical_columns = vec![0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,107,110,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130];
//    config.min_examples_leaf = 1;
//    config.reg_lambda = 0.0;
//    config.max_tree_depth = 6;
//    config.discr_nbits = 4;
//    config.compression_threshold = 0.5;
//    config.min_gain = 1.0;
//
//    config.learning_rate = 0.1;
//    config.niterations = 200;
//    config.objective = String::from("Binary");
//
//    let args: Vec<String> = env::args().collect();
//    let filename = args.get(1).expect("no data file given");
//    let dataset = Dataset::from_csv_file(&config, filename).expect("data error");
//    let ms: [Box<dyn Metric>; 4] = [Box::new(metric::L2::new()),
//                                    Box::new(metric::Rmse::new()),
//                                    Box::new(metric::BinaryLoss::new()),
//                                    Box::new(metric::BinaryError::new())];
//
//    //let booster = Booster::new(&config, &dataset, &ms);
//    let booster = Booster::new(&config, &dataset, &[]);
//    let model = booster.train();
//
//    let targets = dataset.target().get_raw_data();
//    let pred = model.predict(&dataset);
//
//    for m in &ms {
//        let eval = m.eval(targets, &pred);
//        println!("train eval: {:e} ({})", eval, m.name());
//    }
//
//    //if dataset.nexamples() <= 200 {
//        println!();
//        println!("{:4}  {:>15} {:>15} {:>15}", "", "target", "pred", "diff");
//        for (i, (x, y)) in targets.iter().zip(pred).enumerate().take(100) {
//            println!("{:4}: {:15} {:15} {:15e}", i, x, y, (x-y).abs());
//        }
//    //}
//
//    if let Some(testfile) = args.get(2) {
//        println!("Loading test set");
//        let testset = Dataset::from_csv_file(&config, testfile).expect("test data error");
//        let targets = testset.target().get_raw_data();
//        let pred = model.predict(&testset);
//
//        for m in &ms {
//            let eval = m.eval(targets, &pred);
//            println!(" test eval: {:e} ({})", eval, m.name());
//        }
//
//        write_results(&pred).unwrap();
//    }
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
