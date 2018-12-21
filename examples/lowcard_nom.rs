//extern crate spdyboost;
//extern crate pretty_env_logger;
//extern crate log;
////extern crate cpuprofiler;
//
//use std::time::{Instant, Duration};
//use std::env;
//use std::thread::sleep;
//
////use cpuprofiler::PROFILER;
//
//use spdyboost::dataset::{DataSetBuilder, FeatureData};
//use spdyboost::conf::Config;
//use spdyboost::tree::{TreeLearner};
//use spdyboost::tree::loss;
//use spdyboost::tree::eval::FirstOrderSplitEvaluator;
//
//pub fn main() -> Result<(), String> {
//    pretty_env_logger::init();
//
//    let mut conf = Config::default();
//    conf.lowcard_nominal_features = (0..257).collect();
//    conf.target_feature = -1;
//    conf.target_values_limits = (-1.0, 0.0);
//    conf.target_values_nbits = 4;
//    conf.reg_lambda = 0.0;
//
//    let args: Vec<String> = env::args().collect();
//    let filename = &args[1];
//
//    let dataset = DataSetBuilder::from_csv_file(&conf, filename)?;
//    let nexamples = dataset.nexamples();
//
//    let prev_predictions = (0..nexamples).map(|_| 0.0);
//    let target_values = if let FeatureData::Numerical(vec) = dataset.get_target_feature().get_data() {
//        vec.iter().cloned()
//    } else { panic!() };
//
//    //sleep(Duration::from_secs(2));
//
//    //PROFILER.lock().unwrap().start("/tmp/file.prof").unwrap();
//
//    //let repeat = 20;
//    //for depth in [2, 3, 4, 5, 6].iter().cloned() {
//    let repeat = 1;
//    for depth in [6].iter().cloned() {
//        conf.max_tree_depth = depth;
//        let nleaves = 1 << depth;
//
//        let eval = FirstOrderSplitEvaluator::new(&conf, loss::L2Loss::new());
//        let mut learner = TreeLearner::new(&conf, &dataset, &eval,
//                                           target_values.clone(), prev_predictions.clone());
//
//        let mut time = 0.0;
//        for _ in 0..repeat {
//            learner.reset();
//
//            let start = Instant::now();
//            learner.train();
//            let elapsed = start.elapsed();
//            time += elapsed.as_secs() as f32 * 1e3 + elapsed.subsec_micros() as f32 * 1e-3;
//        }
//        time /= repeat as f32;
//
//        println!("[spdyboost: {}, nleaves={:02}, nbits={}] {}", filename, nleaves,
//                 conf.target_values_nbits, time);
//    }
//
//    //PROFILER.lock().unwrap().stop().unwrap();
//
//    Ok(())
//}

pub fn main() {

}
