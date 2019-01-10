extern crate pretty_env_logger;
extern crate spdyboost;

use std::env;
use std::time::Instant;

use spdyboost::config::{Config, Learner};
use spdyboost::dataset::Dataset;
use spdyboost::tree::baseline_tree_learner::TreeLearner as BaselineLearner;
use spdyboost::tree::bit_tree_learner::TreeLearner as BitTreeLearner;
use spdyboost::tree::loss::{L2Loss, LossFunGrad};
use spdyboost::bits::BitSliceLayout4;

type TheBitTreeLearner<'a> = BitTreeLearner<'a, BitSliceLayout4>;

pub fn main() -> Result<(), String> {
    pretty_env_logger::init();

    let mut config = Config::new();
    config.target_feature_id = -1;
    config.categorical_columns = (0..256).collect();
    config.max_tree_depth = 15;
    config.min_sum_hessian = 1.0;
    config.discr_lo = -1.0;
    config.discr_hi = 1.0;
    //config.learner = Learner::Baseline;
    config.learner = Learner::BitLearner;

    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).ok_or("no data file given")?;

    let dataset = Dataset::from_csv_file(&config, filename)?;
    let target = dataset.target().get_raw_data();
    let loss = L2Loss::new();
    let gradients = target.iter().map(|&v| loss.eval_grad(v, 0.0)).collect();

    let mut learner = TheBitTreeLearner::new(&config, &dataset, gradients);
    //let mut learner = BaselineLearner::new(&config, &dataset, gradients);

    let r = 1;
    let now = Instant::now();
    for _ in 0..r { learner.reset(); learner.train(); }
    let elapsed = now.elapsed();
    println!("TRAINED IN {} ms", (elapsed.as_secs() as f32 * 1e3 +
             elapsed.subsec_micros() as f32 * 1e-3) / r as f32);


    let tree = learner.into_tree();
    let eval = tree.evaluate(&dataset, &L2Loss::new());

    println!("eval: {:e}", eval);
    println!("{:?}", tree);

    Ok(())
}
