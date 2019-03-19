use std::env;

use spdyboost::NumT;
use spdyboost::config::Config;
use spdyboost::data::{Data, Dataset};
use spdyboost::objective::{Objective, objective_from_name};
use spdyboost::tree_learner::{TreeLearner, TreeLearnerContext};
use spdyboost::metric::{Metric, metrics_from_names};
use spdyboost::boost::Booster;

pub fn main() {
    pretty_env_logger::init();

    let args: Vec<String> = env::args().collect();
    let res = match args.get(1).map(|x| x.as_str()) {
        Some("tree")  => single_tree(&args[2..]),
        Some("boost") => boost(&args[2..]),
        _             => boost(&args[1..])
    };

    match res {
        Ok(_)    => println!("Done"),
        Err(msg) => println!("Failure: {}", msg),
    }
}

fn single_tree(args: &[String]) -> Result<(), String> {
    let config = Config::parse(args.iter().map(|x| x.as_str()))?;
    let data = Data::from_csv_path(&config, config.train.as_str())?;
    let target = data.get_target();

    let ms = metrics_from_names(&config.metrics).ok_or("unknown metric".to_string())?;
    let mut objective = objective_from_name(&config.objective, &config)
        .ok_or(format!("unknown objective '{}'", config.objective))?;
    objective.initialize(&config, &target);
    objective.update(&target);

    let mut dataset = Dataset::new(&data);
    dataset.update(&config, objective.gradients(), objective.bounds());

    let mut context = TreeLearnerContext::new(&config, &data);
    let learner = TreeLearner::new(&mut context, &dataset, objective.as_mut());
    let mut tree = learner.train();
    tree.set_bias(objective.bias());

    let pred = tree.predict(&data);
    print_summary(&config, target, &pred, objective.as_ref(), &ms);

    Ok(())
}

fn boost(args: &[String]) -> Result<(), String> {
    let config = Config::parse(args.iter().map(|x| x.as_str()))?;
    let data = Data::from_csv_path(&config, config.train.as_str())?;
    let target = data.get_target();

    let ms = metrics_from_names(&config.metrics).ok_or("unknown metric".to_string())?;
    let mut objective = objective_from_name(&config.objective, &config)
        .ok_or(format!("unknown objective '{}'", config.objective))?;
    let booster = Booster::new(&config, &data, objective.as_mut(), &ms);
    let model = booster.train();

    let pred = model.predict(&data);
    print_summary(&config, target, &pred, objective.as_ref(), &ms);

    Ok(())
}

fn print_summary(config: &Config, target: &[NumT], pred: &[NumT], objective: &dyn Objective,
                 ms: &[Box<dyn Metric>]) {
    println!();
    println!("{:4}  {:>15} {:>15} {:>15}", "", "target", "pred", "diff");
    for (i, (x, y)) in target.iter().zip(pred).enumerate().take(200) {
        println!("{:4}: {:15} {:15} {:15e}", i, x, y, (x-y).abs());
    }

    println!();
    println!("=== Objective:   {}", objective.name());
    println!("=== Discr. bits: {}", config.discr_nbits);
    for m in ms {
        let eval = m.eval(&target, &pred);
        println!(" |   Eval {:13} {:e}", m.name(), eval);
    }
    println!("===");
}
