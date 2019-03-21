use std::env;
use std::fs::File;
use std::io::{Write, Error};

use crossbeam_utils::thread;

use spdyboost::NumT;
use spdyboost::config::Config;
use spdyboost::data::Data;
use spdyboost::dataset::Dataset;
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
        Ok(_)    => {},
        Err(msg) => println!("Failure: {}", msg),
    }
}

fn load_data(config: &Config) -> Result<(Data, Option<Data>), String> {
    let (train_data_res, test_data_res): (Result<Data, String>, Option<Result<Data, String>>) = thread::scope(|s| {
        let t1 = s.spawn(move |_| {
            let r = Data::from_csv_path(&config, config.train.as_str());
            println!("[   ] finished loading training data");
            r
        });
        let test_data_res = if !config.test.is_empty() {
            let r = Some(Data::from_csv_path(&config, config.test.as_str()));
            println!("[   ] finished loading test data");
            r
        } else { None };
        let train_data_res = t1.join().unwrap();

        (train_data_res, test_data_res)
    }).map_err(|_| "thread error".to_string())?;

    let train_data = train_data_res?;
    let test_data = match test_data_res {
        Some(res) => Some(res?),
        None => None
    };

    Ok((train_data, test_data))
}

fn single_tree(args: &[String]) -> Result<(), String> {
    let config = Config::parse(args.iter().map(|x| x.as_str()))?;
    let (train_data, test_data) = load_data(&config)?;
    let target = train_data.get_target();

    let ms = metrics_from_names(&config.metrics).ok_or("unknown metric".to_string())?;
    let mut objective = objective_from_name(&config.objective, &config)
        .ok_or(format!("unknown objective '{}'", config.objective))?;
    objective.initialize(&config, &target);
    objective.update(&target);

    let mut dataset = Dataset::new(&config, &train_data);
    dataset.update(&config, objective.gradients(), objective.bounds());

    let mut context = TreeLearnerContext::new(&config, &train_data);
    let learner = TreeLearner::new(&mut context, &dataset, objective.as_mut());
    let mut tree = learner.train();
    tree.set_bias(objective.bias());
    tree.set_supercats(dataset.extract_supercats());

    summary(&config, |d| tree.predict(d), &train_data, test_data.as_ref(),
            objective.as_ref(), &ms);

    Ok(())
}

fn boost(args: &[String]) -> Result<(), String> {
    let config = Config::parse(args.iter().map(|x| x.as_str()))?;
    let (train_data, test_data) = load_data(&config)?;

    let ms = metrics_from_names(&config.metrics).ok_or("unknown metric".to_string())?;
    let mut objective = objective_from_name(&config.objective, &config)
        .ok_or(format!("unknown objective '{}'", config.objective))?;
    let booster = Booster::new(&config, &train_data, objective.as_mut(), &ms);
    let model = booster.train();

    summary(&config, |d| model.predict(d), &train_data, test_data.as_ref(),
            objective.as_ref(), &ms);

    Ok(())
}

fn summary<F>(config: &Config, model: F, train: &Data, test: Option<&Data>,
              objective: &dyn Objective, ms: &[Box<dyn Metric>])
where F: Fn(&Data) -> Vec<NumT>
{
    let train_pred = model(train);
    let test_pred = test.map(|test| (test, model(test)));

    if config.prediction_len > 0 {
        println!();
        println!("[   ] train predictions (first {})", config.prediction_len);
        print_predictions(train.get_target(), &train_pred, config.prediction_len);

        if let Some((test_data, ref test_pred)) = test_pred {
            println!();
            println!("[   ] test predictions (first {})", config.prediction_len);
            print_predictions(test_data.get_target(), &test_pred, config.prediction_len);
        }
    }

    println!();
    println!("[   ] objective: {}", objective.name());
    println!("[   ] discretization bits: {}", config.discr_nbits);

    for m in ms {
        let train_eval = m.eval(train.get_target(), &train_pred);
        let test_eval = match test_pred {
            Some((test_data, ref test_pred)) => {
                let test_eval = m.eval(test_data.get_target(), test_pred);
                format!(",   test {:10.4e}", test_eval)
            },
            None => "".to_string(),
        };
        println!("[   ] eval {:13} train {:10.4e}{}", m.name(), train_eval, test_eval);
    }
}

fn print_predictions(target: &[NumT], prediction: &[NumT], npreds: usize) {
    println!("{:4}  {:>15} {:>15} {:>15}", "", "target", "prediction", "error");
    for (i, (x, y)) in target.iter().zip(prediction).enumerate().take(npreds) {
        println!("{:4}: {:15} {:15} {:15.2e}", i, x, y, (x-y).abs());
    }
}

#[allow(dead_code)]
fn write_results(res: &[NumT]) -> Result<(), Error> {

    let mut file = File::create("/tmp/spdyboost_predictions.txt")?;

    for &r in res {
        writeln!(file, "{}", r)?;
    }

    Ok(())
}
