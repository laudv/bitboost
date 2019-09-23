/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

use std::env;
//use std::time::Instant;
use cpu_time::ProcessTime;


//use crossbeam_utils::thread;

use bitboost::NumT;
use bitboost::config::Config;
use bitboost::data::Data;
use bitboost::dataset::Dataset;
use bitboost::objective::{Objective, objective_from_name};
use bitboost::tree::{AdditiveTree, Tree};
use bitboost::tree::{TreeLearner, TreeLearnerContext};
use bitboost::metric::{Metric, metrics_from_names};
use bitboost::boost::Booster;

pub fn main() -> Result<(), String> {
    pretty_env_logger::init();

    let args: Vec<String> = env::args().collect();
    match args.get(1).map(|x| x.as_str()) {
        Some("tree")  => {
            let (conf, d_train, d_test, mut obj, ms) = parse(&args[2..])?;
            let tree = single_tree(&conf, &d_train, obj.as_mut())?;
            summary(&conf, |d| tree.predict(d), &d_train, d_test.as_ref(), obj.as_ref(), &ms);
        },
        Some("boost") => {
            let (conf, d_train, d_test, mut obj, ms) = parse(&args[2..])?;
            let (model, _) = boost(&conf, &d_train, obj.as_mut())?;
            summary(&conf, |d| model.predict(d), &d_train, d_test.as_ref(), obj.as_ref(), &ms);
        },
        Some("boost_and_predict_raw") => {
            let (conf, d_train, d_test, mut obj, _) = parse(&args[2..])?;
            let (model, time) = boost(&conf, &d_train, obj.as_mut())?;
            println!("__TRAIN_TIME__ {}", time);
            print!("__PREDICTIONS_TRAIN__");
            print_predictions_raw(&model.predict(&d_train));
            if let Some(d_test) = d_test {
                print!("__PREDICTIONS_TEST__");
                print_predictions_raw(&model.predict(&d_test));
            }
        },
        _ => {
            let msg = "spdyboost-run (tree|boost|boost_and_predict_raw)[ (field=value)*]";
            return Err(String::from(msg));
        }
    }

    Ok(())
}

fn parse(args: &[String])
    -> Result<(Config, Data, Option<Data>, Box<dyn Objective>, Vec<Box<dyn Metric>>), String>
{
    let config = Config::parse(args.iter().map(|x| x.as_str()))?;
    let objective = objective_from_name(&config.objective)
        .ok_or(format!("unknown objective '{}'", config.objective))?;
    let ms = metrics_from_names(&config.metrics).ok_or("unknown metric".to_string())?;
    let (train_data, test_data) = load_data(&config)?;
    Ok((config, train_data, test_data, objective, ms))
}

fn load_data(config: &Config) -> Result<(Data, Option<Data>), String> {
    //let (train_data_res, test_data_res) = thread::scope(|s| {
        //let t1 = s.spawn(move |_| {
            let train_data_res = Data::from_csv_path(&config, config.train.as_str());
            if train_data_res.is_ok() { println!("[   ] finished loading training data"); }
            else {                      println!("[   ] failed loading training data: '{}'", config.train); }
        //    train_data_res
        //});
        let test_data_res = if !config.test.is_empty() {
            let rtest = Data::from_csv_path(&config, config.test.as_str());
            if rtest.is_ok() { println!("[   ] finished loading test data"); }
            else {         println!("[   ] failed loading test data: '{}'", config.train); }
            Some(rtest)
        } else { None };
        //let train_data_res = t1.join().unwrap();

    //    (train_data_res, test_data_res)
    //}).map_err(|_| "thread error".to_string())?;

    let train_data = train_data_res?;
    let test_data = match test_data_res {
        Some(res) => Some(res?),
        None => None
    };

    Ok((train_data, test_data))
}

fn single_tree(config: &Config, d_train: &Data, objective: &mut dyn Objective)
    -> Result<Tree, String>
{
    let target = d_train.get_target();
    objective.initialize(&config, &target);
    objective.update(&target);

    let mut dataset = Dataset::new(&config, &d_train);
    dataset.update(&config, objective.gradients(), objective.bounds());

    let mut context = TreeLearnerContext::new(&config, &d_train);
    let supercats = dataset.get_supercats().clone();
    let learner = TreeLearner::new(&mut context, &dataset, supercats, objective);
    let mut tree = learner.train();
    tree.set_bias(objective.bias());

    Ok(tree)
}

fn boost(config: &Config, d_train: &Data, objective: &mut dyn Objective)
    -> Result<(AdditiveTree, f32), String>
{
    let ms = metrics_from_names(&config.metrics).ok_or("unknown metric".to_string())?;
    let booster = Booster::new(&config, &d_train, objective, &ms);

    let start = ProcessTime::now();
    let model = booster.train();
    let el = start.elapsed();
    let secs = el.as_secs() as f32 + el.subsec_micros() as f32 * 1e-6;

    Ok((model, secs))
}

fn summary<F>(config: &Config, model: F, train: &Data, test: Option<&Data>,
              objective: &dyn Objective, ms: &[Box<dyn Metric>])
where F: Fn(&Data) -> Vec<NumT>
{
    let train_pred = model(train);
    let test_pred = test.map(|test| (test, model(test)));

    //if config.prediction_len > 0 {
    //    println!();
    //    println!("[   ] train predictions (first {})", config.prediction_len);
    //    print_predictions2(train.get_target(), &train_pred, objective.predictions(), config.prediction_len);

    //    if let Some((test_data, ref test_pred)) = test_pred {
    //        println!();
    //        println!("[   ] test predictions (first {})", config.prediction_len);
    //        print_predictions(test_data.get_target(), &test_pred, config.prediction_len);
    //    }
    //}

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

#[allow(dead_code)]
fn print_predictions(target: &[NumT], prediction: &[NumT], npreds: usize) {
    println!("{:4}  {:>15} {:>15} {:>15}", "", "target", "prediction", "error");
    for (i, (x, y)) in target.iter().zip(prediction).enumerate().take(npreds) {
        println!("{:4}: {:15} {:15} {:15.2e}", i, x, y, (x-y).abs());
    }
}

#[allow(dead_code)]
fn print_predictions2(target: &[NumT], prediction: &[NumT], objective_preds: &[NumT],
                      npreds: usize)
{
    println!("{:4}  {:>15} {:>15} {:>15} {:>15}", "", "target", "prediction", "obj.pred.", "obj.err.");
    for (i, ((t, x1), x2)) in target.iter().zip(prediction).zip(objective_preds).enumerate().take(npreds) {
        println!("{:4}: {:15} {:15} {:15} {:15.2e}", i, t, x1, x2, (x1-x2).abs());
    }
}

#[allow(dead_code)]
fn print_predictions_raw(predictions: &[NumT]) {
    for (i, pred) in predictions.iter().enumerate() {
        if i == 0 { print!("{}", pred); }
        else      { print!(",{}", pred); }
    }
    println!()
}
