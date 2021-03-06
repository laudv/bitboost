/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

use std::time::Instant;

use crate::config::Config;
use crate::data::{Data};
use crate::dataset::Dataset;
use crate::tree::{AdditiveTree};
use crate::tree::{TreeLearnerContext, TreeLearner};
use crate::objective::{Objective};
use crate::metric::Metric;

macro_rules! time {
    ($($block:tt)*) => {{
        let start = Instant::now();
        let res = {
            $($block)*
        };
        let el = start.elapsed();
        let seconds = el.as_secs() as f32 + el.subsec_micros() as f32 * 1e-6;
        (res, seconds)
    }}
}

pub struct Booster<'a> {
    config: &'a Config,
    data: &'a Data,
    start: Instant,
    dataset: Dataset<'a>,
    objective: &'a mut dyn Objective,
    iter_count: usize, 
    metrics: &'a [Box<dyn Metric>],
    ensemble: AdditiveTree,
}

impl <'a> Booster<'a> {
    pub fn new(config: &'a Config, data: &'a Data,
               objective: &'a mut dyn Objective,
               metrics: &'a [Box<dyn Metric>]) -> Booster<'a>
    {
        let ensemble = AdditiveTree::new();
        let dataset = Dataset::new(config, data);
        Booster {
            config,
            data,
            start: Instant::now(),
            dataset,
            objective,
            iter_count: 0,
            metrics,
            ensemble,
        }
    }

    pub fn train(mut self) -> AdditiveTree {
        assert!(self.iter_count == 0);

        self.print_intro();

        self.start = Instant::now();
        let target = self.data.get_target();
        let mut ctx = TreeLearnerContext::new(self.config, self.data);

        self.objective.initialize(self.config, target);
        self.ensemble.set_bias(self.objective.bias());

        for _ in 0..self.config.niterations {
            self.train_one_iter(&mut ctx);
        }

        self.ensemble
    }

    fn train_one_iter(&mut self, ctx: &mut TreeLearnerContext) {
        let target = self.data.get_target();
        self.iter_count += 1;
        let (_, ot) = time!(self.objective.update(target));
        let (_, dt) = time!(self.dataset.update(self.config, self.objective.gradients(),
                                                self.objective.bounds()));
        // learn a tree
        let supercats = self.dataset.get_supercats().clone();
        let learner = TreeLearner::new(ctx, &self.dataset, supercats, self.objective);
        let (tree, tt) = time!(learner.train());

        // print updates
        let el = self.start.elapsed();
        let seconds = el.as_secs() as f32 + el.subsec_micros() as f32 * 1e-6;
        println!("[{:3}] timings: objective {:5.1}, dataset {:5.1}, tree {:5.1} ms, total {:.3} s",
                 self.iter_count, ot * 1000.0, dt * 1000.0, tt * 1000.0, seconds);
        //let grad_bounds = self.objective.bounds();
        //println!("[   ] gradient bounds ({}, {})", grad_bounds.0, grad_bounds.1);

        let run_metrics = !self.metrics.is_empty()
            && self.config.metric_frequency > 0
            && self.iter_count % self.config.metric_frequency == 0;
        if run_metrics {
            //let pred = tree.predict(self.data);
            //let mag = pred.iter()
            //    .map(|&x| x * x)
            //    .fold(0.0, |x, y| x+y)
            //    .sqrt() / pred.len() as NumT;
            //println!("I{:03} tree value magnitude: {:e}", self.iter_count, mag);

            for m in self.metrics {
                let eval = m.eval(&target, self.objective.predictions());
                println!("[   ] eval {:<13} {:10.4e}", m.name(), eval);
            }
        }

        // shrinkage is automatically applied by objective!
        self.ensemble.push_tree(tree);
    }

    fn print_intro(&self) {
        debug_assert!({
            println!("[ ! ] debug build");
            true
        });
        safety_check!({
            println!("[ ! ] safety checks enabled");
            true
        });
    }
}
