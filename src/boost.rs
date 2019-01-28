use std::time::Instant;

use log::{debug, info};

use crate::NumT;
use crate::config::Config;
use crate::dataset::Dataset;
use crate::tree::{AdditiveTree};
use crate::tree_learner::{TreeLearnerContext, TreeLearner};
use crate::objective::{Objective, objective_from_name};
use crate::metric::Metric;

pub struct Booster<'a> {
    config: &'a Config,
    dataset: &'a Dataset,
    objective: Box<dyn Objective>,
    iter_count: usize, 
    metrics: &'a [Box<dyn Metric>],
    ensemble: AdditiveTree,
}

impl <'a> Booster<'a> {
    pub fn new(config: &'a Config, dataset: &'a Dataset, metrics: &'a [Box<dyn Metric>])
        -> Booster<'a>
    {
        let objective = objective_from_name(&config.objective, config);
        let ensemble = AdditiveTree::new();

        Booster {
            config,
            dataset,
            objective,
            iter_count: 0,
            metrics,
            ensemble,
        }
    }

    pub fn train(mut self) -> AdditiveTree {
        assert!(self.iter_count == 0);
        let start = Instant::now();
        let targets = self.dataset.target().get_raw_data();
        let mut r = TreeLearnerContext::new(self.config, self.dataset);

        self.objective.initialize(self.config, targets);
        self.ensemble.set_bias(self.objective.bias());

        for _ in 0..self.config.niterations {
            self.train_one_iter(&mut r);

            let el = start.elapsed();
            let seconds = el.as_secs() as f32 + el.subsec_micros() as f32 * 1e-6;
            info!("I{:03} time: {} s", self.iter_count, seconds);
        }

        self.ensemble
    }

    fn train_one_iter(&mut self, ctx: &mut TreeLearnerContext) {
        let targets = self.dataset.target().get_raw_data();
        self.iter_count += 1;
        self.objective.update(targets);

        let mut tree = {
            let learner = TreeLearner::new(ctx, self.objective.as_mut());
            let start = Instant::now();
            let tree = learner.train();
            let el = start.elapsed();
            let seconds = el.as_secs() as f32 * 1e3 + el.subsec_micros() as f32 * 1e-3;
            info!("I{:03} tree time: {} ms", self.iter_count, seconds);

            tree
        };

        if !self.metrics.is_empty() {
            let pred = tree.predict(self.dataset);
            let mag = pred.iter()
                .map(|&x| x * x)
                .fold(0.0, |x, y| x+y) / pred.len() as NumT;
            info!("I{:03} tree value magnitude: {:e}", self.iter_count, mag);
        }

        // shrinkage is automatically applied by objective!
        self.ensemble.push_tree(tree);

        if !self.metrics.is_empty() {
            for m in self.metrics {
                let eval = m.eval(&targets, self.objective.predictions());
                info!(" |   eval {:<15}: {:e}", m.name(), eval);
            }
        }
    }
}
