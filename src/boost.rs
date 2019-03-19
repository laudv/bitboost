use std::time::Instant;

use log::{debug, info};

use crate::NumT;
use crate::config::Config;
use crate::data::{Data, Dataset};
use crate::tree::{AdditiveTree};
use crate::tree_learner::{TreeLearnerContext, TreeLearner};
use crate::objective::{Objective, objective_from_name};
use crate::metric::Metric;

pub struct Booster<'a> {
    config: &'a Config,
    dataset: Dataset<'a>,
    objective: Box<dyn Objective>,
    iter_count: usize, 
    metrics: &'a [Box<dyn Metric>],
    ensemble: AdditiveTree,
}

impl <'a> Booster<'a> {
    pub fn new(config: &'a Config, data: &'a Data, metrics: &'a [Box<dyn Metric>])
        -> Option<Booster<'a>>
    {
        let objective = objective_from_name(&config.objective, config)?;
        let ensemble = AdditiveTree::new();
        let dataset = Dataset::construct_from_data(&config, data, objective.gradients());

        Some(Booster {
            config,
            dataset,
            objective,
            iter_count: 0,
            metrics,
            ensemble,
        })
    }

    pub fn train(mut self) -> AdditiveTree {
        assert!(self.iter_count == 0);
        let start = Instant::now();
        let target = self.dataset.get_target();
        let mut r = TreeLearnerContext::new(self.config, &self.dataset);

        self.objective.initialize(self.config, target);
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
        let target = self.dataset.get_target();
        self.iter_count += 1;
        self.objective.update(target);

        let tree = {
            let learner = TreeLearner::new(ctx, self.objective.as_mut());
            let start = Instant::now();
            let tree = learner.train();
            let el = start.elapsed();
            let seconds = el.as_secs() as f32 * 1e3 + el.subsec_micros() as f32 * 1e-3;
            info!("I{:03} tree time: {:.2} ms, #leaf={}", self.iter_count, seconds, tree.nleafs());

            tree
        };

        if !self.metrics.is_empty() {
            let pred = tree.predict(self.dataset);
            let mag = pred.iter()
                .map(|&x| x * x)
                .fold(0.0, |x, y| x+y)
                .sqrt() / pred.len() as NumT;
            info!("I{:03} tree value magnitude: {:e}", self.iter_count, mag);
        }

        // shrinkage is automatically applied by objective!
        self.ensemble.push_tree(tree);

        if !self.metrics.is_empty() {
            for m in self.metrics {
                let eval = m.eval(&target, self.objective.predictions());
                info!(" |   eval {:<15}: {:e}", m.name(), eval);
            }
        }
    }
}
