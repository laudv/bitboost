use std::time::Instant;

use log::{debug, info};

use crate::NumT;
use crate::config::Config;
use crate::data::{Data, Dataset};
use crate::tree::{AdditiveTree};
use crate::tree_learner::{TreeLearnerContext, TreeLearner};
use crate::objective::{Objective};
use crate::metric::Metric;

macro_rules! time {
    ($iter:expr, $msg:expr, $($block:tt)*) => {{
        let start = Instant::now();
        let res = {
            $($block)*
        };
        let el = start.elapsed();
        let seconds = el.as_secs() as f32 + el.subsec_micros() as f32 * 1e-6;
        info!("I{:03} {}: {} s", $iter, $msg, seconds);
        res
    }}
}

pub struct Booster<'a> {
    config: &'a Config,
    data: &'a Data,
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
        let dataset = Dataset::new(data);
        Booster {
            config,
            data,
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
        let target = self.data.get_target();
        let mut r = TreeLearnerContext::new(self.config, self.data);

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
        let target = self.data.get_target();
        self.iter_count += 1;
        time!(self.iter_count, "obj time", self.objective.update(target));
        time!(self.iter_count, "dataset time", 
              self.dataset.update(self.config, self.objective.gradients(), self.objective.bounds()));

        let tree = {
            let learner = TreeLearner::new(ctx, &self.dataset, self.objective);
            time!(self.iter_count, "tree time", learner.train())
        };

        if !self.metrics.is_empty() {
            let pred = tree.predict(self.data);
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
