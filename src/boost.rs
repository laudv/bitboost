use std::time::Instant;

use log::{debug, info};

use crate::NumT;
use crate::config::Config;
use crate::dataset::Dataset;
use crate::tree::{Tree, AdditiveTree};
use crate::tree::bit_tree_learner::TreeLearner;
use crate::tree::loss;
use crate::tree::loss::{LossFun, LossFunGrad};
use crate::tree::eval::Evaluator;

pub struct Booster<'a> {
    config: &'a Config,
    dataset: &'a Dataset,
    loss: loss::L2Loss,
    targets: &'a [NumT],
    iter_count: usize, 

    predictions: Vec<NumT>,
    gradients: Vec<NumT>,

    ensemble: AdditiveTree,
}

impl <'a> Booster<'a> {
    pub fn new(config: &'a Config, dataset: &'a Dataset) -> Booster<'a> {
        let nexamples = dataset.nexamples();

        let predictions = vec![0.0; nexamples];
        let gradients = vec![0.0; nexamples];
        let ensemble = AdditiveTree::new();

        Booster {
            config,
            dataset,
            loss: loss::L2Loss::new(),
            targets: dataset.target().get_raw_data(),
            iter_count: 0,

            predictions,
            gradients,
            ensemble,
        }
    }

    pub fn train(&mut self) {
        assert!(self.iter_count == 0);
        let start = Instant::now();

        for _ in 0..self.config.niterations {
            self.train_one_iter();

            let el = start.elapsed();
            let seconds = el.as_secs() as f32 + el.subsec_micros() as f32 * 1e-6;
            info!("I{:03} time: {} s", self.iter_count, seconds);
        }
    }

    pub fn train_one_iter(&mut self) {
        self.iter_count += 1;
        let pairs = self.targets.iter().cloned().zip(self.predictions.iter().cloned());
        let (min, max, bias) = self.loss.boost_stats(pairs);
        self.update_gradients(bias);

        let mut tree = {
            let grad_bounds = (min, max);
            let mut learner = TreeLearner::new(self.config, self.dataset, &self.gradients,
                                               grad_bounds);
            let start = Instant::now();
            learner.train();
            let el = start.elapsed();
            let seconds = el.as_secs() as f32 * 1e3 + el.subsec_micros() as f32 * 1e-3;
            info!("I{:03} tree time: {} ms", self.iter_count, seconds);

            learner.into_tree()
        };

        tree.set_bias(bias);
        tree.set_shrinkage(self.config.learning_rate);

        if self.config.optimize_leaf_values {
            let tree_target_iter = self.targets.iter().cloned()
                .zip(self.predictions.iter().cloned())
                .map(|(t, p)| t-p);
            tree.optimize_leaf_values(self.dataset, tree_target_iter);
        }

        self.update_predictions(&tree);
        self.ensemble.push_tree(tree);

        let metric = self.evaluate();
        info!("I{:03} range: [{:.3}, {:.3}], bias: {:.4}, {} metric: {}",
              self.iter_count, min, max, bias, self.loss.evaluator_name(), metric);
    }

    fn update_gradients(&mut self, prediction_bias: NumT) {
        for i in 0..self.dataset.nexamples() {
            let t = self.targets[i];
            let p = self.predictions[i] + prediction_bias;
            self.gradients[i] = self.loss.eval_grad(t, p);
        }
    }

    fn update_predictions(&mut self, new_tree: &Tree) {
        new_tree.predict_and(self.dataset, &mut self.predictions, |prediction, accum| {
            *accum += prediction;
        });
    }

    fn evaluate(&self) -> NumT {
        self.loss.evaluate(self.targets.iter().cloned(),
                           self.predictions.iter().cloned())
    }

    pub fn into_model(self) -> AdditiveTree {
        self.ensemble
    }
}
