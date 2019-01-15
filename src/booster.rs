use crate::NumT;
use crate::config::Config;
use crate::dataset::Dataset;
use crate::tree::Tree;
use crate::tree::bit_tree_learner::TreeLearner;

pub struct Booster<'a> {
    config: &'a Config,
    dataset: &'a Dataset,

    predictions: Vec<NumT>,
}

impl <'a> Booster<'a> {

    pub fn new(config: &'a Config, dataset: &'a Dataset) -> Booster<'a> {
        let nexamples = dataset.nexamples();
        Booster {
            config,
            dataset,
            predictions: Vec::with_capacity(nexamples),
        }
    }

    pub fn train(&self) {

    }
}
