use std::ops::{AddAssign, SubAssign};
use log::debug;

use NumT;
use NomT;

use tree::{Tree, SplitCrit};
use tree::baseline::HistStore;
use config::Config;
use dataset::{Dataset, FeatureRepr};


struct Node2Split<'a> {
    node_id: usize,
    loss: NumT,
    examples: &'a mut [usize],
}

impl <'a> Node2Split<'a> {
    fn new(id: usize, loss: NumT, exs: &mut [usize]) -> Node2Split {
        Node2Split { node_id: id, loss: loss, examples: exs }
    }
}

struct Split {
    feature_id: usize,
    split_crit: SplitCrit,
    left_value: NumT,
    right_value: NumT,
    left_loss: NumT,
    right_loss: NumT,
}

#[derive(PartialEq, Clone, Default)]
struct HistVal {
    grad_sum: NumT,
    hess_sum: NumT,
}

impl AddAssign<(NumT, NumT)> for HistVal {
    fn add_assign(&mut self, o: (NumT, NumT)) {
        self.grad_sum += o.0;
        self.hess_sum += o.1;
    }
}

impl SubAssign<(NumT, NumT)> for HistVal {
    fn sub_assign(&mut self, o: (NumT, NumT)) {
        self.grad_sum -= o.0;
        self.hess_sum -= o.1;
    }
}

/// Baseline tree learner. Assumes constant second gradient (hessian) of one (= example count).
pub struct TreeLearner<'a> {
    config: &'a Config,
    dataset: &'a Dataset,
    gradients: Vec<NumT>,
    tree: Tree,
    hist_store: HistStore<HistVal>,
}

impl <'a> TreeLearner<'a> {
    pub fn new(config: &'a Config, data: &'a Dataset, gradients: Vec<NumT>) -> TreeLearner<'a> {
        let tree = Tree::new(config.max_tree_depth);
        let hist_store = HistStore::for_dataset(data);

        TreeLearner {
            config: config,
            dataset: data,
            gradients: gradients,
            tree: tree,
            hist_store: hist_store,
        }
    }

    pub fn train(&mut self) {
        let max_depth = self.tree.max_depth();
        let nexamples = self.dataset.nexamples();
        let mut root_examples: Vec<usize> = (0..nexamples).collect();
        let mut split_stack: Vec<Node2Split> = Vec::with_capacity(max_depth * 2 + 1);

        let (root_value, root_loss) = self.eval_root_loss(&root_examples);
        self.tree.set_root_value(root_value);
        split_stack.push(Node2Split::new(0, root_loss, &mut root_examples));

        // Split until no more splits (bad gains or full tree)
        while let Some(node2split) = split_stack.pop() {
            let Node2Split { node_id, loss, examples } = node2split;

            let left_id = self.tree.left_child(node_id);
            let right_id = self.tree.right_child(node_id);
            let split_opt = self.find_best_split(loss, &examples);

            // don't split if no better split is found
            if split_opt.is_none() { debug!("N{:03} no split", node_id); continue; }

            let split = split_opt.unwrap();
            let gain = loss - split.left_loss - split.right_loss;

            debug!("N{:02}-F{:02} {:?}: gain {} ({:.4}, {:.4})", node_id, split.feature_id,
                split.split_crit, gain, split.left_value, split.right_value);

            self.tree.split_node(node_id, split.split_crit.clone(),
                                 split.left_value, split.right_value);

            if !self.tree.is_max_leaf_node(left_id) {
                // Split indirection to avoid borrower issues
                let split_pos = self.split_examples(examples, &split.split_crit);
                let (left_ex, right_ex) = examples.split_at_mut(split_pos);
                split_stack.push(Node2Split::new(left_id, split.left_loss, left_ex));
                split_stack.push(Node2Split::new(right_id, split.right_loss, right_ex));
            }
        }
    }

    fn find_best_split(&self, parent_loss: NumT, parent_examples: &[usize]) -> Option<Split> {
        let mut best_gain = 0.0;
        let mut best_split = None;

        for feature in self.dataset.features() {
            if let Some(&FeatureRepr::CatFeature(card, ref data)) = feature.get_repr() {
                let split_opt = self.find_best_cat_feat_split(feature.id(), parent_loss,
                                                              parent_examples, card, data);
                if split_opt.is_none() { continue; }
                let split = split_opt.unwrap();

                let gain = parent_loss - split.left_loss - split.right_loss;
                if best_gain < gain {
                    best_gain = gain;
                    best_split = Some(split);
                }
            } else {
                unimplemented!("numerical features not implemented")
            }
        }

        best_split
    }

    fn find_best_cat_feat_split(&self, feature_id: usize, parent_loss: NumT,
                                parent_examples: &[usize],
                                cardinality: usize, data: &[NomT]) -> Option<Split>
    {
        let lambda = self.config.reg_lambda;
        let min_hess = self.config.min_sum_hessian;

        let mut grad_sums = vec![0.0; cardinality];
        let mut hess_sums = vec![0.0; cardinality];
        let mut grad_total = 0.0 as NumT;
        let hess_total = parent_examples.len() as NumT;

        // HOTTEST CODE! --> Reuse histogram
        for &i in parent_examples {
            let j = data[i] as usize;
            grad_sums[j] += self.gradients[i];
            hess_sums[j] += 1.0;
            grad_total += self.gradients[i];
        }

        let mut best_gain = 0.0;
        let mut best_split = None;

        for j in 0..cardinality {
            let left_grad = grad_sums[j];
            let left_hess = hess_sums[j]; 
            let right_grad = grad_total - left_grad;
            let right_hess = hess_total - left_hess;

            // eliminates splits where all examples go left/right
            if left_hess < min_hess || right_hess < min_hess { continue; }

            let left_value = -left_grad / (left_hess + lambda);
            let right_value = -right_grad / (right_hess + lambda);

            let left_loss = -0.5 * ((left_grad * left_grad) / (left_hess + lambda));
            let right_loss = -0.5 * ((right_grad * right_grad) / (right_hess + lambda));

            let gain = parent_loss - left_loss - right_loss;

            debug!("F{:02}={:2} gain {:+6.1} ({:.4}, {:.4})", feature_id, j, gain, left_value, right_value);

            if gain > best_gain {
                best_gain = gain;
                best_split = Some(Split {
                    feature_id: feature_id,
                    split_crit: SplitCrit::EqTest(feature_id, data[j]),
                    left_value: left_value,
                    right_value: right_value,
                    left_loss: left_loss,
                    right_loss: right_loss,
                });
            }
        }

        best_split
    }

    fn split_examples(&'a self, examples: &'a mut [usize], crit: &'a SplitCrit) -> usize {
        match &crit {
            SplitCrit::Undefined => { panic!("unexpected undefined split crit"); },
            SplitCrit::EqTest(feat_id, value) => {
                let (_, cat_data) = self.dataset
                    .get_feature(*feat_id).unwrap()
                    .get_cat_feature_repr().unwrap();
                self.split_examples_cmp(examples, move |i| cat_data[i] == *value)
            }
        }
    }

    fn split_examples_cmp<F>(&'a self, examples: &'a mut [usize], is_left_fn: F) -> usize
    where F: 'a + Fn(usize) -> bool {
        let mut left_i = 0;               // index first unknown example (left or right)
        let mut right_i = examples.len(); // index first right example (invalid initally)

        while left_i != right_i {
            if is_left_fn(left_i) { left_i += 1; }
            else {
                right_i -= 1;
                examples.swap(left_i, right_i);
            }
        }

        // The order of the right examples is reversed
        //let (_, r) = examples.split_at_mut(left_i);
        //r.reverse();
        left_i
    }

    pub fn reset(&mut self) {
        self.tree = Tree::new(self.config.max_tree_depth);
        self.hist_store = HistStore::for_dataset(self.dataset);
    }

    fn best_value_and_loss(&self, grad: NumT, hess: NumT) -> (NumT, NumT) {
        let lambda = self.config.reg_lambda;
        let value = -grad / (hess + lambda);
        let loss = -0.5 * ((grad * grad) / (hess + lambda));
        (value, loss)
    }

    fn eval_root_loss(&mut self, examples: &[usize]) -> (NumT, NumT) {
        let mut grad = 0.0;
        let hess = examples.len() as NumT;
        for &i in examples { grad += self.gradients[i]; }

        self.best_value_and_loss(grad, hess)
    }

    fn build_hist_cat(&self, node_id: usize, parent_examples: &[usize], data: &[NomT],
                      buckets: &mut [HistVal]) {
        for &i in parent_examples {
            let j = data[i] as usize; // bucket number
            buckets[j] += (self.gradients[i], 1.0);
        }
    }
}
