use log::{info, debug};

use bits::{BitSet, ScaledBitSlice};
use conf::Config;
use dataset::{DataSet, Feature, FeatureData};
use dataset::{NumericalType as NumT, NominalType};
use tree::{Tree, SplitCrit};
use tree::eval::SplitEvaluator;

struct Split<'a> {
    feature_id: usize,
    split_crit: SplitCrit,
    left_examples: &'a BitSet,
    left_value: NumT,
    right_value: NumT,
    left_loss: NumT,
    right_loss: NumT,
}

struct NodeToSplit {
    node_id: usize,
    loss: NumT,
    examples: BitSet,
}

impl NodeToSplit {
    fn new(n: usize, l: NumT, e: BitSet) -> NodeToSplit {
        NodeToSplit { node_id: n, loss: l, examples: e }
    }
}

// ------------------------------------------------------------------------------------------------

pub struct TreeLearner<'a, SE>
where SE: 'a + SplitEvaluator<ExampleSelector = BitSet>
{
    config: &'a Config,
    dataset: &'a DataSet,
    split_evaluator: &'a SE,

    /// The data for the split evaluator. This is an encoding of the target values to be learned by
    /// the tree.
    pub evaluator_data: SE::EvaluationData,

    /// The tree in construction.
    tree: Tree,
}

impl <'a, SE> TreeLearner<'a, SE>
where SE: 'a + SplitEvaluator<EvaluationData = ScaledBitSlice<NumT>, ExampleSelector = BitSet>
{
    pub fn new<I1, I2>(config: &'a Config, training_set: &'a DataSet, evaluator: &'a SE,
                  target_values: I1, prev_predictions: I2) -> TreeLearner<'a, SE>
    where I1: Iterator<Item = NumT>,
          I2: Iterator<Item = NumT>,
    {
        let max_depth = config.max_tree_depth;
        let nexamples = training_set.nexamples();
        let eval_data = evaluator.convert_target_values(nexamples, target_values,
                                                        prev_predictions);
        TreeLearner {
            config: config,
            dataset: training_set,
            split_evaluator: evaluator,
            evaluator_data: eval_data,
            tree: Tree::new(max_depth),
        }
    }

    pub fn train(&mut self) {
        let max_depth = self.tree.get_max_depth();
        let nexamples = self.dataset.nexamples();
        let mut split_stack: Vec<NodeToSplit> = Vec::with_capacity(max_depth*2+1); // DFS stack

        // Evaluate root: find optimal value and loss and schedule split
        let root_examples = BitSet::trues(nexamples);
        let (left, _) = self.split_evaluator.eval_split(
            &self.evaluator_data, &root_examples, &root_examples);
        split_stack.push(NodeToSplit::new(0, left.loss, root_examples));

        debug!("ROOT: loss {}, value {}", left.loss, left.optimal_value);

        // Split until tree is complete
        while let Some(node_to_split) = split_stack.pop() {
            let NodeToSplit { node_id, loss, examples } = node_to_split;

            // Find the best split
            let left_id = self.tree.left_child(node_id);
            let right_id = self.tree.right_child(node_id);
            let split = self.find_best_split(loss, &examples);
            let gain = loss - split.left_loss - split.right_loss;

            debug!("SPLIT N{:02}-F{:02}: gain {}, {:?}", node_id, split.feature_id,
                   gain, split.split_crit);

            // Schedule left and right leaf to be split if not leaves. If left is leaf, then right
            // is as well.
            if !self.tree.is_leaf_node(left_id) {
                let left_examples = examples.and(split.left_examples).into_bitset(nexamples);
                let right_examples = examples.andnot(split.left_examples).into_bitset(nexamples);
                split_stack.push(NodeToSplit::new(right_id, split.right_loss, right_examples));
                split_stack.push(NodeToSplit::new(left_id, split.left_loss, left_examples));
            }
        }

        info!("Tree of depth {} constructed", self.tree.get_max_depth());
    }

    fn find_best_split(&self, parent_loss: NumT, node_examples: &BitSet) -> Split<'a> {
        let mut best_gain = -1.0 / 0.0;
        let mut best_split = None;

        for feature in self.dataset.get_input_features() {
            match feature.get_data() {
                &FeatureData::BitSets(ref feature_data) => {
                    let split = self.find_best_lowcard_nom_split(
                        parent_loss,
                        feature.get_id(),
                        node_examples,
                        feature_data);

                    let gain = parent_loss - split.left_loss - split.right_loss;
                    if best_gain < gain {
                        best_gain = gain;
                        best_split = Some(split);
                    }
                },
                _ => {
                    panic!("feature type not supported")
                }
            }
        }

        best_split.unwrap()
    }

    /// Find the best split for a low cardinality feature on the active node
    fn find_best_lowcard_nom_split(&self, parent_loss: NumT, feature_id: usize,
        node_examples: &BitSet, feature_data: &'a [(NominalType, BitSet)]) -> Split<'a>
    {
        let mut best_gain = -1.0 / 0.0;
        let mut best_split = None;

        for (feature_value, value_examples) in feature_data.into_iter() {
            let (left, right) = self.split_evaluator.eval_split(&self.evaluator_data,
                                                                node_examples,
                                                                value_examples);
            let gain = parent_loss - left.loss - right.loss;

            debug!("F{:02}-VAL{:02}: gain {:.2e} ({:4}:{:+.2}|{:4}:{:+.2})",
                feature_id, feature_value, gain,
                left.example_count,  left.optimal_value, 
                right.example_count, right.optimal_value);

            if best_gain < gain {
                best_gain = gain;
                best_split = Some(Split {
                    feature_id: feature_id,
                    split_crit: SplitCrit::EqTest(*feature_value),
                    left_examples: &value_examples,
                    left_value: left.optimal_value,
                    right_value: right.optimal_value,
                    left_loss: left.loss,
                    right_loss: right.loss,
                });
            }
        }

        best_split.unwrap()
    }

    pub fn into_tree(self) -> Tree {
        self.tree
    }
}
