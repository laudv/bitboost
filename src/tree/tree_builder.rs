use bits::{BitSet};
use conf::Config;
use dataset::{DataSet, NumericalType};
use tree::{Tree, SplitCrit};
use tree::TargetValues;

pub struct TreeBuilder<'a> {
    config: &'a Config,
    dataset: &'a DataSet,

    /// The (approximated) target values to be learned by this tree.
    target_values: &'a TargetValues,

    /// The tree in construction
    tree: Tree,

    /// Stores the selection of the current node to be split and all of its ancestors (max
    /// max_depth bitsets)
    selection_path: Vec<BitSet>,

    /// The next node to split
    active_node: usize,
}

impl <'a> TreeBuilder<'a> {
    pub fn new(config: &'a Config, training_set: &'a DataSet,
               target_values: &'a TargetValues) -> TreeBuilder<'a> {
        let max_depth = config.max_tree_depth;
        TreeBuilder {
            config: config,
            dataset: training_set,
            target_values: target_values,
            tree: Tree::new(max_depth),
            selection_path: Vec::new(),
            active_node: 0,
        }
    }

    /// Find the best split feature and value of the currently active node. Returns a tuple
    /// (feature_id, split criterion, loss gain).
    pub fn find_best_active_node_split(&self) -> (usize, SplitCrit, NumericalType) {
        unimplemented!()
    }

    pub fn train() -> Result<(), String> {
        unimplemented!()
    }

    pub fn into_tree(self) -> Tree {
        self.tree
    }
}
