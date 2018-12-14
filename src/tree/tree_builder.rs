use log::{debug};

use bits::{BitSet};
use conf::Config;
use dataset::{DataSet, Feature, FeatureData, NumericalType, NominalType};
use tree::{Tree, SplitCrit};
use tree::TargetValues;

#[derive(Clone)]
pub struct Split {
    feature_id: usize,
    split_crit: SplitCrit,
    left_value: NumericalType,
    right_value: NumericalType,
    gain: NumericalType,
}

struct ParentInfo {
    node_id: usize,
    loss: NumericalType,
    example_selection: BitSet,
}

impl ParentInfo {
    fn new(node_id: usize, loss: NumericalType, example_selection: BitSet) -> ParentInfo {
        ParentInfo {
            node_id: node_id,
            loss: loss,
            example_selection: example_selection,
        }
    }

    fn unpack(self) -> (usize, NumericalType, BitSet) {
        (self.node_id, self.loss, self.example_selection)
    }
}


// ------------------------------------------------------------------------------------------------

pub struct TreeBuilder<'a> {
    config: &'a Config,
    dataset: &'a DataSet,

    /// The (approximated) target values to be learned by this tree.
    target_values: &'a TargetValues,

    /// The tree in construction.
    tree: Tree,

    /// A stack of nodes to split.
    split_stack: Vec<ParentInfo>,
}

impl <'a> TreeBuilder<'a> {
    pub fn new(config: &'a Config, training_set: &'a DataSet,
               target_values: &'a TargetValues) -> TreeBuilder<'a> {
        let max_depth = config.max_tree_depth;
        let nexamples = training_set.nexamples();
        let mut builder = TreeBuilder {
            config: config,
            dataset: training_set,
            target_values: target_values,
            tree: Tree::new(max_depth),
            split_stack: Vec::with_capacity(max_depth*2+1),
        };

        // Bitset of examples in root node; sampling can happen here
        builder.split_stack.push(ParentInfo::new(0, 0.0, BitSet::trues(nexamples)));

        // Value of root node
        builder.tree.set_root_value(target_values.mean());

        builder
    }

    pub fn train(&mut self) {
        while !self.split_stack.is_empty() {
            let (node_id, loss, bitset) = self.split_stack.pop().unwrap().unpack();
            debug_assert!(self.tree.is_node(node_id));
            debug!("visiting node_id {}", node_id);

            if !self.tree.is_leaf_node(node_id) {
                let n = self.dataset.nexamples();
                let left_id = self.tree.left_child(node_id);
                let right_id = self.tree.right_child(node_id);

                //self.split_stack.push((right_id, BitSet::falses(n)));
                //self.split_stack.push((left_id, BitSet::falses(n)));
            }
        }
    }

    /// Find the best split on the active node
    pub fn find_best_split(&self) -> Split {
        // Best split for each feature
        let mut feature_best: Vec<Split> = Vec::with_capacity(self.dataset.ninput_features());

        // Split buffer to avoid allocations
        let mut split_buffer: Vec<Split> = Vec::with_capacity(128);

        for feature in self.dataset.get_input_features() {
            let fbest = self.find_best_feature_split_crit(feature, &mut split_buffer);
            feature_best.push(fbest);
        }

        self.select_best_split(&mut feature_best)
    }

    /// Find the best split for the given feature on the active node
    fn find_best_feature_split_crit(&self, feature: &Feature, split_buffer: &mut Vec<Split>)
        -> Split
    {
        match feature.get_data() {
            &FeatureData::BitSets(ref bitsets) => {
                self.find_best_lowcard_nom_split(bitsets, split_buffer)
            },
            _ => {
                panic!("Feature type supported")
            }
        }
    }

    /// Find the best split for a low cardinality feature on the active node
    fn find_best_lowcard_nom_split(&self, value_bitsets: &[(NominalType, BitSet)],
        split_buffer: &mut Vec<Split>) -> Split
    {
        for i in 0..value_bitsets.len() {
            let split = self.compute_lowcard_nom_split(value_bitsets, i);
            split_buffer.push(split);
        }
        self.select_best_split(split_buffer)
    }

    /// Compute split information for specific value of low cardinality feature on active node 
    fn compute_lowcard_nom_split(&self, value_bitsets: &[(NominalType, BitSet)], value_id: usize)
        -> Split
    {
        let value_mask = &value_bitsets[value_id].1;
        //let parent_examples: &BitSet = self.bitset_stack.last().unwrap();

        let left_value = 0.0;
        let right_value = 0.0;

        let gain = 0.0;

        unimplemented!()
    }

    fn select_best_split(&self, split_buffer: &mut Vec<Split>) -> Split {
        let mut max_gain = -1.0 / 0.0;
        let best = {
            let mut best = None;
            for split in split_buffer.iter() {
                if max_gain <= split.gain {
                    max_gain = split.gain;
                    best = Some(split);
                }
            }
            best.expect("0 features or invalid gain values").clone()
        };
        split_buffer.clear();
        best
    }

    pub fn into_tree(self) -> Tree {
        self.tree
    }
}
