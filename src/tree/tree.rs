use std::fmt::{Debug, Formatter, Result as FmtResult};

use crate::{NumT, NomT};
use crate::dataset::Dataset;
use crate::tree::loss::LossFun;

#[derive(Debug, Clone, PartialEq)]
pub enum SplitCrit {
    Undefined,

    /// (feat_id, cat_value), equal goes left, not equal goes right
    EqTest(usize, NomT),
}

impl SplitCrit {
    pub fn unpack_eqtest(&self) -> Option<(usize, NomT)> {
        if let &SplitCrit::EqTest(feat_id, feat_val) = self {
            Some((feat_id, feat_val))
        } else { None }
    }
}

pub struct Tree {
    ninternal: usize,
    max_depth: usize,
    split_crits: Vec<SplitCrit>,
    node_values: Vec<NumT>,

    shrinkage: NumT,
    bias: NumT,
}

impl Tree {
    pub fn new(max_depth: usize) -> Tree {
        assert!(max_depth > 0);
        let mut tree = Tree {
            ninternal: 0, // the root is the only leaf
            max_depth: max_depth,
            split_crits: Vec::new(),
            node_values: Vec::new(),

            shrinkage: 1.0,
            bias: 0.0,
        };
        let nnodes = tree.max_nnodes();

        tree.split_crits.resize(nnodes, SplitCrit::Undefined);
        tree.node_values.resize(nnodes, 0.0);

        tree
    }

    pub fn is_valid_node_id(&self, node_id: usize) -> bool { node_id < self.nnodes() }
    pub fn left_child(&self, node_id: usize) -> usize { 2 * node_id + 1 }
    pub fn right_child(&self, node_id: usize) -> usize { 2 * node_id + 2 }
    pub fn parent(&self, node_id: usize) -> usize { (node_id - 1) / 2 }

    pub fn max_ninternal(&self) -> usize { (1 << self.max_depth) - 1 }
    pub fn max_nleaves(&self) -> usize { 1 << self.max_depth }
    pub fn ninternal(&self) -> usize { self.ninternal }
    pub fn nleaves(&self) -> usize { self.ninternal + 1 }
    pub fn max_nnodes(&self) -> usize { (1 << (self.max_depth+1)) - 1 }
    pub fn nnodes(&self) -> usize { self.ninternal() + self.nleaves() }

    pub fn node_value(&self, node_id: usize) -> NumT { self.node_values[node_id] }

    pub fn max_depth(&self) -> usize { self.max_depth }
    pub fn set_root_value(&mut self, value: NumT) {
        self.node_values[0] = value;
    }

    /// Can this node still be split? We can't grow deeper than max_depth.
    pub fn is_max_leaf_node(&self, node_id: usize) -> bool {
        node_id >= self.max_ninternal()
    }

    pub fn split_node(&mut self, node_id: usize, split_crit: SplitCrit, left_value: NumT,
                      right_value: NumT)
    {
        assert_eq!(self.split_crits[node_id], SplitCrit::Undefined);
        assert!(!self.is_max_leaf_node(node_id));

        let left_id = self.left_child(node_id);
        let right_id = self.right_child(node_id);

        self.split_crits[node_id] = split_crit;
        self.node_values[left_id] = left_value;
        self.node_values[right_id] = right_value;

        self.ninternal += 1;
    }

    fn predict_leaf_id_of_example(&self, dataset: &Dataset, i: usize) -> usize {
        let mut node_id = 0;
        loop {
            let split_crit = &self.split_crits[node_id];
            match split_crit {
                &SplitCrit::EqTest(feat_id, split_val) => { // this is an internal node
                    let value = dataset.get_cat_value(feat_id, i).expect("invalid feat_id");
                    node_id = if value == split_val { self.left_child(node_id) }
                              else                  { self.right_child(node_id) };
                },
                &SplitCrit::Undefined => { // this is a leaf node
                    return node_id;
                },
            }
        }
    }

    /// Optimize by taking the mean of the targets in each leaf. This is optimal for L2
    /// TODO generalize this for more loss functions.
    pub fn optimize_leaf_values<I>(&mut self, dataset: &Dataset, targets: I)
    where I: Iterator<Item=NumT>,
    {
        let mut leaf_sums = vec![0.0; self.max_nnodes()];
        let mut leaf_counts = vec![0; self.max_nnodes()];

        for (i, target) in targets.enumerate() {
            let leaf_id = self.predict_leaf_id_of_example(dataset, i);
            leaf_sums[leaf_id] += target;
            leaf_counts[leaf_id] += 1;
        }

        let mut stack = Vec::new();
        stack.push(0);
        while let Some(node_id) = stack.pop() {
            match self.split_crits[node_id] {
                SplitCrit::Undefined => {
                    let sum = leaf_sums[node_id];
                    let count = leaf_counts[node_id] as NumT;
                    self.node_values[node_id] = sum / count;
                },
                _ => {
                    stack.push(self.left_child(node_id));
                    stack.push(self.right_child(node_id));
                }
            }
        }
    }

    pub fn set_shrinkage(&mut self, scale: NumT) {
        self.shrinkage *= scale;
    }

    pub fn set_bias(&mut self, bias: NumT) {
        self.bias = bias;
    }

    /// Predict and store the result as defined by `f` in `predict_buf`.
    pub fn predict_and<F>(&self, dataset: &Dataset, predict_buf: &mut [NumT], f: F)
    where F: Fn(NumT, &mut NumT) {
        let targets = dataset.target().get_raw_data();
        assert_eq!(predict_buf.len(), dataset.nexamples());
        for i in 0..dataset.nexamples() {
            let leaf_id = self.predict_leaf_id_of_example(dataset, i);
            let leaf_value = self.node_values[leaf_id];
            let prediction = self.shrinkage * (leaf_value + self.bias);
            //print!("prediction[{:4}]: {:+.4}", i, predict_buf[i]);
            f(prediction, &mut predict_buf[i]);
            //println!(" -> {:+.4} (tree_pred={:+.4} target={:+.4})", predict_buf[i],
            //    prediction, targets[i]);
        }
    }

    pub fn predict_buf(&self, dataset: &Dataset, predict_buf: &mut [NumT]) {
        self.predict_and(dataset, predict_buf, |prediction, buf_elem| {
            *buf_elem = prediction;
        });
    }

    pub fn predict(&self, dataset: &Dataset) -> Vec<NumT> {
        let mut predictions = vec![0.0; dataset.nexamples()];
        self.predict_buf(dataset, &mut predictions);
        predictions
    }
}


impl Debug for Tree {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let mut stack = vec![(0, 0)];
        while !stack.is_empty() {
            let (node_id, depth) = stack.pop().unwrap();

            let indent: String = std::iter::repeat("   ").take(depth).collect();
            write!(f, "{}[{:<3}] val {:.5}", indent, node_id, self.node_values[node_id])?;

            match self.split_crits[node_id] {
                SplitCrit::EqTest(feat_id, split_value) => {
                    let left = self.left_child(node_id);
                    let right = self.right_child(node_id);
                    writeln!(f, " eq.test F{:02}=={}", feat_id, split_value)?;
                    stack.push((right, depth+1));
                    stack.push((left, depth+1));
                },
                SplitCrit::Undefined => {
                    writeln!(f, " leaf")?;
                }
            }
        }
        Ok(())
    }
}




// ------------------------------------------------------------------------------------------------

pub struct AdditiveTree {
    trees: Vec<Tree>,
}

impl AdditiveTree {
    pub fn new() -> AdditiveTree {
        AdditiveTree {
            trees: Vec::new(),
        }
    }

    pub fn push_tree(&mut self, tree: Tree) {
        self.trees.push(tree);
    }

    pub fn predict(&self, dataset: &Dataset) -> Vec<NumT> {
        let nexamples = dataset.nexamples();
        let mut accum = vec![0.0; nexamples];
        for tree in &self.trees {
            tree.predict_and(dataset, &mut accum, |prediction, accum| {
                *accum += prediction;
            });
        }
        accum
    }
}








// ------------------------------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use crate::tree::Tree;

    #[test]
    fn test_tree() {
        let tree = Tree::new(3);
        assert_eq!(tree.max_depth(), 3);
        assert_eq!(tree.max_nleaves(), 8);
        assert_eq!(tree.max_ninternal(), 7);
        assert_eq!(tree.max_nnodes(), 8+7);
    }
}
