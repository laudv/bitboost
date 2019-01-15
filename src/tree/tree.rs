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
    node_values: Vec<NumT>
}

impl Tree {
    pub fn new(max_depth: usize) -> Tree {
        assert!(max_depth > 0);
        let mut tree = Tree {
            ninternal: 0, // the root is the only leaf
            max_depth: max_depth,
            split_crits: Vec::new(),
            node_values: Vec::new(),
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

    pub fn predict(&self, dataset: &Dataset) -> Vec<NumT> {
        let mut predictions = Vec::with_capacity(dataset.nexamples());
        for i in 0..dataset.nexamples() {
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
                        predictions.push(self.node_values[node_id]);
                        break;
                    },
                }
            }
        }
        predictions
    }

    // TODO implement 'Metric', separately from Loss
    pub fn evaluate<L>(&self, dataset: &Dataset, loss: &L) -> NumT
    where L: LossFun {
        let targets = dataset.target().get_raw_data();
        let predictions = self.predict(dataset);
        let mut l = 0.0;

        for (&y, yhat) in targets.iter().zip(predictions) {
            //debug!("{:+.4}   {:+.4} -> loss={:.4}", y, yhat, loss.eval(y, yhat));
            l += loss.eval(y, yhat);
        }

        l / dataset.nexamples() as NumT
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

    pub fn predict(&self, data: &Dataset, results: &mut Vec<NumT>) {
        unimplemented!()
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
