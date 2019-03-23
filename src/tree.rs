use std::fmt::{Debug, Formatter, Result as FmtResult};

use crate::{NumT, CatT, into_cat};
use crate::data::Data;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitType {
    /// No split, this node is a leaf.
    NoSplit,
    LoCardCatEq,
    HiCardCatLt,
    NumLt,
}

#[derive(Debug, Clone)]
pub struct SplitCrit {
    pub split_type: SplitType,
    pub feature_id: usize,
    pub split_value: NumT,
}

impl SplitCrit {
    pub fn no_split() -> SplitCrit {
        SplitCrit {
            split_type: SplitType::NoSplit,
            feature_id: 0,
            split_value: 0.0,
        }
    }

    pub fn is_no_split(&self) -> bool {
        self.split_type == SplitType::NoSplit
    }
}

pub struct Tree {
    ninternal: usize,
    max_depth: usize,
    split_crits: Vec<SplitCrit>,
    node_values: Vec<NumT>,

    shrinkage: NumT,
    bias: NumT,

    supercats: Vec<Vec<CatT>>,
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

            supercats: Vec::new(),
        };
        let nnodes = tree.max_nnodes();

        tree.split_crits.resize(nnodes, SplitCrit::no_split());
        tree.node_values.resize(nnodes, 0.0);

        tree
    }

    pub fn is_valid_node_id(&self, node_id: usize) -> bool { node_id < self.nnodes() }
    pub fn left_child(&self, node_id: usize) -> usize { 2 * node_id + 1 }
    pub fn right_child(&self, node_id: usize) -> usize { 2 * node_id + 2 }
    pub fn parent(&self, node_id: usize) -> usize { (node_id - 1) / 2 }

    pub fn max_ninternal(&self) -> usize { (1 << self.max_depth) - 1 }
    pub fn max_nleafs(&self) -> usize { 1 << self.max_depth }
    pub fn ninternal(&self) -> usize { self.ninternal }
    pub fn nleafs(&self) -> usize { self.ninternal + 1 }
    pub fn max_nnodes(&self) -> usize { (1 << (self.max_depth+1)) - 1 }
    pub fn nnodes(&self) -> usize { self.ninternal() + self.nleafs() }

    pub fn node_value(&self, node_id: usize) -> NumT { self.node_values[node_id] }

    pub fn max_depth(&self) -> usize { self.max_depth }
    pub fn set_value(&mut self, node_id: usize, value: NumT) {
        safety_check!(value.is_finite());
        self.node_values[node_id] = value;
    }

    /// Can this node still be split? We can't grow deeper than max_depth.
    pub fn is_max_leaf_node(&self, node_id: usize) -> bool {
        node_id >= self.max_ninternal()
    }

    pub fn split_node(&mut self, node_id: usize, split_crit: SplitCrit) {
        assert!(self.split_crits[node_id].is_no_split());
        assert!(!split_crit.is_no_split());
        assert!(!self.is_max_leaf_node(node_id));

        self.split_crits[node_id] = split_crit;
        self.ninternal += 1;
    }

    fn predict_leaf_id_of_example(&self, data: &Data, i: usize) -> usize {
        let mut node_id = 0;
        loop {
            let split_crit = &self.split_crits[node_id];
            let feat_id = split_crit.feature_id;
            let split_value = split_crit.split_value;
            let value = data.get_feature(feat_id)[i];

            match split_crit.split_type {
                SplitType::LoCardCatEq => {
                    node_id = if value == split_value { self.left_child(node_id) }
                              else                    { self.right_child(node_id) }
                },
                SplitType::HiCardCatLt => {
                    debug_assert!(!self.supercats.is_empty(), "tree supercats not set?");
                    let spcat = self.supercats[feat_id][into_cat(value) as usize];
                    let split_value = into_cat(split_value);
                    node_id = if spcat <= split_value { self.left_child(node_id) }
                              else                    { self.right_child(node_id) }
                },
                SplitType::NumLt => {
                    node_id = if value < split_value { self.left_child(node_id) }
                              else                   { self.right_child(node_id) }
                }
                SplitType::NoSplit => { // this is a terminal (leaf) node
                    return node_id;
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

    pub fn set_supercats(&mut self, supercats: Vec<Vec<CatT>>) {
        debug_assert!(self.supercats.is_empty(), "setting supercats twice?");
        debug_assert!(!supercats.is_empty(), "setting empty supercats?");
        self.supercats = supercats;
    }

    pub fn predict_single(&self, data: &Data, i: usize) -> NumT {
       let leaf_id = self.predict_leaf_id_of_example(data, i);
       let leaf_value = self.node_values[leaf_id];
       self.shrinkage * (leaf_value + self.bias)
    }

    /// Predict and store the result as defined by `f` in `predict_buf`.
    pub fn predict_and<F>(&self, data: &Data, predict_buf: &mut [NumT], f: F)
    where F: Fn(NumT, &mut NumT) {
        assert_eq!(predict_buf.len(), data.nexamples());
        for i in 0..data.nexamples() {
            let prediction = self.predict_single(data, i);
            //print!("prediction[{:4}]: {:+.4}", i, predict_buf[i]);
            f(prediction, &mut predict_buf[i]);
            //println!(" -> {:+.4} (tree_pred={:+.4} target={:+.4})", predict_buf[i],
            //    prediction, targets[i]);
        }
    }

    pub fn predict_buf(&self, data: &Data, predict_buf: &mut [NumT]) {
        self.predict_and(data, predict_buf, |prediction, buf_elem| {
            *buf_elem = prediction;
        });
    }

    pub fn predict(&self, data: &Data) -> Vec<NumT> {
        let mut predictions = vec![0.0; data.nexamples()];
        self.predict_buf(data, &mut predictions);
        predictions
    }
}


impl Debug for Tree {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        //let mut stack = vec![(0, 0)];
        //while !stack.is_empty() {
        //    let (node_id, depth) = stack.pop().unwrap();

        //    let indent: String = std::iter::repeat("   ").take(depth).collect();
        //    write!(f, "{}[{:<3}] val {:.5}", indent, node_id, self.node_values[node_id])?;

        //    match self.split_crits[node_id] {
        //        SplitCrit::CatEqTest(feat_id, split_value) => {
        //            let left = self.left_child(node_id);
        //            let right = self.right_child(node_id);
        //            writeln!(f, " cat.eq.test F{:02}=={}", feat_id, split_value)?;
        //            stack.push((right, depth+1));
        //            stack.push((left, depth+1));
        //        },
        //        SplitCrit::Undefined => {
        //            writeln!(f, " leaf")?;
        //        }
        //        _ => { unimplemented!() }
        //    }
        //}
        writeln!(f, "A Tree Debug TODO")?;
        Ok(())
    }
}




// ------------------------------------------------------------------------------------------------

pub struct AdditiveTree {
    bias: NumT,
    trees: Vec<Tree>,
}

impl AdditiveTree {
    pub fn new() -> AdditiveTree {
        AdditiveTree {
            bias: 0.0,
            trees: Vec::new(),
        }
    }

    pub fn set_bias(&mut self, bias: NumT) {
        self.bias = bias;
    }

    pub fn push_tree(&mut self, tree: Tree) {
        self.trees.push(tree);
    }

    pub fn predict(&self, data: &Data) -> Vec<NumT> {
        let nexamples = data.nexamples();
        let mut accum = vec![self.bias; nexamples];
        for tree in &self.trees {
            tree.predict_and(data, &mut accum, |prediction, accum| {
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
        assert_eq!(tree.max_nleafs(), 8);
        assert_eq!(tree.max_ninternal(), 7);
        assert_eq!(tree.max_nnodes(), 8+7);
    }
}
