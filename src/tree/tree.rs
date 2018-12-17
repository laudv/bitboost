use dataset::{NumericalType, NominalType};

#[derive(Debug, Clone, PartialEq)]
pub enum SplitCrit {
    Undefined,

    /// Equality test: equal goes left, others go right.
    EqTest(NominalType),

    /// Ordinal test: less than goes left, others go right.
    LtTest(NumericalType),
}

//impl Eq for SplitCrit {}

/// Tree structure. This is the full tree, not an individual node (we don't use the recursive
/// definition).
/// Indexation:
///  - left node of i:      2i + 1
///  - right node of i:     2i + 2
///  - parent of node i:    floor((i-1)/2)
pub struct Tree {
    max_depth: usize,
    split_crits: Vec<SplitCrit>,
    node_values: Vec<NumericalType>,
}

impl Tree {
    pub fn new(max_depth: usize) -> Tree {
        assert!(max_depth > 0);
        let mut tree = Tree {
            max_depth: max_depth,
            split_crits: Vec::new(),
            node_values: Vec::new(),
        };

        let ninternal = tree.ninternal();
        let nnodes = tree.nnodes();

        tree.split_crits.resize(ninternal, SplitCrit::Undefined);
        tree.node_values.resize(nnodes, 0.0);

        tree
    }

    pub fn left_child(&self, node_id: usize) -> usize { 2 * node_id + 1 }
    pub fn right_child(&self, node_id: usize) -> usize { 2 * node_id + 2 }
    pub fn parent(&self, node_id: usize) -> usize { (node_id - 1) / 2 }
    pub fn is_node(&self, node_id: usize) -> bool { node_id < self.nnodes() }
    pub fn nleaves(&self) -> usize { 1 << self.max_depth }
    pub fn ninternal(&self) -> usize { (1 << self.max_depth) - 1 }
    pub fn nnodes(&self) -> usize { (1 << (self.max_depth+1)) - 1 }
    pub fn get_max_depth(&self) -> usize { self.max_depth }
    pub fn set_value(&mut self, node_id: usize, value: NumericalType) {
        self.node_values[node_id] = value;
    }
    pub fn is_leaf_node(&self, node_id: usize) -> bool {
        node_id >= self.ninternal()
    }

    pub fn split_node(&mut self, node_id: usize, split_crit: SplitCrit, left_value: NumericalType,
                      right_value: NumericalType) {
        debug_assert!(self.split_crits[node_id] == SplitCrit::Undefined);

        let left_id = self.left_child(node_id);
        let right_id = self.right_child(node_id);

        self.split_crits[node_id] = split_crit;
        self.node_values[left_id] = left_value;
        self.node_values[right_id] = right_value;
    }
}



#[cfg(test)]
mod test {
    use tree::Tree;

    #[test]
    fn test_tree() {
        let tree = Tree::new(3);
        assert_eq!(tree.get_max_depth(), 3);
        assert_eq!(tree.nleaves(), 8);
        assert_eq!(tree.ninternal(), 7);
        assert_eq!(tree.nnodes(), 8+7);
    }
}
