use dataset::NumericalType;

pub enum SplitCrit {
    /// Corresponds to `FeatureData::BitSets`. Everything equal goes left, rest goes right.
    BitSetsEq { value_index: usize }
}

/// Tree structure. This is the full tree, not an individual node (we don't use the recursive
/// definition).
/// Indexation:
///  - left node of i:      2i + 1
///  - right node of i:     2i + 2
///  - parent of node i:    floor((i-1)/2)
pub struct Tree {
    max_depth: usize,
    split_crit: Vec<SplitCrit>,
    leaf_values: Vec<NumericalType>,
}

impl Tree {
    pub fn new(max_depth: usize) -> Tree {
        Tree {
            max_depth: max_depth,
            split_crit: Vec::with_capacity(1 << (max_depth-1)),
            leaf_values: Vec::with_capacity(1 << (max_depth-1)),
        }
    }
}
