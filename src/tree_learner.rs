use std::ops::Sub;

use crate::NumT;
use crate::config::Config;
use crate::dataset::{Dataset, FeatureType};
use crate::tree::{Tree, SplitCrit};
use crate::slice_store::{SliceRange, HistStore, BitBlockStore, BitVecRef};
use crate::slice_store::{BitSliceLayout, BitSliceLayout1, BitSliceLayout2, BitSliceLayout4};
use crate::objective::Objective;

// ------------------------------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Copy)]
struct HistVal {
    grad_sum: NumT,
    example_count: u32,
}

impl HistVal {
    fn unpack(self) -> (NumT, u32) {
        (self.grad_sum, self.example_count)
    }
}

impl Sub for HistVal {
    type Output = HistVal;
    fn sub(self, other: HistVal) -> HistVal {
        HistVal {
            grad_sum: self.grad_sum - other.grad_sum,
            example_count: self.example_count - other.example_count,
        }
    }
}






// ------------------------------------------------------------------------------------------------

/// Contains information about the current node to split.
struct Node2Split {
    node_id: usize,
    example_count: u32,
    grad_sum: NumT,
    compressed: bool,

    // information about the examples that sort to this node:
    // - histogram for each feature
    // - indices of non-zero masks (non-consecutive if compressed==true)
    // - non-zero masks
    // - gradient values for each non-zero block
    hists_range: SliceRange,
    idx_range: SliceRange,
    mask_range: SliceRange,
    grad_range: SliceRange,
}

impl Node2Split {
    fn new(node_id: usize, hist_store: &mut HistStore<HistVal>) -> Node2Split {
        Node2Split {
            node_id,
            example_count: 0,
            grad_sum: 0.0,
            compressed: false,
            hists_range: hist_store.alloc_hists(),
            idx_range: (0, 0),
            mask_range: (0, 0),
            grad_range: (0, 0),
        }
    }
}

struct Split {
    split_crit: SplitCrit, // we use SplitCrit::NoSplit to indicate that no split is possible
    left_grad_sum: NumT,
    right_grad_sum: NumT,
    left_example_count: u32,
    right_example_count: u32,
}

impl Split {
    fn no_split() -> Split {
        Split {
            split_crit: SplitCrit::no_split(),
            left_grad_sum: 0.0,
            right_grad_sum: 0.0,
            left_example_count: 0,
            right_example_count: 0,
        }
    }
}






// ------------------------------------------------------------------------------------------------

/// Resources used by the tree learner that can be reused.
pub struct TreeLearnerContext<'a> {
    config: &'a Config,
    dataset: &'a Dataset,

    hist_store: HistStore<HistVal>,
    idx_store: BitBlockStore,
    mask_store: BitBlockStore,
    grad_store: BitBlockStore,

    n2s_stack: Vec<Node2Split>,
}

impl <'a> TreeLearnerContext<'a> {
    pub fn new(config: &'a Config, dataset: &'a Dataset) -> Self {
        TreeLearnerContext {
            config,
            dataset,

            hist_store: HistStore::for_dataset(dataset),
            idx_store: BitBlockStore::new(4024),
            mask_store: BitBlockStore::new(4024),
            grad_store: BitBlockStore::new(4024 * config.discr_nbits),

            n2s_stack: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.hist_store.reset();
        self.idx_store.reset();
        self.mask_store.reset();
        self.grad_store.reset();
        self.n2s_stack.clear();
    }
}

/// Helper macro for calling code that is specialized for each value of `discr_nbits`. We rely on
/// constant propagation for each BitSliceLayout, and rustc stops specializing methods pretty
/// quickly. That's why we force it by using macros.
///
/// This is probably not the fastest way to do dynamic dispatch, but the amount of work done by
/// each of the specialized functions overshadows the cost of the function selection.
macro_rules! dispatch {
    ($self:ident, $fun_w1:ident, $fun_w2:ident, $fun_w4:ident) => {
        dispatch!($self, $fun_w1, $fun_w2, $fun_w4; )
    };
    ($self:ident, $fun_w1:ident, $fun_w2:ident, $fun_w4:ident; $( $args:expr ),*) => {{
        let discr_nbits = $self.ctx.config.discr_nbits;
        match discr_nbits {
            1 => $fun_w1($self, $( $args )*),
            2 => $fun_w2($self, $( $args )*),
            4 => $fun_w4($self, $( $args )*),
            _ => panic!("invalid discr_width"),
        }
    }}
}







// ------------------------------------------------------------------------------------------------

pub struct TreeLearner<'a, 'b>
where 'a: 'b { // a lives longer than b
    ctx: &'b mut TreeLearnerContext<'a>,
    objective: &'b dyn Objective,
    tree: Tree,
}

impl <'a, 'b> TreeLearner<'a, 'b>
where 'a: 'b {
    pub fn new(ctx: &'b mut TreeLearnerContext<'a>, objective: &'b dyn Objective)
        -> TreeLearner<'a, 'b>
    {
        ctx.reset();
        let tree = Tree::new(ctx.config.max_tree_depth);
        TreeLearner {
            ctx,
            objective,
            tree,
        }
    }

    pub fn train(mut self) -> Tree {
        let root_n2s = self.get_root_n2s();
        self.ctx.n2s_stack.push(root_n2s);

        while let Some(n2s) = self.ctx.n2s_stack.pop() {
            let node_id = n2s.node_id;

            let split = self.find_best_split(&n2s);
            if split.split_crit.is_no_split() { continue; }

            // If the children aren't at maximal depth, push to split later.
            // Else, the children are leafs: generate leaf values.
            if !self.tree.is_max_leaf_node(self.tree.left_child(node_id)) {
                let (left_n2s, right_n2s) = self.get_left_right_n2s(&n2s, &split);
                self.ctx.n2s_stack.push(right_n2s);
                self.ctx.n2s_stack.push(left_n2s);
            } else {
                unimplemented!()
            }

            // Split tree and free some memory (we can't free idxs/grads, might be shared).
            self.tree.split_node(node_id, split.split_crit);
            self.ctx.hist_store.free_hists(n2s.hists_range);
            self.ctx.mask_store.free_blocks(n2s.mask_range);
        }
        
        assert!(self.tree.nnodes() > 1);
        self.tree
    }

    fn find_best_split(&mut self, n2s: &Node2Split) -> Split {
        let mut best_split = Split::no_split();
        let mut best_gain = self.ctx.config.min_gain;
        let min_examples = self.ctx.config.min_examples_leaf;

        let (pgrad, pcount) = (n2s.grad_sum, n2s.example_count);
        let ploss = self.get_loss(pgrad, pcount);

        // Compute best split based on each feature's histogram.
        for feat_id in 0..self.ctx.dataset.nfeatures() {
            let hist = self.ctx.hist_store.get_hist(n2s.hists_range, feat_id);

            for fval_id in 0..hist.len() {
                let (lgrad, lcount) = hist[fval_id].unpack();
                let (rgrad, rcount) = (pgrad - lgrad, pcount - lcount);

                if lcount < min_examples || rcount < min_examples { continue; }

                let lloss = self.get_loss(lgrad, lcount);
                let rloss = self.get_loss(rgrad, rcount);
                let gain = ploss - lloss - rloss;
                let feat = self.ctx.dataset.get_feature(feat_id);

                if gain > best_gain {
                    best_gain = gain;
                    best_split.split_crit = SplitCrit::for_feature(feat, fval_id);
                    best_split.left_grad_sum = lgrad;
                    best_split.left_example_count = lcount;
                    best_split.right_grad_sum = rgrad;
                    best_split.right_example_count = rcount;
                }
            }
        }

        best_split
    }

    fn get_root_n2s(&mut self) -> Node2Split {
        dispatch!(self, get_root_n2s_w1, get_root_n2s_w2, get_root_n2s_w4)
    }

    fn get_left_right_n2s(&mut self, parent_n2s: &Node2Split, split: &Split)
        -> (Node2Split, Node2Split)
    {
        // TODO continue here
        // + find fval_mask:
        //    - cat. split -> just get from dataset
        //    - num. split -> generate in some buffer (THIS IS NEW!)
        // + then split_examples(parent_n2s, left_id, &fval_mask, |m| m) + compress (same cat/num)
        // + build histograms
        // + use sum grad and count from parent hist
        //
        // + split examples right
        // + derive histograms
        //
        // Is the grad_sum we need in split.left_grad_sum, split.left_example_count?

        unimplemented!()
    }

    fn split_examples<F>(&mut self, parent_n2s: &Node2Split, child_id: usize,
                         fval_mask: &BitVecRef, f: F) -> Node2Split
    where F: Fn(u32) -> u32
    {
        // TODO continue here - copy, should be mostly the same
        // compress examples: add macro
        unimplemented!()
    }
    
    fn build_histograms(&mut self, n2s: &Node2Split) {
        dispatch!(self, build_histograms_w1, build_histograms_w2, build_histograms_w4; n2s)
    }

    fn derive_histograms(&mut self, parent_n2s: &Node2Split, left_n2s: &Node2Split,
                         right_n2s: &Node2Split)
    {
        self.ctx.hist_store.hists_subtract(parent_n2s.hists_range, left_n2s.hists_range,
                                           right_n2s.hists_range);
    }

    fn get_loss(&self, grad_sum: NumT, example_count: u32) -> NumT {
        let lambda = self.ctx.config.reg_lambda;
        -0.5 * ((grad_sum * grad_sum) / (example_count as NumT + lambda))
    }
}







// ------------------------------------------------------------------------------------------------

macro_rules! get_root_n2s {
    ($f:ident, $bsl:ident, $hist_fun:ident) => {
        fn $f(this: &mut TreeLearner) -> Node2Split {
            let bounds = this.objective.get_bounds();
            let nexamples  = this.ctx.dataset.nexamples();
            let grad_range = this.ctx.grad_store.alloc_zero_bitslice::<$bsl>(nexamples);
            let mask_range = this.ctx.mask_store.alloc_one_bits(nexamples);
            let nblocks    = this.ctx.mask_store.get_bitvec(mask_range).block_len::<u32>();
            let idx_range  = this.ctx.idx_store.alloc_from_iter::<u32, _>(nblocks, 0..nblocks as u32);

            // put gradients in bitslice
            let mut grad_sum = 0;
            let mut grad_slice = this.ctx.grad_store.get_bitslice_mut::<$bsl>(grad_range);
            for (i, &v) in this.objective.gradients().iter().enumerate() {
                let x = grad_slice.set_scaled_value(i, v, bounds);
                grad_sum += x as u64;
            }

            assert!(nexamples < u32::max_value() as usize);

            let mut n2s = Node2Split::new(0, &mut this.ctx.hist_store);
            n2s.idx_range     = idx_range;
            n2s.mask_range    = mask_range;
            n2s.grad_range    = grad_range;
            n2s.example_count = nexamples as u32;
            n2s.grad_sum      = $bsl::linproj(grad_sum as NumT, nexamples as NumT, bounds);

            // build histograms
            $hist_fun(this, &n2s);

            n2s
        }
    }
}

get_root_n2s!(get_root_n2s_w1, BitSliceLayout1, build_histograms_w1);
get_root_n2s!(get_root_n2s_w2, BitSliceLayout2, build_histograms_w2);
get_root_n2s!(get_root_n2s_w4, BitSliceLayout4, build_histograms_w4);

macro_rules! build_histograms {
    ($f:ident, $bsl:ident, $sum_method:ident) => {
        fn $f(this: &mut TreeLearner, n2s: &Node2Split) {
            get_grad_sum!(get_grad_sum, $bsl, $sum_method);

            // TODO continue here
            //
            // for cat. feat.: loop over all values
            // for num. feat.: loop over all buckets -> generate buckets?
            unimplemented!()
        }
    }
}

macro_rules! get_grad_sum {
    ($f:ident, $bsl:ident, simd) => {
        fn $f(this: &TreeLearner, n2s: &Node2Split, fval_mask: &BitVecRef) -> (NumT, NumT) {
            // TODO copy code, should be exactly the same
            unimplemented!()
        }
    };
    ($f:ident, $bsl:ident, simd) => {
        fn $f(this: &TreeLearner, n2s: &Node2Split, fval_mask: &BitVecRef) -> (NumT, NumT) {
            // TODO copy code, should be exactly the same
            unimplemented!()
        }
    }
}

build_histograms!(build_histograms_w1, BitSliceLayout1, simd);
build_histograms!(build_histograms_w2, BitSliceLayout2, simd);
build_histograms!(build_histograms_w4, BitSliceLayout4, simd);

