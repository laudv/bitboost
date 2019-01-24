use std::ops::Sub;
use std::mem::size_of;
use std::collections::HashSet;

use log::debug;

use crate::NumT;
use crate::config::Config;
use crate::dataset::{Dataset, Feature, FeatureType};
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
    fval_id: usize,
}

impl Split {
    fn no_split() -> Split {
        Split {
            split_crit: SplitCrit::no_split(),
            fval_id: 0,
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
    example_buffer: Vec<usize>,
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
            example_buffer: Vec::new(), // used for leaf value predictions
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
    ($self:expr, $fun_w1:ident, $fun_w2:ident, $fun_w4:ident) => {
        dispatch!($self, $fun_w1, $fun_w2, $fun_w4; )
    };
    ($self:expr, $fun_w1:ident, $fun_w2:ident, $fun_w4:ident; $( $args:expr ),*) => {{
        let discr_nbits = $self.ctx.config.discr_nbits;
        match discr_nbits {
            1 => $fun_w1($self, $( $args, )*),
            2 => $fun_w2($self, $( $args, )*),
            4 => $fun_w4($self, $( $args, )*),
            _ => panic!("invalid discr_width"),
        }
    }}
}







// ------------------------------------------------------------------------------------------------

pub struct TreeLearner<'a, 'b>
where 'a: 'b { // a lives longer than b
    ctx: &'b mut TreeLearnerContext<'a>,
    objective: &'b mut dyn Objective,
    tree: Tree,
}

impl <'a, 'b> TreeLearner<'a, 'b>
where 'a: 'b {
    pub fn new(ctx: &'b mut TreeLearnerContext<'a>, objective: &'b mut dyn Objective)
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
            if split.split_crit.is_no_split() {
                // predict leaf value, easy case: we have example masks
                self.predict_leaf_value(&n2s);
                continue;
            }

            debug_assert!(self.debug_print(&n2s, &split));

            // If the children aren't at maximal depth, push to split later.
            // Else, the children are leafs: generate leaf values.
            if !self.tree.is_max_leaf_node(self.tree.left_child(node_id)) {
                let (left_n2s, right_n2s) = self.get_left_right_n2s(&n2s, &split);
                self.ctx.n2s_stack.push(right_n2s);
                self.ctx.n2s_stack.push(left_n2s);
            } else {
                // predict leaf values, hard case: we don't have example masks for children.
                self.predict_child_leaf_values(&n2s, &split);
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

                //let better = if gain > best_gain { "better" } else { "" };
                //debug!("N{:03}-F{:02} possible split gain={} {}",
                //       n2s.node_id, feat_id, gain, better);

                if gain > best_gain {
                    best_gain = gain;
                    best_split.split_crit = SplitCrit::for_feature(feat, fval_id);
                    best_split.fval_id = fval_id;
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
        let feat_id = split.split_crit.feature_id;
        let parent_hist = self.ctx.hist_store.get_hist(parent_n2s.hists_range, feat_id);
        let (pgrad, pcount) = (parent_n2s.grad_sum, parent_n2s.example_count);
        let (lgrad, lcount) = parent_hist[split.fval_id].unpack();
        let (rgrad, rcount) = (pgrad - lgrad, pcount - lcount);
        
        let feature = self.ctx.dataset.get_feature(feat_id);
        let fval_mask = self.get_fval_mask(feature, split.fval_id);
        let left_id = self.tree.left_child(parent_n2s.node_id);
        let right_id = self.tree.right_child(parent_n2s.node_id);

        let mut left_n2s = self.split_examples(parent_n2s, left_id, &fval_mask, |m| m);
        left_n2s.grad_sum = lgrad;
        left_n2s.example_count = lcount;
        self.build_histograms(&left_n2s);

        let mut right_n2s = self.split_examples(parent_n2s, right_id, &fval_mask, |m| !m);
        right_n2s.grad_sum = rgrad;
        right_n2s.example_count = rcount;
        self.derive_histograms(parent_n2s, &left_n2s, &right_n2s);

        (left_n2s, right_n2s)
    }

    fn get_fval_mask<'c>(&self, feature: &'c Feature, fval_id: usize) -> BitVecRef<'c> {
        match feature.get_feature_type() {
            FeatureType::NomCat(bitvecs) => bitvecs.get_bitvec(fval_id),
            FeatureType::OrdCat(bitvecs) => bitvecs.get_bitvec(fval_id),
            FeatureType::OrdNum(splitter) => splitter.get_bitvec(fval_id),
            FeatureType::Uninitialized => panic!("uninitialized split"),
        }
    }

    fn split_examples<F>(&mut self, parent_n2s: &Node2Split, child_id: usize,
                         fval_mask: &BitVecRef, f: F) -> Node2Split
    where F: Fn(u32) -> u32
    {
        let parent_indices = self.ctx.idx_store.get_bitvec(parent_n2s.idx_range);
        let nblocks = parent_indices.len();
        let n_u32 = parent_indices.block_len::<u32>();

        let mut child_n2s = Node2Split::new(child_id, &mut self.ctx.hist_store);
        child_n2s.compressed = parent_n2s.compressed; // child is compr if parent is compr
        child_n2s.idx_range = parent_n2s.idx_range;
        child_n2s.grad_range = parent_n2s.grad_range;
        child_n2s.mask_range = self.ctx.mask_store.alloc_zero_blocks(nblocks);

        let (parent_mask, mut child_mask) = self.ctx.mask_store.get_two_bitvecs_mut(
            parent_n2s.mask_range,
            child_n2s.mask_range);

        let mut zero_count = 0;
        for j in 0..n_u32 {
            let i = *parent_indices.get::<u32>(j); // j is node index (compr), i is global index
            let pmask = *parent_mask.get::<u32>(j);
            let fmask = *fval_mask.get::<u32>(i as usize);
            let mask = pmask & f(fmask);

            if mask == 0 { zero_count += 1; }

            child_mask.set(j, mask);
        }

        // We encountered enough zero blocks to apply compression
        let zero_ratio = zero_count as NumT / n_u32 as NumT;
        if zero_ratio > self.ctx.config.compression_threshold {
            debug!("N{:03}->N{:03} applying compression, #zero = {}/{} (ratio={:.2})",
                   parent_n2s.node_id, child_n2s.node_id, zero_count, n_u32, zero_ratio);

            let n_u32_child = n_u32 - zero_count; // number of non-zero blocks in child
            self.compress_examples(parent_n2s, &mut child_n2s, n_u32_child);
        }
        child_n2s
    }

    fn compress_examples(&mut self, parent_n2s: &Node2Split, child_n2s: &mut Node2Split,
                         n_u32_child: usize) {
        dispatch!(self, compr_examples_w1, compr_examples_w2, compr_examples_w4;
                  parent_n2s, child_n2s, n_u32_child)
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

    /// Use only the discretized gradient values to find leaf value, rather than letting the
    /// objective function provide a slower/more accurate leaf value. Probably only makes sense for
    /// L2.
    fn get_best_leaf_value(&self, grad_sum: NumT, example_count: u32) -> NumT {
        let lambda = self.ctx.config.reg_lambda;
        -grad_sum / (example_count as NumT + lambda)
    }

    fn predict_leaf_value(&mut self, n2s: &Node2Split) {
        let value;

        if self.ctx.config.optimize_leaf_values {
            let targets = self.ctx.dataset.target().get_raw_data();
            let mask = self.ctx.mask_store.get_bitvec(n2s.mask_range);
            let examples = &mut self.ctx.example_buffer;
            examples.clear();

            if n2s.compressed {
                let idxs = self.ctx.idx_store.get_bitvec(n2s.idx_range);
                examples.extend(mask.index_iter_compr(&idxs));
            } else {
                examples.extend(mask.index_iter());
            }

            value = self.objective.predict_leaf_value(targets, examples);
        } else {
            value = self.get_best_leaf_value(n2s.grad_sum, n2s.example_count);
        }

        self.tree.set_value(n2s.node_id, value);

        debug!("N{:03} leaf value {} (no more splits)", n2s.node_id, value);
    }

    fn predict_child_leaf_values(&mut self, n2s: &Node2Split, split: &Split) {
        let feature = self.ctx.dataset.get_feature(split.split_crit.feature_id);
        let left_value;
        let right_value;

        if self.ctx.config.optimize_leaf_values {
            let targets = self.ctx.dataset.target().get_raw_data();
            let pmask = self.ctx.mask_store.get_bitvec(n2s.mask_range);
            let fmask = self.get_fval_mask(feature, split.fval_id);
            let examples = &mut self.ctx.example_buffer;

            examples.clear();
            if n2s.compressed {
                let idxs = self.ctx.idx_store.get_bitvec(n2s.idx_range);
                examples.extend(pmask.index_iter_and_compr(&fmask, &idxs));
            } else {
                examples.extend(pmask.index_iter_and(&fmask));
            }
            left_value = self.objective.predict_leaf_value(targets, examples);

            examples.clear();
            if n2s.compressed {
                let idxs = self.ctx.idx_store.get_bitvec(n2s.idx_range);
                examples.extend(pmask.index_iter_andnot_compr(&fmask, &idxs));
            } else {
                examples.extend(pmask.index_iter_andnot(&fmask));
            }
            right_value = self.objective.predict_leaf_value(targets, examples);
        } else {
            let hist = self.ctx.hist_store.get_hist(n2s.hists_range, feature.id());
            let (pgrad, pcount) = (n2s.grad_sum, n2s.example_count);
            let (lgrad, lcount) = hist[split.fval_id].unpack();
            let (rgrad, rcount) = (pgrad - lgrad, pcount - lcount);
            left_value = self.get_best_leaf_value(lgrad, lcount);
            right_value = self.get_best_leaf_value(rgrad, rcount);
        }

        let left_id = self.tree.left_child(n2s.node_id);
        let right_id = self.tree.right_child(n2s.node_id);
        self.tree.set_value(left_id, left_value);
        self.tree.set_value(right_id, right_value);

        debug!("N{:03} leaf value {} (max leaf)", left_id, left_value);
        debug!("N{:03} leaf value {} (max leaf)", right_id, right_value);
    }

    fn debug_print(&self, n2s: &Node2Split, split: &Split) -> bool {
        match self.ctx.config.discr_nbits {
            1 => self.debug_print_bsl::<BitSliceLayout1>(n2s, split),
            2 => self.debug_print_bsl::<BitSliceLayout2>(n2s, split),
            4 => self.debug_print_bsl::<BitSliceLayout4>(n2s, split),
            _ => panic!(),
        }
    }

    fn debug_print_bsl<BSL>(&self, n2s: &Node2Split, split: &Split) -> bool
    where BSL: BitSliceLayout {
        let nid = n2s.node_id;
        let feat_id = split.split_crit.feature_id;
        let split_val = split.split_crit.split_value;
        let fval_id = split.fval_id;
        let feature = self.ctx.dataset.get_feature(feat_id);
        let hist = self.ctx.hist_store.get_hist(n2s.hists_range, feat_id);
        let (pgrad, pcount) = (n2s.grad_sum, n2s.example_count);
        let (lgrad, lcount) = hist[fval_id].unpack();
        let (rgrad, rcount) = (pgrad - lgrad, pcount - lcount);
        let gain = self.get_loss(pgrad, pcount)
            - self.get_loss(lgrad, lcount)
            - self.get_loss(rgrad, rcount);

        let pidxs = self.ctx.idx_store.get_bitvec(n2s.idx_range);
        let pmask = self.ctx.mask_store.get_bitvec(n2s.mask_range);
        let (op, fval_mask) = match feature.get_feature_type() {
            FeatureType::NomCat(bitsets) => {
                ("=", bitsets.get_bitvec(fval_id))
            },
            _ => {
                unimplemented!("debug numeric")
            }
        };

        debug!("N{:03}-F{:02} {} {:.3} [fval_id={:<2}] gain={}; counts: {} -> {}, {}", nid,
               feat_id, op, split_val, fval_id, gain, pcount, lcount, rcount);

        println!("LEFT N{:03}->N{:03}:", nid, self.tree.left_child(nid));
        for i in pmask.index_iter_and_compr(&fval_mask, &pidxs) {
            println!(" - {:4}: {}", i, self.objective.gradients()[i]);
        }

        println!("RIGHT N{:03}->N{:03}:", nid, self.tree.right_child(nid));
        for i in pmask.index_iter_andnot_compr(&fval_mask, &pidxs) {
            println!(" - {:4}: {}", i, self.objective.gradients()[i]);
        }

        true
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

macro_rules! compress_examples {
    ($f:ident, $bsl:ident) => {
        fn $f(this: &mut TreeLearner, parent_n2s: &Node2Split, child_n2s: &mut Node2Split,
              n_u32_child: usize)
        {
            child_n2s.compressed = true;

            // Allocate space for indices, gradients for child (reuse masks from `split_examples`)
            let nvalues = n_u32_child * size_of::<u32>() * 8;
            child_n2s.idx_range = this.ctx.idx_store.alloc_zero_bits(nvalues);
            child_n2s.grad_range = this.ctx.grad_store.alloc_zero_bitslice::<$bsl>(nvalues);

            // Get references to the structures
            let (parent_indices, mut child_indices) = this.ctx.idx_store.get_two_bitvecs_mut(
                parent_n2s.idx_range,
                child_n2s.idx_range);
            let n_u32_parent = parent_indices.block_len::<u32>();
            let mut child_mask = this.ctx.mask_store.get_bitvec_mut(child_n2s.mask_range);
            let (parent_grads, mut child_grads) = this.ctx.grad_store.get_two_bitslices_mut::<$bsl>(
                parent_n2s.grad_range,
                child_n2s.grad_range);

            let mut k = 0; // child node index (compr)
            for j in 0..n_u32_parent {
                debug_assert!(j >= k);

                let i = *parent_indices.get::<u32>(j); // j is parent index (compr), i is global index

                // in `split_examples`, we stored all zeros and used the parent indices `j`; copy this
                // to the smaller child indices `k` by skipping zero blocks:
                let mask = *child_mask.get::<u32>(j);
                if mask == 0 { continue; }

                child_mask.set::<u32>(k, mask);
                child_indices.set::<u32>(k, i);
                child_grads.copy_block_from::<u32, _>(&parent_grads, j, k); // from=j, to=k
                k += 1;
            }

            debug_assert_eq!(k, n_u32_child);

            // Zero the remaining blocks of child_mask: not doing this breaks things!
            for j in n_u32_child..n_u32_parent {
                child_mask.set::<u32>(j, 0);
            }

            // Resize the masks; we use the same memory allocated in split_examples, but resize the
            // range such that the compressed zero blocks at the end are no longer included
            let idx_len = child_n2s.idx_range.1 - child_n2s.idx_range.0;
            child_n2s.mask_range = (child_n2s.mask_range.0, child_n2s.mask_range.0 + idx_len);
        }
    }
}

compress_examples!(compr_examples_w1, BitSliceLayout1);
compress_examples!(compr_examples_w2, BitSliceLayout2);
compress_examples!(compr_examples_w4, BitSliceLayout4);

macro_rules! build_histograms {
    ($f:ident, $bsl:ident, $sum_method:ident) => {
        fn $f(this: &mut TreeLearner, n2s: &Node2Split) {
            get_grad_sum!(get_grad_sum, $bsl, $sum_method);

            for feature in this.ctx.dataset.features() {
                let feat_id = feature.id();
                let nbuckets = feature.get_nbuckets();

                match feature.get_feature_type() {
                    // bucket for each categorical feature value
                    FeatureType::NomCat(ref bitvecs) | FeatureType::OrdCat(ref bitvecs) => {
                        for fval_id in 0..nbuckets {
                            let fval_mask = bitvecs.get_bitvec(fval_id);
                            let (grad_sum, example_count) = get_grad_sum(this, n2s, &fval_mask);
                            let histval = HistVal { grad_sum, example_count };
                            let hist = this.ctx.hist_store.get_hist_mut(n2s.hists_range, feat_id);
                            hist[fval_id] = histval;
                        }
                    }

                    // bucket for each split proposed by the splitter
                    FeatureType::OrdNum(ref splitter) => {
                        unimplemented!("TODO histogram numerical feature")
                    }

                    FeatureType::Uninitialized => { panic!("uninitialized split"); }
                }
            }
        }
    }
}

macro_rules! get_grad_sum {
    ($f:ident, $bsl:ident, naive) => {
        fn $f(this: &TreeLearner, n2s: &Node2Split, fval_mask: &BitVecRef) -> (NumT, u32) {
            let bounds     = this.objective.get_bounds();
            let ems_bitset = this.ctx.mask_store.get_bitvec(n2s.mask_range);
            let idx_bitset = this.ctx.idx_store.get_bitvec(n2s.idx_range);

            let vms = fval_mask.cast::<u32>(); // feature value mask
            let ems = ems_bitset.cast::<u32>(); // example mask
            let indices = idx_bitset.cast::<u32>();
            let gradients = this.ctx.grad_store.get_bitslice::<$bsl>(n2s.grad_range);

            let mut count = 0;
            let mut sum = 0.0;

            for (i, &ju32) in indices.iter().enumerate() {
                let j = ju32 as usize;
                let vm = vms[j];
                let em = ems[i];
                let m = vm & em;

                let (mut count1, mut sum1) = (0, 0.0);

                for k in 0..32 {
                    if m >> k & 0x1 == 0x1 {
                        sum1 += gradients.get_scaled_value(i*32 + k, bounds);
                        count1 += 1;
                    }
                }

                sum += sum1;
                count += count1;
            }

            (sum, count)
        }
    };
    ($f:ident, $bsl:ident, simd) => {
        fn $f(this: &TreeLearner, n2s: &Node2Split, fval_mask: &BitVecRef) -> (NumT, u32) {
            fn sum_uc(this: &TreeLearner, n2s: &Node2Split, fval_mask: &BitVecRef) -> (NumT, u32) {
                let bounds = this.objective.get_bounds();
                let ems = this.ctx.mask_store.get_bitvec(n2s.mask_range);
                let grads = this.ctx.grad_store.get_bitslice::<$bsl>(n2s.grad_range);

                let count = ems.count_ones_and(&fval_mask) as u32;
                if count < this.ctx.config.min_examples_leaf { return (0.0, count); }
                let sum = unsafe { grads.sum_all_masked2_unsafe(&ems, &fval_mask) };
                let sum = $bsl::linproj(sum as NumT, count as NumT, bounds);
                (sum, count)
            }

            fn sum_c(this: &TreeLearner, n2s: &Node2Split, fval_mask: &BitVecRef) -> (NumT, u32) {
                let bounds = this.objective.get_bounds();
                let ems = this.ctx.mask_store.get_bitvec(n2s.mask_range);
                let grads = this.ctx.grad_store.get_bitslice::<$bsl>(n2s.grad_range);
                let indices = this.ctx.idx_store.get_bitvec(n2s.idx_range);

                let count = unsafe { ems.count_ones_and_compr_unsafe(&indices, &fval_mask) };
                let count = count as u32;
                if count < this.ctx.config.min_examples_leaf { return (0.0, count); }

                //if hess < min_hess || (n2s.hess_sum - hess) < min_hess { return (0.0, 0.0); }
                let sum = unsafe { grads.sum_all_masked2_compr_unsafe(&indices, &ems, &fval_mask) };
                let sum = $bsl::linproj(sum as NumT, count as NumT, bounds);
                (sum, count)
            }

            if !n2s.compressed { sum_uc(this, n2s, fval_mask) }
            else               { sum_c(this, n2s, fval_mask)  }
        }
    }
}

build_histograms!(build_histograms_w1, BitSliceLayout1, simd);
build_histograms!(build_histograms_w2, BitSliceLayout2, simd);
build_histograms!(build_histograms_w4, BitSliceLayout4, simd);
