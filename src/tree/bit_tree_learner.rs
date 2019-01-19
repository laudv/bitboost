use std::ops::{Sub, Add};
use std::mem::size_of;

use log::debug;

use crate::NumT;
use crate::config::Config;
use crate::tree::{Tree, SplitCrit};
use crate::dataset::{Dataset, FeatureRepr};
use crate::slice_store::{SliceRange, HistStore, BitBlockStore};
use crate::slice_store::{BitVecRef};
use crate::slice_store::{BitSliceLayout, BitSliceLayout1, BitSliceLayout2, BitSliceLayout4};

struct Node2Split {
    /// The id of the node to split.
    node_id: usize,

    /// Cache grad_sum and hess_sum to compute right stats given left stats.
    grad_sum: NumT,
    hess_sum: NumT,

    compressed: bool,

    /// Range to lookup histograms in hist_store
    hists_range: SliceRange,

    /// Indices to non-zero blocks
    idx_range: SliceRange,

    /// For each non-zero block, store a mask that indicates which examples sort into this node.
    mask_range: SliceRange,

    /// Copied target values to improve memory locality
    grad_range: SliceRange,
}

impl Node2Split {
    /// Create a new n2s: allocates histogram, all other stuff is uninitialized!
    fn new(node_id: usize, hist_store: &mut HistStore<HistVal>) -> Node2Split {
        Node2Split {
            node_id: node_id,
            grad_sum: 0.0,
            hess_sum: 0.0,
            compressed: false,
            hists_range: hist_store.alloc_hists(),
            idx_range: (0, 0),
            mask_range: (0, 0),
            grad_range: (0, 0),
        }
    }
}

struct Split {
    split_crit: SplitCrit,
    left_grad_sum: NumT,
    left_hess_sum: NumT,
    right_grad_sum: NumT,
    right_hess_sum: NumT,
}






// ------------------------------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Copy)]
struct HistVal {
    grad: NumT,
    hess: NumT,
}

impl HistVal {
    fn unpack(&self) -> (NumT, NumT) { (self.grad, self.hess) }
}

impl Sub for HistVal {
    type Output = HistVal;
    fn sub(self, other: HistVal) -> HistVal {
        HistVal {
            grad: self.grad - other.grad,
            hess: self.hess - other.hess,
        }
    }
}

impl Add for HistVal {
    type Output = HistVal;
    fn add(self, other: HistVal) -> HistVal {
        HistVal {
            grad: self.grad + other.grad,
            hess: self.hess + other.hess,
        }
    }
}

pub struct LearnerResources {
    hist_store: HistStore<HistVal>,
    index_store: BitBlockStore,
    mask_store: BitBlockStore,
    gradient_store: BitBlockStore,
}

impl LearnerResources {
    pub fn new(config: &Config, data: &Dataset) -> Self {
        LearnerResources {
            hist_store: HistStore::for_dataset(data),
            index_store: BitBlockStore::new(4024),
            mask_store: BitBlockStore::new(4024),
            gradient_store: BitBlockStore::new(4024 * config.discr_nbits),
        }
    }
}






// ------------------------------------------------------------------------------------------------

pub struct TreeLearner<'a> {
    config: &'a Config,
    dataset: &'a Dataset,
    gradients: &'a [NumT],
    tree: Tree,
    grad_bounds: (NumT, NumT),

    /// Storage for histograms.
    hist_store: &'a mut HistStore<HistVal>,

    /// Storage for indices pointing to none-zero 32-bit example selection blocks.
    index_store: &'a mut BitBlockStore,

    /// Storage for example selection masks.
    mask_store: &'a mut BitBlockStore,

    /// Store for gradients
    gradient_store: &'a mut BitBlockStore,


    /// get_root_node2split function pointer
    get_root_n2s_fun: GetRootN2SFun,

    /// compress_examples function pointer
    compress_examples_fun: CompressExamplesFun,

    /// build_histograms function pointer
    build_histograms_fun: BuildHistogramsFun,
}

impl <'a> TreeLearner<'a> {
    pub fn new(config: &'a Config, data: &'a Dataset, gradients: &'a [NumT],
               grad_bounds: (NumT, NumT), r: &'a mut LearnerResources)
        -> Self
    {
        let tree = Tree::new(config.max_tree_depth);

        let discr_width = config.discr_nbits;
        let get_root_n2s_fun = get_get_root_node2split_fun(discr_width);
        let compress_examples_fun = get_compress_examples_fun(discr_width);
        let build_histograms_fun = get_build_histograms_fun(discr_width);

        TreeLearner {
            config,
            dataset: data,
            gradients,
            tree,
            grad_bounds,

            hist_store: &mut r.hist_store,
            index_store: &mut r.index_store,
            mask_store: &mut r.mask_store,
            gradient_store: &mut r.gradient_store,

            get_root_n2s_fun,
            compress_examples_fun,
            build_histograms_fun,
        }
    }

    pub fn train(&mut self) {
        assert_eq!(self.tree.ninternal(), 0, "tree already trained");

        let max_depth = self.config.max_tree_depth;
        let mut stack = Vec::<Node2Split>::with_capacity(max_depth * 2);

        let root_n2s = self.get_root_node2split();
        self.tree.set_root_value(self.get_best_value(root_n2s.grad_sum, root_n2s.hess_sum));

        stack.push(root_n2s);

        while let Some(n2s) = stack.pop() {
            let node_id = n2s.node_id;

            let split_opt = self.find_best_split(&n2s);
            if split_opt.is_none() { debug!("N{:03} no split", node_id); continue; }

            let split = split_opt.unwrap();
            let (feat_id, feat_val) = split.split_crit.unpack_eqtest().expect("not an EqTest");

            //self.debug_print_split(&n2s, &split);

            // Enqueue children to be split if they are not leaves
            if !self.tree.is_max_leaf_node(self.tree.left_child(node_id)) {
                let val_masks = self.dataset.features()[feat_id].get_bitvec(feat_val).unwrap();
                let (left_n2s, right_n2s) = self.get_left_right_node2split(&n2s, val_masks);

                // Schedule the left and right children to be split
                stack.push(right_n2s);
                stack.push(left_n2s);
            }

            // Set tree node values
            let left_value = self.get_best_value(split.left_grad_sum, split.left_hess_sum);
            let right_value = self.get_best_value(split.right_grad_sum, split.right_hess_sum);
            self.tree.split_node(node_id, split.split_crit, left_value, right_value);

            // Free histograms and masks for node that was just split
            // We can't free indices/gradients, as they can be shared between nodes
            self.hist_store.free_hists(n2s.hists_range);
            self.mask_store.free_blocks(n2s.mask_range);
        }
    }

    fn find_best_split(&mut self, n2s: &Node2Split) -> Option<Split> {
        let mut best_gain = self.config.min_gain;
        let mut best_split = None;

        //print!("N{:03}: ", n2s.node_id);
        //self.hist_store.debug_print(n2s.hists_range);

        let (pgrad, phess) = (n2s.grad_sum, n2s.hess_sum);
        let ploss = self.get_loss(pgrad, phess);

        for feat_id in 0..self.dataset.nfeatures() {
            let hist = self.hist_store.get_hist(n2s.hists_range, feat_id);

            // Compute best split based on histogram
            for feat_val in 0..hist.len() {
                let (lgrad, lhess) = hist[feat_val].unpack();
                let (rgrad, rhess) = (pgrad - lgrad, phess - lhess);

                if lhess < self.config.min_sum_hessian { continue; }
                if rhess < self.config.min_sum_hessian { continue; }

                let lloss = self.get_loss(lgrad, lhess);
                let rloss = self.get_loss(rgrad, rhess);
                let gain = ploss - lloss - rloss;

//                debug!("N{:03}-F{:02}={:<3} candidate gain {:.3}",
//                       n2s.node_id, feat_id, feat_val, gain);

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some(Split { 
                        split_crit: SplitCrit::EqTest(feat_id, feat_val as u16),
                        left_grad_sum: lgrad,
                        left_hess_sum: lhess,
                        right_grad_sum: rgrad,
                        right_hess_sum: rhess,
                    });
                }
            }
        }
 
        best_split
    }

    fn get_root_node2split(&mut self) -> Node2Split {
        (self.get_root_n2s_fun)(self)
    }

    fn get_left_right_node2split(&mut self, parent_n2s: &Node2Split, fval_mask: BitVecRef)
        -> (Node2Split, Node2Split)
    {
        let left_id = self.tree.left_child(parent_n2s.node_id);
        let right_id = self.tree.right_child(parent_n2s.node_id);

        let mut left_n2s = self.split_examples(parent_n2s, left_id, &fval_mask, |m| m);
        self.build_histograms(&left_n2s);
        let (grad_sum, hess_sum) = self.hist_store.sum_hist(left_n2s.hists_range, 0).unpack();
        left_n2s.grad_sum = grad_sum;
        left_n2s.hess_sum = hess_sum;

        let mut right_n2s = self.split_examples(parent_n2s, right_id, &fval_mask, |m| !m);
        self.derive_histograms(&parent_n2s, &left_n2s, &right_n2s);
        right_n2s.grad_sum = parent_n2s.grad_sum - left_n2s.grad_sum;
        right_n2s.hess_sum = parent_n2s.hess_sum - left_n2s.hess_sum;

        (left_n2s, right_n2s)
    }

    fn split_examples<F>(&mut self, parent_n2s: &Node2Split, child_id: usize,
                         fval_mask: &BitVecRef, f: F) -> Node2Split
    where F: Fn(u32) -> u32
    {
        let parent_indices = self.index_store.get_bitvec(parent_n2s.idx_range);
        let nblocks = parent_indices.len();
        let n_u32 = parent_indices.block_len::<u32>();

        let mut child_n2s = Node2Split::new(child_id, &mut self.hist_store);
        child_n2s.compressed = parent_n2s.compressed; // child is compr if parent is compr
        child_n2s.idx_range = parent_n2s.idx_range;
        child_n2s.grad_range = parent_n2s.grad_range;
        child_n2s.mask_range = self.mask_store.alloc_zero_blocks(nblocks);

        let (parent_masks, mut child_masks) = self.mask_store.get_two_bitvecs_mut(
            parent_n2s.mask_range,
            child_n2s.mask_range);

        let mut zero_count = 0;
        for j in 0..n_u32 {
            let i = *parent_indices.get::<u32>(j); // j is node index (compr), i is global index
            let pmask = *parent_masks.get::<u32>(j);
            let fmask = *fval_mask.get::<u32>(i as usize);
            let mask = pmask & f(fmask);

            if mask == 0 { zero_count += 1; }

            child_masks.set(j, mask);
        }

        // We encountered enough zero blocks to apply compression
        let zero_ratio = zero_count as NumT / n_u32 as NumT;
        if zero_ratio > self.config.compression_threshold {
            debug!("N{:03}->N{:03} applying compression, #zero = {}/{} (ratio={:.2})",
                   parent_n2s.node_id, child_n2s.node_id, zero_count, n_u32, zero_ratio);

            let n_u32_child = n_u32 - zero_count;
            let child_n2s = self.compress_examples(parent_n2s, child_n2s, n_u32_child);

            child_n2s
        } else {
            //debug!("N{:03}->N{:03} NOT applying compression, #zero = {}/{} (ratio={:.2})",
            //       parent_n2s.node_id, child_n2s.node_id, zero_count, n_u32, zero_ratio);
            child_n2s
        }
    }

    fn compress_examples(&mut self, parent_n2s: &Node2Split, child_n2s: Node2Split,
                         n_u32_child: usize) -> Node2Split
    {
        (self.compress_examples_fun)(self, parent_n2s, child_n2s, n_u32_child)
    }

    fn build_histograms(&mut self, n2s: &Node2Split) {
        (self.build_histograms_fun)(self, n2s)
    }

    fn derive_histograms(&mut self, parent_n2s: &Node2Split, left_n2s: &Node2Split,
                         right_n2s: &Node2Split)
    {
        self.hist_store.hists_subtract(
            parent_n2s.hists_range,
            left_n2s.hists_range,
            right_n2s.hists_range);
    }

    fn get_loss(&self, grad_sum: NumT, hess_sum: NumT) -> NumT {
        let lambda = self.config.reg_lambda;
        -0.5 * ((grad_sum * grad_sum) / (hess_sum + lambda))
    }

    fn get_best_value(&self, grad_sum: NumT, hess_sum: NumT) -> NumT {
        let lambda = self.config.reg_lambda;
        -grad_sum / (hess_sum + lambda)
    }

    pub fn into_tree(self) -> Tree {
        self.hist_store.reset();
        self.index_store.reset();
        self.mask_store.reset();
        self.gradient_store.reset();

        self.tree
    }

    #[allow(dead_code)]
    fn debug_print_split(&self, n2s: &Node2Split, split: &Split) {
        match self.config.discr_nbits {
            1 => self.debug_print_split_generic::<BitSliceLayout1>(n2s, split),
            2 => self.debug_print_split_generic::<BitSliceLayout2>(n2s, split),
            4 => self.debug_print_split_generic::<BitSliceLayout4>(n2s, split),
            _ => panic!("invalid config.discr_nbits"),
        }
    }

    #[allow(dead_code)]
    fn debug_print_split_generic<BSL>(&self, n2s: &Node2Split, split: &Split)
    where BSL: BitSliceLayout {
        let nid = n2s.node_id;
        let bounds = self.grad_bounds;
        let (feat_id, feat_val) = split.split_crit.unpack_eqtest().unwrap();

        let ems_bitset = self.mask_store.get_bitvec(n2s.mask_range);
        let idx_bitset = self.index_store.get_bitvec(n2s.idx_range);

        let ems = ems_bitset.cast::<u32>(); // example mask
        let indices = idx_bitset.cast::<u32>();
        let gradients = self.gradient_store.get_bitslice::<BSL>(n2s.grad_range);
        let column = self.dataset.features()[feat_id].get_raw_data();
        let feat_val_mask = self.dataset.features()[feat_id].get_bitvec(feat_val).unwrap();

        let mut nexamples = 0;
        for i in 0..indices.len() {
            nexamples += ems[i].count_ones();
        }

        let (lid, rid) = (self.tree.left_child(nid), self.tree.right_child(nid));
        let lvalue = self.get_best_value(split.left_grad_sum, split.left_hess_sum);
        let rvalue = self.get_best_value(split.right_grad_sum, split.right_hess_sum);
        let gain = self.get_loss(n2s.grad_sum, n2s.hess_sum)
            - self.get_loss(split.left_grad_sum, split.left_hess_sum)
            - self.get_loss(split.right_grad_sum, split.right_hess_sum);

        let loss = self.get_loss(n2s.grad_sum, n2s.hess_sum);
        println!("n2s(N{:03}, #ex={}, loss={:.3}, g={:.3}, h={:.3}, gain={:.3} crit={:?})",
            nid, nexamples, loss, n2s.grad_sum, n2s.hess_sum, gain, split.split_crit);
        println!("{:>6} {:>6} {:>8} {:>8} {:>6} {:>4} {:>4}", "blck", "idx", "grad", "trgt",
            "dif?", "col", "L/R");
        for i in 0..indices.len() {
            for k in 0..32 {
                if ems[i] >> k & 0x1 != 0x1 { continue; }
                let x = indices[i] as usize * 32 + k;
                let y = i * 32 + k;
                println!("{:6} {:6} {:8.3} {:8.3} {:6.1} {:4} {:>4}",
                         i,
                         x,
                         gradients.get_scaled_value(y, bounds),
                         self.gradients[x],
                         (self.gradients[x] - gradients.get_scaled_value(y, bounds)).abs().log10(),
                         column[x],
                         if feat_val_mask.get_bit(x) { "L " } else { " R" });
            }
        }

        debug!("N{:03}-F{:02}=={:<3} gain={:.3} N{:03}={:.2} N{:03}={:.2}", nid, feat_id,
            feat_val, gain, lid, lvalue, rid, rvalue);
    }
}





// - 'Generic' methods: implementations depends on discr_width ------------------------------------

macro_rules! get_root_node2split {
    ($f:ident, $bsl:ident) => {
        fn $f(l: &mut TreeLearner) -> Node2Split {
            let nexamples  = l.dataset.nexamples();
            let grad_range = l.gradient_store.alloc_zero_bitslice::<$bsl>(nexamples);
            let mask_range = l.mask_store.alloc_one_bits(nexamples);
            let nblocks    = l.mask_store.get_bitvec(mask_range).block_len::<u32>();
            let idx_range  = l.index_store.alloc_from_iter::<u32, _>(nblocks, 0..nblocks as u32);

            // Put target gradient values in slice
            {
                let mut slice = l.gradient_store.get_bitslice_mut::<$bsl>(grad_range);
                for (i, &v) in l.gradients.iter().enumerate() {
                    slice.set_scaled_value(i, v, l.grad_bounds);
                }
            }

            let mut n2s = Node2Split::new(0, &mut l.hist_store);
            n2s.idx_range = idx_range;
            n2s.mask_range = mask_range;
            n2s.grad_range = grad_range;

            l.build_histograms(&n2s);
            let (grad_sum, hess_sum) = l.hist_store.sum_hist(n2s.hists_range, 0).unpack();
            n2s.grad_sum = grad_sum;
            n2s.hess_sum = hess_sum;
            n2s
        }
    }
}

macro_rules! compress_examples {
    ($f:ident, $bsl:ident) => {
        fn $f(l: &mut TreeLearner, parent_n2s: &Node2Split, mut child_n2s: Node2Split,
              n_u32_child: usize) -> Node2Split
        {
            // This will become a compressed node
            child_n2s.compressed = true;

            // Allocate space for indices, gradients for child (reuse masks from `split_examples`)
            let nvalues = n_u32_child * size_of::<u32>() * 8;
            child_n2s.idx_range = l.index_store.alloc_zero_bits(nvalues);
            child_n2s.grad_range = l.gradient_store.alloc_zero_bitslice::<$bsl>(nvalues);

            // Get references to the structures
            let (parent_indices, mut child_indices) = l.index_store.get_two_bitvecs_mut(
                parent_n2s.idx_range,
                child_n2s.idx_range);
            let n_u32_parent = parent_indices.block_len::<u32>();
            let mut child_masks = l.mask_store.get_bitvec_mut(child_n2s.mask_range);
            let (parent_grads, mut child_grads) = l.gradient_store.get_two_bitslices_mut::<$bsl>(
                parent_n2s.grad_range,
                child_n2s.grad_range);

            let mut k = 0; // child node index (compr)
            for j in 0..n_u32_parent {
                debug_assert!(j >= k);

                let i = *parent_indices.get::<u32>(j); // j is parent index (compr), i is global index

                // in `split_examples`, we stored all zeros and used the parent indices `j`; copy this
                // to the smaller child indices `k` by skipping zero blocks:
                let mask = *child_masks.get::<u32>(j);
                if mask == 0 { continue; }

                child_masks.set::<u32>(k, mask);
                child_indices.set::<u32>(k, i);
                child_grads.copy_block_from::<u32, _>(&parent_grads, j, k); // from=j, to=k
                k += 1;
            }

            debug_assert_eq!(k, n_u32_child);

            // Zero the remaining blocks of child_masks: not doing this breaks things!
            for j in n_u32_child..n_u32_parent {
                child_masks.set::<u32>(j, 0);
            }

            // Resize the masks, it has the size of the parent masks, but we've just compressed it
            let idx_len = child_n2s.idx_range.1 - child_n2s.idx_range.0;
            child_n2s.mask_range = (child_n2s.mask_range.0, child_n2s.mask_range.0 + idx_len);

            child_n2s
        }
    }
}

macro_rules! build_histograms {
    ($f:ident, $bsl:ident, $sum_method:ident) => {
        fn $f(l: &mut TreeLearner, n2s: &Node2Split) {
            get_grad_and_hess_sums!(get_grad_and_hess_sums, $bsl, $sum_method);

            for feature in l.dataset.features() {
                match feature.get_repr() {
                    Some(&FeatureRepr::BitVecFeature(ref f)) => {
                        for i in 0..f.card {
                            let fval_mask = f.get_bitvec(i as u16);
                            let (grad, hess) = get_grad_and_hess_sums(l, n2s, &fval_mask);
                            let histval = HistVal { grad: grad, hess: hess };
                            let hist = l.hist_store.get_hist_mut(n2s.hists_range, feature.id());
                            hist[i] = histval;
                        }
                    },
                    _ => { panic!("unsupported feature type") }
                }
            }
        }
    }
}

macro_rules! get_grad_and_hess_sums {
    ($f:ident, $bsl:ident, naive) => {
        fn $f(l: &TreeLearner, n2s: &Node2Split, fval_mask: &BitVecRef) -> (NumT, NumT)
        {
            let ems_bitset = l.mask_store.get_bitvec(n2s.mask_range);
            let idx_bitset = l.index_store.get_bitvec(n2s.idx_range);

            let vms = feat_val_mask.cast::<u32>(); // feature value mask
            let ems = ems_bitset.cast::<u32>(); // example mask
            let indices = idx_bitset.cast::<u32>();
            let gradients = l.gradient_store.get_bitslice::<$bsl>(n2s.grad_range);

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
                        sum1 += gradients.get_scaled_value(i*32 + k, l.grad_bounds);
                        count1 += 1;
                    }
                }

                sum += sum1;
                count += count1;
            }

            (sum, count as NumT)
        }
    };

    ($f:ident, $bsl:ident, simd) => {
        fn $f(l: &TreeLearner, n2s: &Node2Split, fval_mask: &BitVecRef) -> (NumT, NumT)
        {
            fn sum_uncompr(l: &TreeLearner, n2s: &Node2Split, fval_mask: &BitVecRef)
                -> (NumT, NumT)
            {
                let ems = l.mask_store.get_bitvec(n2s.mask_range);
                let gradients = l.gradient_store.get_bitslice::<$bsl>(n2s.grad_range);

                let count = ems.count_ones_and(&fval_mask) as NumT;
                let hess = count; // TODO proper hess support

                //if hess < min_hess || (n2s.hess_sum - hess) < min_hess { return (0.0, 0.0); }
                let sum = unsafe {
                    gradients.sum_all_masked2_unsafe(&ems, &fval_mask)
                };
                let sum = $bsl::linproj(sum as NumT, count, l.grad_bounds);

                (sum, hess)
            }

            fn sum_compr(l: &TreeLearner, n2s: &Node2Split, fval_mask: &BitVecRef)
                -> (NumT, NumT)
            {
                let ems = l.mask_store.get_bitvec(n2s.mask_range);
                let gradients = l.gradient_store.get_bitslice::<$bsl>(n2s.grad_range);
                let indices = l.index_store.get_bitvec(n2s.idx_range);

                let count = unsafe { ems.count_ones_and_compr_unsafe(&indices, &fval_mask) as NumT };
                let hess = count; // TODO proper hess support

                //if hess < min_hess || (n2s.hess_sum - hess) < min_hess { return (0.0, 0.0); }
                let sum = unsafe {
                    gradients.sum_all_masked2_compr_unsafe(&indices, &ems, &fval_mask)
                };
                let sum = $bsl::linproj(sum as NumT, count, l.grad_bounds);

                (sum, hess)
            }

            if !n2s.compressed { sum_uncompr(l, n2s, fval_mask) }
            else               { sum_compr(l, n2s, fval_mask)   }
        }
    }
}

// Generate specialized implementations for each BitSlice layout (discr_width)
// We do this this way because rustc didn't generate efficient code when TreeBuilder was given a
// generic paramter BSL: BitSliceLayout
get_root_node2split!(get_root_node2split_w1, BitSliceLayout1);
get_root_node2split!(get_root_node2split_w2, BitSliceLayout2);
get_root_node2split!(get_root_node2split_w4, BitSliceLayout4);

compress_examples!(compress_examples_w1, BitSliceLayout1);
compress_examples!(compress_examples_w2, BitSliceLayout2);
compress_examples!(compress_examples_w4, BitSliceLayout4);

build_histograms!(build_histograms_w1, BitSliceLayout1, simd);
build_histograms!(build_histograms_w2, BitSliceLayout2, simd);
build_histograms!(build_histograms_w4, BitSliceLayout4, simd);

type GetRootN2SFun = fn(&mut TreeLearner) -> Node2Split;
type CompressExamplesFun = fn(&mut TreeLearner, &Node2Split, Node2Split, usize) -> Node2Split;
type BuildHistogramsFun = fn(&mut TreeLearner, &Node2Split);

fn get_get_root_node2split_fun(discr_width: usize)
    -> GetRootN2SFun
{
    match discr_width {
        1 => get_root_node2split_w1,
        2 => get_root_node2split_w2,
        4 => get_root_node2split_w4,
        _ => panic!("invalid discr_width"),
    }
}

fn get_compress_examples_fun(discr_width: usize)
    -> CompressExamplesFun
{
    match discr_width {
        1 => compress_examples_w1,
        2 => compress_examples_w2,
        4 => compress_examples_w4,
        _ => panic!("invalid discr_width"),
    }
}

fn get_build_histograms_fun(discr_width: usize)
    -> BuildHistogramsFun
{
    match discr_width {
        1 => build_histograms_w1,
        2 => build_histograms_w2,
        4 => build_histograms_w4,
        _ => panic!("invalid discr_width"),
    }
}
