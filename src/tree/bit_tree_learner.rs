use std::ops::{Sub, Add};

use log::debug;

use crate::NumT;
use crate::config::Config;
use crate::tree::{Tree, SplitCrit};
use crate::dataset::{Dataset, FeatureRepr};
use crate::slice_store::{SliceRange, HistStore, BitBlockStore};
use crate::slice_store::{BitVecRef, BitVecMut, BitSliceRef, BitSliceMut};
use crate::slice_store::{BitSliceInfo, BitSliceScaleInfo, BitSliceRuntimeInfo};

struct Node2Split {
    /// The id of the node to split.
    node_id: usize,

    /// Cache grad_sum and hess_sum to compute right stats given left stats.
    grad_sum: NumT,
    hess_sum: NumT,

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

pub struct TreeLearner<'a, I>
where I: 'a + BitSliceInfo + BitSliceScaleInfo {
    config: &'a Config,
    dataset: &'a Dataset,
    gradients: Vec<NumT>,
    tree: Tree,
    bitslice_info: I,

    /// Storage for histograms.
    hist_store: HistStore<HistVal>,

    /// Storage for indices pointing to none-zero 32-bit example selection blocks.
    index_store: BitBlockStore,

    /// Storage for example selection masks.
    mask_store: BitBlockStore,

    /// Store for gradients
    gradient_store: BitBlockStore,
}

impl <'a> TreeLearner<'a, BitSliceRuntimeInfo> {
    pub fn new(config: &'a Config, data: &'a Dataset, gradients: Vec<NumT>) -> Self {
        let tree = Tree::new(config.max_tree_depth);

        let bitslice_info = BitSliceRuntimeInfo::new(
            config.discr_bits,
            config.discr_lo,
            config.discr_hi);

        let hist_store = HistStore::for_dataset(data);
        let index_store = BitBlockStore::new(1024);
        let mask_store = BitBlockStore::new(1024);
        let gradient_store = BitBlockStore::new(1024 * bitslice_info.width());

        TreeLearner {
            config: config,
            dataset: data,
            gradients: gradients,
            tree: tree,
            bitslice_info: bitslice_info,

            hist_store: hist_store,
            index_store: index_store,
            mask_store: mask_store,
            gradient_store: gradient_store,
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

            self.debug_print_split(&n2s, &split);

            // Enqueue children to be split if they are not leaves
            if !self.tree.is_max_leaf_node(self.tree.left_child(node_id)) {
                let val_masks = self.dataset.features()[feat_id].get_bitvec(feat_val).unwrap();
                let (left_n2s, right_n2s) = self.get_left_right_node2split(&n2s, val_masks);

                // Schedule the left and right children to be split
                stack.push(right_n2s);
                stack.push(left_n2s);

                //let mut left = self.split_examples(&n2s, left_id, split.left_loss, val_masks,
                //                               |pm, vm| pm & vm);
                //let mut right = self.split_examples(&n2s, right_id, split.right_loss, val_masks,
                //                                |pm, vm| pm & !vm);

                //// Ensure that nodes2split have histograms ready to use
                //self.build_histograms(&mut left);
                //self.derive_histograms(&mut right, n2s.hists_range, left.hists_range);

            }

            // Set tree node values
            let left_value = self.get_best_value(split.left_grad_sum, split.left_hess_sum);
            let right_value = self.get_best_value(split.right_grad_sum, split.right_hess_sum);
            self.tree.split_node(node_id, split.split_crit, left_value, right_value);

            // Free histogram of node that was just split
            self.hist_store.free_hists(n2s.hists_range);
        }
    }

    fn find_best_split(&mut self, n2s: &Node2Split) -> Option<Split> {
        let mut best_gain = self.config.min_gain;
        let mut best_split = None;

        print!("N{:03}: ", n2s.node_id);
        self.hist_store.debug_print(n2s.hists_range);

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

                debug!("N{:03}-F{:02}={:<3} candidate gain {:.3}",
                       n2s.node_id, feat_id, feat_val, gain);

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
        // TODO bagging - reservoir sampling
        let nexamples  = self.dataset.nexamples();
        let grad_range = self.gradient_store.alloc_zero_bitslice(nexamples, &self.bitslice_info);
        let mask_range = self.mask_store.alloc_one_bits(nexamples);
        let nblocks    = self.mask_store.get_bitvec(mask_range).block_len::<u32>();
        let idx_range  = self.index_store.alloc_from_iter::<u32, _>(nblocks, 0..nblocks as u32);

        // Put target gradient values in slice
        {
            let mut slice = self.gradient_store.get_bitslice_mut(grad_range, &self.bitslice_info);
            for (i, &v) in self.gradients.iter().enumerate() {
                slice.set_scaled_value(i, v);
            }
        }

        let mut n2s = Node2Split::new(0, &mut self.hist_store);
        n2s.idx_range = idx_range;
        n2s.mask_range = mask_range;
        n2s.grad_range = grad_range;

        self.build_histograms(&n2s);
        let (grad_sum, hess_sum) = self.hist_store.sum_hist(n2s.hists_range, 0).unpack();
        n2s.grad_sum = grad_sum;
        n2s.hess_sum = hess_sum;
        n2s
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
        {
            // test if equal to parent-left
            let (g, h) = self.hist_store.sum_hist(right_n2s.hists_range, 0).unpack();
            let dg = (g - right_n2s.grad_sum).abs();
            let dh = (h - right_n2s.hess_sum).abs();
            println!("grad_diff {}vs{}: {}", left_id, right_id, dg);
            println!("hess_diff {}vs{}: {}", left_id, right_id, dh);
        }

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
        
        child_n2s
    }

    fn build_histograms(&mut self, n2s: &Node2Split) {
        for feature in self.dataset.features() {
            match feature.get_repr() {
                Some(&FeatureRepr::BitVecFeature(ref f)) => {
                    for i in 0..f.card {
                        let feat_val_mask = f.get_bitvec(i as u16);
                        let (grad, hess) = self.get_grad_and_hess_sums(n2s, feat_val_mask);
                        let histval = HistVal { grad: grad, hess: hess };
                        let hist = self.hist_store.get_hist_mut(n2s.hists_range, feature.id());
                        hist[i] = histval;
                    }
                },
                _ => { panic!("unsupported feature type") }
            }
        }
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


    /// OPTIMIZE THIS
    fn get_grad_and_hess_sums(&self, n2s: &Node2Split, feat_val_mask: BitVecRef) -> (NumT, NumT) {
        let ems_bitset = self.mask_store.get_bitvec(n2s.mask_range);
        let idx_bitset = self.index_store.get_bitvec(n2s.idx_range);

        let vms = feat_val_mask.cast::<u32>(); // feature value mask
        let ems = ems_bitset.cast::<u32>(); // example mask
        let indices = idx_bitset.cast::<u32>();
        let gradients = self.gradient_store.get_bitslice(n2s.grad_range, &self.bitslice_info);

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
                    sum1 += gradients.get_scaled_value(i*32 + k);
                    count1 += 1;
                }
            }

            sum += sum1;
            count += count1;
        }

        (sum, count as NumT)
    }

    pub fn reset(&mut self) {
        self.tree = Tree::new(self.config.max_tree_depth);
        self.hist_store = HistStore::for_dataset(self.dataset);
    }

    pub fn into_tree(self) -> Tree {
        self.tree
    }

    fn debug_print_split(&self, n2s: &Node2Split, split: &Split) {
        let nid = n2s.node_id;
        let (feat_id, feat_val) = split.split_crit.unpack_eqtest().unwrap();

        let ems_bitset = self.mask_store.get_bitvec(n2s.mask_range);
        let idx_bitset = self.index_store.get_bitvec(n2s.idx_range);

        let ems = ems_bitset.cast::<u32>(); // example mask
        let indices = idx_bitset.cast::<u32>();
        let gradients = self.gradient_store.get_bitslice(n2s.grad_range, &self.bitslice_info);
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
                         gradients.get_scaled_value(y),
                         self.gradients[x],
                         (self.gradients[x] - gradients.get_scaled_value(y)).abs().log10(),
                         column[x],
                         if feat_val_mask.get_bit(x) { "L " } else { " R" });
            }
        }

        debug!("N{:03}-F{:02}=={:<3} gain={:.3} N{:03}={:.2} N{:03}={:.2}", nid, feat_id,
            feat_val, gain, lid, lvalue, rid, rvalue);
    }
}
