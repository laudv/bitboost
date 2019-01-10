use std::ops::{Sub};
use std::marker::PhantomData;

use log::debug;

use NumT;
use NomT;

use config::Config;
use tree::{Tree, SplitCrit};
use tree::loss::LossFun;
use dataset::{Dataset, FeatureRepr};
use bits::{BitBlock, BitVec};
use bits::{ScaledBitSlice, BitSliceLayout};
use slice_store::{SliceRange, HistStore};

struct Node2Split<L>
where L: BitSliceLayout {
    /// The id of the node to split.
    node_id: usize,

    /// The local loss due to this node.
    loss: NumT,

    /// Range to lookup histograms in hist_store
    hists_range: SliceRange,

    /// Indices to non-zero blocks
    indices: BitVec,

    /// For each non-zero block, store a mask that indicates which examples sort into this node.
    mask: BitVec,

    /// Copied target values to improve memory locality
    gradients: ScaledBitSlice<L>,
}

struct Split {
    split_crit: SplitCrit,
    left_value: NumT,
    right_value: NumT,
    left_loss: NumT,
    right_loss: NumT,
    gain: NumT,
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

pub struct TreeLearner<'a, L>
where L: 'a + BitSliceLayout {
    config: &'a Config,
    dataset: &'a Dataset,
    gradients: Vec<NumT>,
    hist_store: HistStore<HistVal>,
    tree: Tree,
    _marker: PhantomData<L>,
}

impl <'a, L> TreeLearner<'a, L>
where L: 'a + BitSliceLayout {
    pub fn new(config: &'a Config, data: &'a Dataset, gradients: Vec<NumT>) -> Self {
        let tree = Tree::new(config.max_tree_depth);
        let hist_store = HistStore::for_dataset(data);

        TreeLearner {
            config: config,
            dataset: data,
            gradients: gradients,
            hist_store: hist_store,
            tree: tree,
            _marker: PhantomData,
        }
    }

    pub fn train(&mut self) {
        self.reset();

        let max_depth = self.config.max_tree_depth;
        let mut stack = Vec::<Node2Split<L>>::with_capacity(max_depth * 2);

        let (root_value, mut root_node2split) = self.get_root_value_and_node2split();
        self.tree.set_root_value(root_value);
        self.build_histograms(&mut root_node2split);

        debug!("N000: loss {}, value {}", root_node2split.loss, root_value);

        stack.push(root_node2split);

        while let Some(node2split) = stack.pop() {
            let node_id = node2split.node_id;

            let split_opt = self.find_best_split(&node2split);
            if split_opt.is_none() { debug!("N{:03} no split", node_id); continue; }

            let split = split_opt.unwrap();
            let left_id = self.tree.left_child(node_id);
            let right_id = self.tree.right_child(node_id);
            let (feat_id, feat_val) = split.split_crit.unpack_eqtest().expect("not an EqTest");

            //self.debug_print_split(&node2split, &split);
            debug!("N{:03}-F{:02}=={:<3} split gain={:.4} l={} r={}", node_id, feat_id, feat_val,
                   split.gain, left_id, right_id);

            self.tree.split_node(node_id, split.split_crit, split.left_value, split.right_value);

            if !self.tree.is_max_leaf_node(left_id) {
                let value_masks = self.dataset.features()[feat_id]
                    .get_bitvec(feat_val).unwrap()
                    .cast::<u32>();
                let mut left = self.split_examples(&node2split, left_id, split.left_loss, value_masks,
                                               |pm, vm| pm & vm);
                let mut right = self.split_examples(&node2split, right_id, split.right_loss, value_masks,
                                                |pm, vm| pm & !vm);

                // Ensure that nodes2split have histograms ready to use
                self.build_histograms(&mut left);
                self.derive_histograms(&mut right, node2split.hists_range, left.hists_range);

                // Schedule the left and right children to be split
                stack.push(right);
                stack.push(left);
            }

            self.hist_store.free_hists(node2split.hists_range);
        }
    }

    fn find_best_split(&mut self, n2s: &Node2Split<L>) -> Option<Split> {
        let parent_loss = n2s.loss;

        let mut best_gain = self.config.min_gain;
        let mut best_split = None;

        for feat_id in 0..self.dataset.nfeatures() {
            let hist = self.hist_store.get_hist(n2s.hists_range, feat_id);
            let (mut fgrad, mut fhess) = (0.0, 0.0);

            // Compute total grad/hess of histogram for this feature
            for feat_val in 0..hist.len() {
                let (grad, hess) = hist[feat_val].unpack();
                fgrad += grad;
                fhess += hess;
            }

            // Compute best split based on histogram
            for feat_val in 0..hist.len() {
                let (lgrad, lhess) = hist[feat_val].unpack();
                let (rgrad, rhess) = (fgrad - lgrad, fhess - lhess);

                if lhess < self.config.min_sum_hessian { continue; }
                if rhess < self.config.min_sum_hessian { continue; }

                let (lval, lloss) = self.best_value_and_loss(lgrad, lhess);
                let (rval, rloss) = self.best_value_and_loss(rgrad, rhess);
                
                let gain = parent_loss - lloss - rloss;

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some(Split { 
                        split_crit: SplitCrit::EqTest(feat_id, feat_val as u16),
                        left_value: lval,
                        right_value: rval,
                        left_loss: lloss,
                        right_loss: rloss,
                        gain: gain,
                    });
                }
            }
        }

        best_split
    }

    fn split_examples<F>(&self, n2s: &Node2Split<L>, child_node_id: usize, child_loss: NumT,
                         value_masks: &[u32], f: F)
        -> Node2Split<L>
    where F: Fn(u32, u32) -> u32
    {
        // NOTE: can save memory by reusing parent, left/right structures if indices are equal
        // --> reuse indices and child_gradients

        let parent_masks = n2s.mask.cast::<u32>();
        let parent_indices = n2s.indices.cast::<u32>();

        let mut child_indices_bitvec = n2s.indices.clone();
        let mut child_gradients = n2s.gradients.clone();
        let mut child_masks_bitvec = BitVec::zero_bits(parent_indices.len() * 32);
        {
            let child_masks = child_masks_bitvec.cast_mut::<u32>();

            for (i, &ju32) in parent_indices.iter().enumerate() {
                let j = ju32 as usize;
                let pmask = parent_masks[i];
                let vmask = value_masks[j];
                let mask = f(pmask, vmask);

                child_masks[i] = mask;
            }
        };

        Node2Split {
            node_id: child_node_id,
            loss: child_loss,
            hists_range: (0, 0),
            indices: child_indices_bitvec,
            mask: child_masks_bitvec,
            gradients: child_gradients,
        }
    }

    fn get_root_value_and_node2split(&mut self) -> (NumT, Node2Split<L>) {
        // TODO bagging - reservoir sampling
        let nexamples = self.dataset.nexamples();
        let slice = ScaledBitSlice::<L>::new(nexamples, self.config.discr_lo, self.config.discr_hi,
                                             self.gradients.iter().cloned());
        let mask = BitVec::one_bits(nexamples);
        let nblocks = mask.cast::<u32>().len();
        let indices = BitVec::from_block_iter::<u32, _>(nblocks, 0..nblocks as u32);

        let mut n2s = Node2Split {
            node_id: 0,
            loss: 0.0,
            hists_range: (0, 0),
            indices: indices,
            mask: mask,
            gradients: slice,
        };

        let (sum_grad, sum_hess) = Self::get_grad_and_hess_sums(&n2s, &n2s.mask);
        let (value, loss) = self.best_value_and_loss(sum_grad, sum_hess);

        n2s.loss = loss;

        (value, n2s)
    }

    fn build_histograms(&mut self, n2s: &mut Node2Split<L>) {
        let hists_range = self.hist_store.alloc_hists();

        for feature in self.dataset.features() {
            let histogram = self.hist_store.get_hist_mut(hists_range, feature.id());
            match feature.get_repr() {
                Some(&FeatureRepr::BitVecFeature(ref bitvecs)) => {
                    for (i, feat_val_mask) in bitvecs.iter().enumerate() {
                        let (grad, hess) = Self::get_grad_and_hess_sums(n2s, feat_val_mask);
                        let histval = HistVal { grad: grad, hess: hess };
                        histogram[i] = histval;
                    }
                },
                _ => { panic!("unsupported feature type") }
            }
        }

        n2s.hists_range = hists_range;
    }

    fn derive_histograms(&mut self, n2s: &mut Node2Split<L>,
                         prange: SliceRange, lrange: SliceRange)
    {
        let rrange = self.hist_store.alloc_hists();
        self.hist_store.hists_subtract(prange, lrange, rrange);

        n2s.hists_range = rrange;
    }

    fn best_value_and_loss(&self, grad_sum: NumT, hess_sum: NumT) -> (NumT, NumT) {
        let lambda = self.config.reg_lambda;
        let value = -grad_sum / (hess_sum + lambda);
        let loss = -0.5 * ((grad_sum * grad_sum) / (hess_sum + lambda));
        (value, loss)
    }

    /// OPTIMIZE THIS
    fn get_grad_and_hess_sums(n2s: &Node2Split<L>, feat_val_mask: &BitVec) -> (NumT, NumT) {
        let vms = feat_val_mask.cast::<u32>(); // feature value mask
        let ems = n2s.mask.cast::<u32>();      // example mask

        let mut count = 0;
        let mut sum = 0.0;

        //print!("N({:03}): 0.0", self.node_id);

        for (i, &ju32) in n2s.indices.cast::<u32>().iter().enumerate() {
            let j = ju32 as usize;
            let vm = vms[j];
            let em = ems[i];
            let m = vm & em;

            //let (count0, sum0) = self.gradients.sum_block32(i, m);
            let (mut count1, mut sum1) = (0, 0.0);

            for k in 0..32 {
                if m >> k & 0x1 == 0x1 {
                    //print!(" + {:.3}", self.gradients.get_value(i*32 + k));
                    sum1 += n2s.gradients.get_value(i*32 + k);
                    count1 += 1;
                }
            }

            //assert_eq!(count0, count1);
            //println!("\n{:e}", (sum0-sum1).abs());
            //assert!((sum0 - sum1).abs() < 1e-5);

            sum += sum1;
            count += count1;
        }
        //println!(" == {} ({})", sum, count);

        (sum, count as NumT)
    }

    pub fn reset(&mut self) {
        self.tree = Tree::new(self.config.max_tree_depth);
        self.hist_store = HistStore::for_dataset(self.dataset);
    }

    pub fn into_tree(self) -> Tree {
        self.tree
    }

    fn debug_print_split(&self, n2s: &Node2Split<L>, split: &Split) {
        let nid = n2s.node_id;
        let (feat_id, feat_val) = split.split_crit.unpack_eqtest().unwrap();

        let indices = n2s.indices.cast::<u32>();
        let masks = n2s.mask.cast::<u32>();
        let column = self.dataset.features()[feat_id].get_raw_data();
        let bitvec = self.dataset.features()[feat_id].get_bitvec(feat_val).unwrap();

        let mut nexamples = 0;
        for i in 0..indices.len() {
            nexamples += masks[i].count_ones();
        }

        println!("node2split(N{:03}, #examples={}, loss={}, crit={:?})", nid, nexamples,
                n2s.loss, split.split_crit);
        println!("{:>6} {:>6} {:>8} {:>8} {:>6} {:>4} {:>4}", "blck", "idx", "grad", "trgt",
                 "dif?", "col", "L/R");
        for i in 0..indices.len() {
            for k in 0..32 {
                if masks[i] >> k & 0x1 != 0x1 { continue; }
                let x = indices[i] as usize * 32 + k;
                let y = i * 32 + k;
                println!("{:6} {:6} {:8.3} {:8.3} {:6.1} {:4} {:>4}",
                         i,
                         x,
                         n2s.gradients.get_value(y),
                         self.gradients[x],
                         (self.gradients[x] - n2s.gradients.get_value(y)).abs().log10(),
                         column[x],
                         if bitvec.get_bit(x) { "L " } else { " R" });
            }
        }
    }
}
