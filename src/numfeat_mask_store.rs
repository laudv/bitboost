use crate::NumT;
use crate::bitblock::set_bit;
use crate::dataset::{Feature};
use crate::slice_store::{BitBlockStore, BitVecRef, SliceRange};

/// Proposes split candidates for numerical features and generates bitvec masks given a particular
/// feature value.
pub struct NumFeatMaskStore {
    fval_mask_store: BitBlockStore,
}

impl NumFeatMaskStore {
    pub fn new(initial_cap: usize) -> NumFeatMaskStore {
        NumFeatMaskStore {
            fval_mask_store: BitBlockStore::new(initial_cap),
        }
    }

    //pub fn gen_split_candidates<'a, 'b, I>(&'a mut self, feature_data: &'b [NumT],
    //                                       gradients: &'b [NumT],
    //                                       (min_value, max_value): (NumT, NumT),
    //                                       examples: I)
    //    -> SplitCandidateIter<'a>
    //where I: Iterator<Item = usize> + 'b
    //{
    //    self.hist.reset();
    //    self.hist.set_min_max(min_value, max_value);

    //    // Insert all feature values in the histogram, and accumulate grad sums as additional
    //    // bucket data.
    //    assert_eq!(feature_data.len(), gradients.len());
    //    for i in examples {
    //        let grad = gradients[i];
    //        let fval = feature_data[i];

    //        self.hist.insert(fval, grad);
    //    }

    //    let cumsum = self.hist.cumsum();

    //    // The iterator loops over the configured number of quantiles and generates (split_value,
    //    // grad_sum, example_count) tuples.
    //    SplitCandidateIter {
    //        i: 0,
    //        nsplits: self.nsplits,
    //        cumsum,
    //    }
    //}

    pub fn gen_fval_mask(&mut self, feature: &Feature, split_value: NumT) -> SliceRange {
        let data = feature.get_raw_data();
        let range = self.fval_mask_store.alloc_zero_bits(data.len());
        let mut fval_mask = self.fval_mask_store.get_bitvec_mut(range);

        let mut mask = 0x0u64;
        let mut mask_i = 0;
        let mut k = 0u8;
        for &value in data {
            if k == 64 {
                fval_mask.set(mask_i, mask);
                mask_i += 1;
                mask = 0x0;
                k = 0;
            }

            mask = set_bit::<u64>(mask, k, value < split_value);
            k += 1;
        }
        if mask != 0 { fval_mask.set(mask_i, mask); }

        range
    }

    pub fn get_fval_mask(&self, range: SliceRange) -> BitVecRef {
        self.fval_mask_store.get_bitvec(range)
    }

    pub fn free_fval_mask(&mut self, slice_range: SliceRange) {
        self.fval_mask_store.free_blocks(slice_range);
    }

    pub fn reset(&mut self) {
        self.fval_mask_store.reset();
    }
}

//pub struct SplitCandidateIter<'a> {
//    i: usize,
//    nsplits: usize,
//    cumsum: CumSum<'a, NumT>,
//}
//
//impl <'a> Iterator for SplitCandidateIter<'a> {
//    type Item = (NumT, NumT, u32);
//    fn next(&mut self) -> Option<(NumT, NumT, u32)> { // split_value, grad_sum, ex_count
//        if self.i < self.nsplits {
//            let q = self.i as NumT / self.nsplits as NumT;
//            let (ex_count, split_value, grad_sum) = self.cumsum.approx_quantile_and_data(q);
//            self.i += 1;
//            Some((split_value, grad_sum, ex_count))
//        } else { None }
//    }
//}
