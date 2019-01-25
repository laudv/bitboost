use crate::NumT;
use crate::dataset::{Feature};
use crate::slice_store::{BitBlockStore, BitVecRef, SliceRange};
use crate::quantile::{ApproxQuantileStats, ApproxQuantile};

pub struct Splitter {
    quantile: ApproxQuantile,
    fval_mask_store: BitBlockStore,
    split_cand_buffer: Vec<(NumT, NumT, u32)>,
    // quantile
    // bitblock store for cached fval_masks
    //     hashmap mapping (feat_id, fval_id) => (split_value: NumT, slice_range)
    //     do we then still need split_value in histogram??
}

impl Splitter {
    pub fn new() -> Splitter {
        unimplemented!();
    }

    pub fn gen_split_candidates<'a, 'b, I>(&'a mut self, stats: &ApproxQuantileStats,
                                           feature: &'b Feature, examples: I)
        -> SplitCandidateIter<'a>
    where I: Iterator<Item = usize> + 'b
    {
        self.quantile.reset();
        self.quantile.set_stats(stats);

        let features_values = feature.get_raw_data();

        // TODO we need to accumulate !!gradients!!, counts!
        // we need to have access to gradient bitslice
        // we need to sum quickly
        for i in examples {
            let fval = features_values[i];
            self.quantile.feed(fval);
        }

        SplitCandidateIter {
            splitter: self,
            grad_sum_accum: 0.0,
            example_count_accum: 0,
        }
    }

    pub fn gen_fval_mask(&mut self, feature: &Feature, fval_id: usize) -> SliceRange {
        unimplemented!()
    }

    pub fn get_fval_mask(&self, range: SliceRange) -> BitVecRef {
        unimplemented!()
    }

    pub fn free_fval_mask(&mut self, slice_range: SliceRange) {
        unimplemented!()
    }

    pub fn reset(&mut self) {
        self.fval_mask_store.reset();
        self.split_cand_buffer.clear();
    }
}

pub struct SplitCandidateIter<'a> {
    splitter: &'a Splitter,
    grad_sum_accum: NumT,
    example_count_accum: u32,
}

impl <'a> Iterator for SplitCandidateIter<'a> {
    type Item = (NumT, NumT, u32);
    fn next(&mut self) -> Option<(NumT, NumT, u32)> {
        None
    }
}
