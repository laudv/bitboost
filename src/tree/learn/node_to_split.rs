/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

use std::sync::Arc;

use crate::NumT;
use crate::bitblock::{BitBlock, BitBlocks};
use crate::bitset::Bitset;
use crate::bitslice::Bitslice;
use crate::tree::learn::Histogram;

pub struct NodeToSplit {
    node_id: usize,
    examples_count: u32,
    grad_sum: NumT,

    histogram: Histogram,
    instance_set: Bitset,
    gradients: Arc<Bitslice>,
    indexes: Arc<BitBlocks>, // empty if no compression
}

impl NodeToSplit {

    pub fn is_compressed(&self) -> bool {
        !self.indexes.is_empty()
    }



    /*
    #[inline(always)]
    pub unsafe fn get_instance_set_block_ptr(&self, index: usize) -> *const BitBlock {
        safety_check!(self.indexes.is_empty());
        safety_check!(self.instance_set.len() > index);
        self.instance_set.as_ptr().add(index)
    }

    #[inline(always)]
    pub unsafe fn get_instance_set_block_ptr_c(&self, index: usize) -> *const BitBlock {
        0 as *const BitBlock
    }
    */
}
