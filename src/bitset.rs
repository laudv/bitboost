/*
 * Copyright (c) DTAI - KU Leuven - All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Laurens Devos, 2019
*/

use std::ops::{Deref, DerefMut};

use crate::bitblock::{BitBlock, BitBlocks};




// - Compressed -----------------------------------------------------------------------------------

pub struct FullBitset {
    blocks: BitBlocks,
}

impl FullBitset {
    pub fn zeros(nbits: usize) -> FullBitset {
        FullBitset { 
            blocks: BitBlocks::zero_bits(nbits)
        }
    }

    pub fn ones(nbits: usize) -> FullBitset {
        FullBitset { 
            blocks: BitBlocks::one_bits(nbits)
        }
    }

    pub fn get_bit(&self, index: usize) -> bool {
        let i = index / BitBlock::nbits();
        let j = index % BitBlock::nbits();
        self.blocks[i].get_bit(j)
    }

    pub fn set_bit(&mut self, index: usize, bit: bool) {
        let i = index / BitBlock::nbits();
        let j = index % BitBlock::nbits();
        self.blocks[i].set_bit(j, bit);
    }

    pub fn enable_bit(&mut self, index: usize) {
        let i = index / BitBlock::nbits();
        let j = index % BitBlock::nbits();
        self.blocks[i].enable_bit(j)
    }

    pub fn disable_bit(&mut self, index: usize) {
        let i = index / BitBlock::nbits();
        let j = index % BitBlock::nbits();
        self.blocks[i].enable_bit(j)
    }

    pub fn into_bitset(self) -> Bitset {
        Bitset::Full(self)
    }
}

impl Deref for FullBitset {
    type Target = BitBlocks;
    fn deref(&self) -> &BitBlocks {
        &self.blocks
    }
}

impl DerefMut for FullBitset {
    fn deref_mut(&mut self) -> &mut BitBlocks {
        &mut self.blocks
    }
}



// - Compressed -----------------------------------------------------------------------------------

pub struct CompressedBitset {
    indexes: BitBlocks,
    blocks: BitBlocks,
}

impl CompressedBitset {
    pub fn into_bitset(self) -> Bitset {
        Bitset::Compressed(self)
    }
}





// - Either a full or a compressed bitset: choose dynamically -------------------------------------

pub enum Bitset {
    Full(FullBitset),
    Compressed(CompressedBitset),
}

impl Bitset {
    pub fn count_ones_and(b1: &Bitset, b2: &Bitset) -> u64 {
        use Bitset::*;

        match (b1, b2) {
            (Full(ref full1), Full(ref full2)) => {

            },
            (Compressed(ref compr), Full(ref full)) |
            (Full(ref full), Compressed(ref compr)) => {

            },
            (Compressed(ref compr1), Compressed(ref compr2)) => {
            },
        }
        0
    }

    pub fn compress_into(b1: &FullBitset, compr: &mut CompressedBitset) {
        unimplemented!()
    }
}









// ------------------------------------------------------------------------------------------------

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn full_basic() {
        let bs = FullBitset::ones(3);
        assert_eq!(bs.get_bit(0), true);
        assert_eq!(bs.get_bit(1), true);
        assert_eq!(bs.get_bit(2), true);
        assert_eq!(bs.get_bit(3), false);

        assert_eq!(bs.len(), 1);
    }
}
