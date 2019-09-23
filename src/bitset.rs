/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

use std::ops::{Deref, DerefMut};

use crate::bitblock::{BitBlock, BitBlocks};




pub struct Bitset {
    blocks: BitBlocks,
}

impl Bitset {
    pub fn zeros(nbits: usize) -> Bitset {
        Bitset {
            blocks: BitBlocks::zero_bits(nbits)
        }
    }

    pub fn ones(nbits: usize) -> Bitset {
        Bitset {
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
}

impl Deref for Bitset {
    type Target = BitBlocks;
    fn deref(&self) -> &BitBlocks {
        &self.blocks
    }
}

impl DerefMut for Bitset {
    fn deref_mut(&mut self) -> &mut BitBlocks {
        &mut self.blocks
    }
}









// ------------------------------------------------------------------------------------------------

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn full_basic() {
        let bs = Bitset::ones(3);
        assert_eq!(bs.get_bit(0), true);
        assert_eq!(bs.get_bit(1), true);
        assert_eq!(bs.get_bit(2), true);
        assert_eq!(bs.get_bit(3), false);

        assert_eq!(bs.len(), 1);
    }
}
