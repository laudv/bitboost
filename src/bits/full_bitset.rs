use std::fmt::{Debug, Result as FmtResult, Formatter};

use bits::bitblock::{BitBlockOps, BitBlock};
use bits::util::Bool2BlockIter;
use bits::{Mask, MaskMut};


/// A bitset that stores all bits explicitly
pub struct FullBitSet {
    blocks: Vec<BitBlock>,
}

impl FullBitSet {
    pub fn new(nbits: usize) -> FullBitSet {
        let nblocks = BitBlock::nblocks(nbits);
        FullBitSet {
            blocks: vec![0; nblocks]
        }
    }

    pub fn from_block_iter<I>(iter: I) -> FullBitSet
    where I: Iterator<Item = BitBlock> {
        FullBitSet {
            blocks: iter.collect()
        }
    }

    pub fn from_bool_iter<I>(iter: I) -> FullBitSet
    where I: Iterator<Item = bool> {
        Self::from_block_iter(Bool2BlockIter::new(iter))
    }

    pub fn get_bit(&self, index: usize) -> bool {
        let (block_index, block_bitpos) = Self::get_block_indices(index);
        self.blocks[block_index].get_bit(block_bitpos)
    }

    pub fn set_bit(&mut self, index: usize, value: bool) {
        let (block_index, block_bitpos) = Self::get_block_indices(index);
        let block = &mut self.blocks[block_index];
        *block = block.set_bit(block_bitpos, value);
    }

    fn get_block_indices(index: usize) -> (usize, usize) {
        let block_size = BitBlock::nbits();
        let block_index = index / block_size;
        let block_bitpos = index % block_size;
        (block_index, block_bitpos)
    }
}

impl Mask for FullBitSet {
    fn nblocks(&self) -> usize { self.blocks.len() }
    fn nstored(&self) -> usize { self.blocks.len() }

    #[inline(always)]
    unsafe fn get_index_unchecked(&self, index: usize) -> usize {
        index
    }

    unsafe fn get_block_unchecked(&self, index: usize) -> BitBlock {
        debug_assert!(index < self.nstored(), "out of bounds {} < {}", index, self.nstored());
        *self.blocks.get_unchecked(index)
    }
}

impl MaskMut for FullBitSet {
    unsafe fn set_index_unchecked(&mut self, index: usize, value: BitBlock) {
        debug_assert!(index < self.nstored(), "out of bounds {} < {}", index, self.nstored());
        let block = &mut self.blocks[index];
        *block = value;
    }
}

impl Debug for FullBitSet {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        writeln!(f, "FullBitSet[nblocks={}]", self.nblocks())?;
        for i in 0..self.nblocks() {
            writeln!(f, "{:7}: {:064b}", i, self.blocks[i])?;
        }
        Ok(())
    }
}








#[cfg(test)]
mod test {
    use bits::{FullBitSet};
    use bits::bitblock::{BitBlockOps, BitBlock};
    use bits::{Mask, MaskMut};

    #[test]
    fn test_fullbitset_new() {
        let bi = FullBitSet::new(10);
        assert!(bi.nblocks() == 10 / BitBlock::nbits() + 1);
        let bi = FullBitSet::new(64);
        assert!(bi.nblocks() == 64 / BitBlock::nbits());
        let bi = FullBitSet::new(65);
        assert!(bi.nblocks() == 64 / BitBlock::nbits() + 1);
    }

    #[test]
    fn test_fullbitset_basic() {
        let mut bi = FullBitSet::new(100);
        bi.set_bit(10, true);
        assert!(bi.get_bit(10) == true);
        assert!(bi.get_block(0) == 0x400);
        bi.set_bit(0, true);
        assert!(bi.get_bit(0) == true);
        bi.set_bit(1, true);
        assert!(bi.get_bit(1) == true);
        bi.set_bit(63, true);
        assert!(bi.get_bit(63) == true);
        bi.set_bit(64, true);
        assert!(bi.get_bit(64) == true);
        assert!(bi.get_block(0) == 0x8000000000000403);
        assert!(bi.get_block(1) == 0x1);
        bi.set_bit(63, false);
        assert!(bi.get_bit(63) == false);
        bi.set_bit(64, false);
        assert!(bi.get_bit(64) == false);
        assert!(bi.get_bit(1) == true);
        assert!(bi.get_block(0) == 0x403);
        assert!(bi.get_block(1) == 0x0);
        bi.set_bit(127, true);
        assert!(bi.get_bit(127) == true);
    }

    #[test]
    #[should_panic]
    fn test_fullbitset_out_of_bounds1() {
        let bi = FullBitSet::new(64);
        bi.get_block(2);
    }

    #[test]
    fn test_from_iter() {
        let f = |i| i%64 == 0;
        let iter = (0..129).map(f);
        let bi = FullBitSet::from_bool_iter(iter);
        assert_eq!(bi.nblocks(), 129 / BitBlock::nbits() + 1);
        assert_eq!(bi.get_block(0), 0b1);
        assert_eq!(bi.get_block(1), 0b1);
        assert_eq!(bi.get_block(2), 0b1);
        for i in 0..129 {
            assert_eq!(bi.get_bit(i), f(i));
        }
    }

    #[test]
    fn test_bit_index_vec() {
        let bi = FullBitSet::from_bool_iter((0..128).map(|i| i%7==0));
        let l = bi.bit_index_vec();
        for (a, b) in l.into_iter().zip((0..128).filter(|i| i%7==0)) {
            assert_eq!(a, b);
        }
    }
}
