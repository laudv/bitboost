mod sum_simd;

mod bitblock;
mod bitvec;
mod bitset;
mod bitslice;

pub use self::bitblock::BitBlock;
pub use self::bitvec::BitVec;
pub use self::bitset::BitSet;
pub use self::bitslice::BitSlice;

//pub use self::bitblock::{BitBlockOps, BitBlock};
//pub use self::bitset::BitSet;
//pub use self::bitslice::{BitSlice2, BitSlice4};
//
//pub trait Mask: Sized {
//    /// The number of blocks that are actually stored
//    fn nstored(&self) -> usize;
//
//    /// The total number of blocks, also the unstored blocks.
//    fn nblocks(&self) -> usize;
//
//    unsafe fn get_index_unchecked(&self, i: usize) -> usize;
//    unsafe fn get_block_unchecked(&self, i: usize) -> BitBlock;
//
//    fn get_index(&self, i: usize) -> usize {
//        if i >= self.nblocks() { panic!("index out of bounds"); }
//        unsafe { self.get_index_unchecked(i) }
//    }
//
//    fn get_block(&self, i: usize) -> BitBlock {
//        if i >= self.nblocks() { panic!("index out of bounds"); }
//        unsafe { self.get_block_unchecked(i) }
//    }
//
//    /// Generic iterator, don't use when performance is important
//    fn iter<'a>(&'a self) -> BitSetIter<'a, Self> {
//        BitSetIter { set: &self, stored_index: 0, block_index: 0 }
//    }
//
//    /// Generate a list of indices at which true bits are stored.
//    fn bit_index_vec(&self) -> Vec<usize> {
//        let mut res = Vec::new();
//        let mut index = 0;
//        for block in self.iter() {
//            for i in 0..BitBlock::nbits() {
//                if block.get_bit(i) { res.push(index); }
//                index += 1;
//            }
//        }
//        res
//    }
//}
//
//pub trait MaskMut: Mask {
//    unsafe fn set_index_unchecked(&mut self, i: usize, block: BitBlock);
//
//    fn set_index(&mut self, i: usize, block: BitBlock) {
//        if i >= self.nblocks() { panic!("index out of bounds"); }
//        unsafe { self.set_index_unchecked(i, block); }
//    }
//}
//
//pub struct BitSetIter<'a, T>
//where T: 'a + Mask {
//    set: &'a T,
//    stored_index: usize,
//    block_index: usize,
//}
//
//impl<'a, T> Iterator for BitSetIter<'a, T>
//where T: 'a + Mask {
//    type Item = BitBlock;
//    fn next(&mut self) -> Option<BitBlock> {
//        if self.stored_index >= self.set.nstored() { None }
//        else if self.set.get_index(self.stored_index) > self.block_index {
//            self.block_index += 1;
//            Some(0)
//        } else {
//            self.block_index += 1;
//            self.stored_index += 1;
//            Some(self.set.get_block(self.stored_index - 1))
//        }
//    }
//}
//
//
//
//pub trait ValueStore {
//    type Value: Copy;
//
//    fn nvalues(&self) -> usize;
//
//    fn get_value(&self, index: usize) -> Self::Value;
//    fn set_value(&mut self, index: usize, value: Self::Value);
//
//    /// Sum all value for which a 1 is present in the given mask.
//    fn sum_full(&self, mask: &BitSet) -> Self::Value;
//
//    ///// Generate a mask that contains 1's for equal values, and 0's for others.
//    //fn test_equal<M>(&self, value: Self::Value, m: &mut M) where M: MaskMut;
//
//    ///// Generate a mask that contains 1's for 'less than' values, and 0's otherwise.
//    //fn test_lt<M>(&self, value: Self::Value, m: &mut M) where M: MaskMut;
//}
