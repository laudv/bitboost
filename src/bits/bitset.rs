//use std::fmt::{Debug, Result as FmtResult, Formatter};
use std::ops::{Deref};

use bits::BitVec;

/// An immutable BitVec with a buffered true-count.
pub struct BitSet {
    vec: BitVec,
    true_count: u64,
    used_nbits: u64,
}

impl BitSet {

    pub fn falses(nbits: usize) -> BitSet {
        let vec = BitVec::zero_bits(nbits);
        Self::from_bitvec(nbits, vec)
    }

    pub fn trues(nbits: usize) -> BitSet {
        let vec = BitVec::one_bits(nbits);
        Self::from_bitvec(nbits, vec)
    }

    pub fn from_bool_iter<I>(nbits: usize, iter: I) -> BitSet
    where I: Iterator<Item = bool> {
        let vec = BitVec::from_bool_iter(iter);
        Self::from_bitvec(nbits, vec)
    }

    pub fn from_bitvec(nbits: usize, vec: BitVec) -> BitSet {
        let true_count = vec.count_ones();
        BitSet {
            vec: vec,
            true_count: true_count,
            used_nbits: nbits as u64,
        }
    }

    pub fn random(nbits: usize, frac1: f64) -> BitSet {
        let bitvec = BitVec::random(nbits, frac1);
        Self::from_bitvec(nbits, bitvec)
    }

    pub fn true_count(&self) -> u64 { self.true_count }
    pub fn false_count(&self) -> u64 { self.used_nbits - self.true_count }
    pub fn len(&self) -> u64 { self.used_nbits }
    pub fn into_bitvec(self) -> BitVec { self.vec }
}

impl Deref for BitSet {
    type Target = BitVec;
    fn deref(&self) -> &BitVec { &self.vec }
}

//impl Debug for BitSet {
//    fn fmt(&self, f: &mut Formatter) -> FmtResult {
//        writeln!(f, "BitSet[nblocks={}]", self.nblocks())?;
//        for i in 0..self.nblocks() {
//            writeln!(f, "{:7}: {:064b}", i, self.blocks[i])?;
//        }
//        Ok(())
//    }
//}








#[cfg(test)]
mod test {
    use bits::BitSet;

    #[test]
    fn test_bitset() {
        let n = 1024;
        let bs = BitSet::from_bool_iter(n, (0..n).map(|i| i%7==0));
        assert_eq!(147, bs.true_count());
    }
}
